"""
Training script — Xception DeepFake Detector.
Адаптирован под несбалансированный датасет 1:3 (real:fake).

	train: 750 real  + 2250 fake = 3000
	val:   150 real  +  450 fake =  600
	test:  100 real  +  300 fake =  400

Стратегия борьбы с дисбалансом:
	1. CrossEntropyLoss(weight=[w_fake, w_real]) — real получает в 3x больший штраф
	2. shuffle=True в train (без WeightedSampler) — модель видит реальное распределение
	3. Оптимальный порог подбирается по val F1 после обучения (обычно 0.30–0.40)
	4. Ранний останов и выбор best checkpoint по AUC-ROC (не зависит от порога)
	5. Честные метрики: balanced_accuracy + MCC вместо accuracy

Hardware: RTX 4070 Super (12 GB VRAM)
"""

import json, os, time, argparse, warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np

from xception_model import build_xception
from imbalance_utils import (
		build_weighted_criterion,
		build_train_loader,
		compute_metrics,
		print_metrics,
		metrics_to_dict,
		find_optimal_threshold,
		get_transforms,
)

warnings.filterwarnings("ignore")


def get_args():
		p = argparse.ArgumentParser(description="Train Xception — imbalanced 1:3 dataset")
		p.add_argument("--data_dir",     type=str,   default="deepfake_dataset_1000")
		p.add_argument("--output_dir",   type=str,   default="checkpoints_xception")
		p.add_argument("--epochs",       type=int,   default=60)
		p.add_argument("--batch_size",   type=int,   default=16)
		p.add_argument("--lr",           type=float, default=1e-4)
		p.add_argument("--weight_decay", type=float, default=1e-4)
		p.add_argument("--dropout",      type=float, default=0.5)
		p.add_argument("--img_size",     type=int,   default=299)
		p.add_argument("--num_workers",  type=int,   default=0)
		p.add_argument("--seed",         type=int,   default=42)
		p.add_argument("--amp",          action="store_true", default=True)
		p.add_argument("--early_stop",   type=int,   default=12)
		p.add_argument("--no_test",      action="store_true", default=False)
		# При сильном дисбалансе иногда помогает WeightedSampler — флаг для экспериментов
		p.add_argument("--weighted_sampler", action="store_true", default=False,
									 help="Use WeightedRandomSampler instead of class_weight in loss")
		return p.parse_args()


def build_dataloaders(data_dir, img_size, batch_size, num_workers, weighted_sampler):
		train_ds = datasets.ImageFolder(
				os.path.join(data_dir, "train"), transform=get_transforms(img_size, "train", "xception"),
		)
		val_ds = datasets.ImageFolder(
				os.path.join(data_dir, "val"), transform=get_transforms(img_size, "val", "xception"),
		)
		test_dir = os.path.join(data_dir, "test")
		test_ds  = datasets.ImageFolder(
				test_dir, transform=get_transforms(img_size, "test", "xception"),
		) if os.path.isdir(test_dir) else None

		class_counts = np.bincount([s[1] for s in train_ds.samples])

		train_loader = build_train_loader(train_ds, batch_size, num_workers, weighted_sampler)
		val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
															num_workers=num_workers, pin_memory=True)
		test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
															num_workers=num_workers, pin_memory=True) if test_ds else None

		print(f"\n{'='*62}")
		print(f"  Model   : Xception  |  img={img_size}×{img_size}")
		print(f"  Dataset : {data_dir}")
		print(f"  Classes : {train_ds.class_to_idx}")
		print(f"  Train   : {len(train_ds)} ({dict(zip(train_ds.classes, class_counts))})")
		print(f"  Val     : {len(val_ds)}  |  Test: {len(test_ds) if test_ds else 'N/A'}")
		print(f"  Imbalance ratio: 1:{int(class_counts.max()/class_counts.min())} "
					f"— using class_weight loss + optimal threshold")
		print(f"  Sampler : {'WeightedSampler' if weighted_sampler else 'shuffle (class_weight in loss)'}")
		print(f"{'='*62}\n")

		return train_loader, val_loader, test_loader, train_ds.class_to_idx, class_counts


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
		model.train()
		running_loss, total = 0.0, 0
		for imgs, labels in loader:
				imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
				optimizer.zero_grad(set_to_none=True)
				with autocast(enabled=use_amp):
						loss = criterion(model(imgs), labels)
				scaler.scale(loss).backward()
				scaler.unscale_(optimizer)
				nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				scaler.step(optimizer); scaler.update()
				running_loss += loss.item() * imgs.size(0)
				total += imgs.size(0)
		return running_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
		model.eval()
		loss_sum = 0.0
		all_labels, all_preds, all_probs = [], [], []
		for imgs, labels in loader:
				imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
				with autocast(enabled=use_amp):
						logits = model(imgs)
						loss_sum += criterion(logits, labels).item() * imgs.size(0)
				probs = torch.softmax(logits, dim=1)
				all_labels.extend(labels.cpu().numpy())
				all_preds.extend(probs.argmax(1).cpu().numpy())
				all_probs.extend(probs.cpu().numpy())
		return loss_sum / len(loader.dataset), all_labels, all_preds, all_probs


def main():
		args = get_args()
		torch.manual_seed(args.seed); np.random.seed(args.seed)
		output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if device.type == "cuda":
				torch.backends.cudnn.benchmark = True
				print(f"  GPU : {torch.cuda.get_device_name(0)}")
				print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

		train_loader, val_loader, test_loader, class_to_idx, class_counts = build_dataloaders(
				args.data_dir, args.img_size, args.batch_size, args.num_workers, args.weighted_sampler,
		)
		idx_to_class = {v: k for k, v in class_to_idx.items()}
		class_names  = [idx_to_class[i] for i in range(len(idx_to_class))]

		model     = build_xception(num_classes=2, dropout_rate=args.dropout).to(device)
		criterion = build_weighted_criterion(class_counts, label_smoothing=0.05, device=device)
		optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
		scaler    = GradScaler(enabled=args.amp)

		best_val_auc, no_improve, history = 0.0, 0, []
		print(f"  Training: {args.epochs} epochs | AMP={'ON' if args.amp else 'OFF'}\n")

		for epoch in range(1, args.epochs + 1):
				t0 = time.time()
				train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args.amp)
				val_loss, vl, vp, vprob = evaluate(model, val_loader, criterion, device, args.amp)
				vm = compute_metrics(vl, vp, vprob, class_names, threshold=0.5)
				scheduler.step()

				print(
						f"Epoch {epoch:03d}/{args.epochs} | Train Loss: {train_loss:.4f} | "
						f"Val Loss: {val_loss:.4f}  AUC: {vm['auc_roc']:.4f}  "
						f"BalAcc: {vm['balanced_accuracy']:.4f}  MCC: {vm['mcc']:.4f} | "
						f"{time.time()-t0:.1f}s"
				)
				history.append({"epoch": epoch, "train_loss": train_loss,
												"val_loss": val_loss, **metrics_to_dict(vm)})

				if vm["auc_roc"] > best_val_auc:
						best_val_auc = vm["auc_roc"]
						torch.save({"epoch": epoch, "state_dict": model.state_dict(),
												"optimizer": optimizer.state_dict(), "val_auc": best_val_auc,
												"class_to_idx": class_to_idx, "args": vars(args)},
											 output_dir / "xception_best.pth")
						print(f"  ✓ Saved best  (val AUC={best_val_auc:.4f})")
						no_improve = 0
				else:
						no_improve += 1

				if args.early_stop > 0 and no_improve >= args.early_stop:
						print(f"  Early stopping after {no_improve} epochs.")
						break

		torch.save({"epoch": epoch, "state_dict": model.state_dict(),
								"class_to_idx": class_to_idx}, output_dir / "xception_last.pth")

		# ── Финальная оценка с оптимальным порогом ────────────────────────────────
		best_ckpt = torch.load(output_dir / "xception_best.pth", map_location=device)
		model.load_state_dict(best_ckpt["state_dict"])
		best_epoch = best_ckpt["epoch"]

		# Ищем оптимальный порог по val
		
		# Claude code----------------------------------------------------------------
		# _, vl, vp, vprob = evaluate(model, val_loader, criterion, device, args.amp)
		# opt_threshold = find_optimal_threshold(np.array(vl), np.array(vprob), metric="f1")
		# print(f"\n  Optimal threshold (val F1): {opt_threshold:.3f}  "
		#       f"(default 0.5 → bias corrected for 1:3 imbalance)")
		#----------------------------------------------------------------------------

		# Qwen-----------------------------------------------------------------------
		_, vl, vp, vprob = evaluate(model, val_loader, criterion, device, args.amp)
		
		# 1. Строго 1D-вектор меток (убираем лишние скобки/столбцы)
		labels_1d = np.array(vl).ravel()
		
		# 2. Работаем с вероятностями:
		#    Если модель вернула (N, 2) → берём столбец класса 1 ("real")
		#    Если уже (N,) → просто выпрямляем
		vprob_np = np.array(vprob)
		if vprob_np.ndim == 2 and vprob_np.shape[1] == 2:
				proba_1d = vprob_np[:, 1]  # вероятность positive class (real)
		else:
				proba_1d = vprob_np.ravel()
				
		# 3. Передаём только 1D-массивы
		opt_threshold = find_optimal_threshold(labels_1d, proba_1d, metric="f1")
		
		print(f"\n  Optimal threshold (val F1): {opt_threshold:.3f}  "
					f"(default 0.5 → bias corrected for 1:3 imbalance)")
		#----------------------------------------------------------------------------

		print(f"\n  {'═'*58}")
		print(f"  FINAL VAL  (epoch {best_epoch}, threshold={opt_threshold:.3f})")
		final_val_m = compute_metrics(vl, vp, vprob, class_names, threshold=opt_threshold)
		print_metrics(final_val_m, class_names, "VAL", best_epoch)

		final_test_m = None
		if test_loader and not args.no_test:
				_, tl, tp_, tprob = evaluate(model, test_loader, criterion, device, args.amp)
				print(f"\n  {'═'*58}")
				print(f"  FINAL TEST  (epoch {best_epoch}, threshold={opt_threshold:.3f})")
				final_test_m = compute_metrics(tl, tp_, tprob, class_names, threshold=opt_threshold)
				print_metrics(final_test_m, class_names, "TEST", best_epoch)

		results = {
				"model": "Xception", "best_epoch": best_epoch,
				"optimal_threshold": opt_threshold,
				"val_metrics":  metrics_to_dict(final_val_m),
				"test_metrics": metrics_to_dict(final_test_m) if final_test_m else None,
				"imbalance_note": "1:3 ratio — use balanced_accuracy and MCC for fair comparison",
				"history": history, "args": vars(args),
		}
		rp = output_dir / "results.json"
		with open(rp, "w") as f: json.dump(results, f, indent=2)

		print(f"\n{'═'*62}")
		print(f"  Xception complete  |  best epoch: {best_epoch}  |  threshold: {opt_threshold:.3f}")
		print(f"  Val  AUC={final_val_m['auc_roc']:.4f}  BalAcc={final_val_m['balanced_accuracy']:.4f}  MCC={final_val_m['mcc']:.4f}")
		if final_test_m:
				print(f"  Test AUC={final_test_m['auc_roc']:.4f}  BalAcc={final_test_m['balanced_accuracy']:.4f}  MCC={final_test_m['mcc']:.4f}")
		print(f"  Results: {rp}")
		print(f"{'═'*62}\n")


if __name__ == "__main__":
		main()