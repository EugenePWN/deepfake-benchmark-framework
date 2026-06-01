"""
Training script — EfficientNet-B4 DeepFake Detector.
Адаптирован под несбалансированный датасет 1:3 (real:fake).

Двухфазное обучение:
  Фаза 1: backbone заморожен, обучается только голова (5-10 эпох)
  Фаза 2: fine-tune всего backbone с дифференциальными lr

RTX 4070 Super: batch_size=16, img_size=380
A100:           batch_size=32-64
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

from efficientnet_model import build_efficientnet_b4
from imbalance_utils import (
    build_weighted_criterion, build_train_loader,
    compute_metrics, print_metrics, metrics_to_dict,
    find_optimal_threshold, get_transforms,
)

warnings.filterwarnings("ignore")


def get_args():
    p = argparse.ArgumentParser(description="Train EfficientNet-B4 — imbalanced 1:3 dataset")
    p.add_argument("--data_dir",        type=str,   default="deepfake_dataset_1000")
    p.add_argument("--output_dir",      type=str,   default="checkpoints_effnet")
    p.add_argument("--img_size",        type=int,   default=380)
    p.add_argument("--phase1_epochs",   type=int,   default=10)
    p.add_argument("--epochs",          type=int,   default=40)
    p.add_argument("--lr_head",         type=float, default=1e-3)
    p.add_argument("--lr_backbone",     type=float, default=1e-5)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--dropout",         type=float, default=0.4)
    p.add_argument("--unfreeze_blocks", type=int,   default=None)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--num_workers",     type=int,   default=0)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--amp",             action="store_true", default=True)
    p.add_argument("--early_stop",      type=int,   default=10)
    p.add_argument("--no_test",         action="store_true", default=False)
    p.add_argument("--weighted_sampler",action="store_true", default=False)
    return p.parse_args()


def build_dataloaders(data_dir, img_size, batch_size, num_workers, weighted_sampler):
    train_ds = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=get_transforms(img_size, "train", "efficientnet"),
    )
    val_ds = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=get_transforms(img_size, "val", "efficientnet"),
    )
    test_dir = os.path.join(data_dir, "test")
    test_ds  = datasets.ImageFolder(
        test_dir, transform=get_transforms(img_size, "test", "efficientnet"),
    ) if os.path.isdir(test_dir) else None

    class_counts = np.bincount([s[1] for s in train_ds.samples])
    train_loader = build_train_loader(train_ds, batch_size, num_workers, weighted_sampler)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True) if test_ds else None

    print(f"\n{'='*62}")
    print(f"  Model   : EfficientNet-B4  |  img={img_size}×{img_size}")
    print(f"  Dataset : {data_dir}")
    print(f"  Classes : {train_ds.class_to_idx}")
    print(f"  Train   : {len(train_ds)} ({dict(zip(train_ds.classes, class_counts))})")
    print(f"  Val     : {len(val_ds)}  |  Test: {len(test_ds) if test_ds else 'N/A'}")
    print(f"  Imbalance: 1:{int(class_counts.max()/class_counts.min())} "
          f"— class_weight loss + ImageNet normalization")
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


def run_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
              scaler, device, args, output_dir, class_names, phase_name, n_epochs, start_epoch=1):
    best_val_auc, no_improve, history = 0.0, 0, []
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [{phase_name}] {n_epochs} epochs | trainable: {trainable:,}\n")

    for epoch in range(start_epoch, start_epoch + n_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args.amp)
        val_loss, vl, vp, vprob = evaluate(model, val_loader, criterion, device, args.amp)
        vm = compute_metrics(vl, vp, vprob, class_names, threshold=0.5)
        if scheduler: scheduler.step()

        print(f"Epoch {epoch:03d} [{phase_name}] | Loss: {train_loss:.4f} → {val_loss:.4f} | "
              f"AUC: {vm['auc_roc']:.4f}  BalAcc: {vm['balanced_accuracy']:.4f}  MCC: {vm['mcc']:.4f} | "
              f"{time.time()-t0:.1f}s")

        history.append({"epoch": epoch, "phase": phase_name,
                        "train_loss": train_loss, "val_loss": val_loss, **metrics_to_dict(vm)})

        if vm["auc_roc"] > best_val_auc:
            best_val_auc = vm["auc_roc"]
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "val_auc": best_val_auc,
                        "class_to_idx": args._class_to_idx, "args": vars(args)},
                       output_dir / "effnet_best.pth")
            print(f"  ✓ Saved best  (val AUC={best_val_auc:.4f})")
            no_improve = 0
        else:
            no_improve += 1

        if args.early_stop > 0 and no_improve >= args.early_stop:
            print(f"  Early stopping after {no_improve} epochs.")
            break

    return best_val_auc, history


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
    args._class_to_idx = class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names  = [idx_to_class[i] for i in range(len(idx_to_class))]
    scaler = GradScaler(enabled=args.amp)
    all_history = []

    # ── Фаза 1 ───────────────────────────────────────────────────────────────
    if args.phase1_epochs > 0:
        print(f"\n{'━'*62}\n  PHASE 1: head only (backbone frozen)\n{'━'*62}")
        model = build_efficientnet_b4(dropout_rate=args.dropout, pretrained=True,
                                       freeze_backbone=True).to(device)
        criterion = build_weighted_criterion(class_counts, label_smoothing=0.05, device=device)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr_head, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.phase1_epochs, eta_min=1e-6)
        _, h1 = run_phase(model, train_loader, val_loader, criterion, optimizer,
                          scheduler, scaler, device, args, output_dir,
                          class_names, "Phase1", args.phase1_epochs)
        all_history.extend(h1)
    else:
        model = build_efficientnet_b4(dropout_rate=args.dropout, pretrained=True,
                                       freeze_backbone=False).to(device)

    # ── Фаза 2 ───────────────────────────────────────────────────────────────
    print(f"\n{'━'*62}\n  PHASE 2: full fine-tune\n{'━'*62}")
    model.unfreeze_backbone(unfreeze_last_n_blocks=args.unfreeze_blocks)
    criterion = build_weighted_criterion(class_counts, label_smoothing=0.05, device=device)
    optimizer = optim.AdamW(
        model.get_param_groups(lr_backbone=args.lr_backbone, lr_head=args.lr_head),
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    _, h2 = run_phase(model, train_loader, val_loader, criterion, optimizer,
                      scheduler, scaler, device, args, output_dir,
                      class_names, "Phase2", args.epochs, args.phase1_epochs + 1)
    all_history.extend(h2)

    # ── Финальная оценка ──────────────────────────────────────────────────────
    best_ckpt = torch.load(output_dir / "effnet_best.pth", map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
    best_epoch = best_ckpt["epoch"]

    # Claude code----------------------------------------------------------------
    # _, vl, vp, vprob = evaluate(model, val_loader, criterion, device, args.amp)
    # opt_threshold = find_optimal_threshold(np.array(vl), np.array(vprob), metric="f1")
    # print(f"\n  Optimal threshold: {opt_threshold:.3f}")
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

    print(f"\n  {'═'*58}\n  FINAL VAL  (epoch {best_epoch}, threshold={opt_threshold:.3f})")
    final_val_m = compute_metrics(vl, vp, vprob, class_names, threshold=opt_threshold)
    print_metrics(final_val_m, class_names, "VAL", best_epoch)

    final_test_m = None
    if test_loader and not args.no_test:
        _, tl, tp_, tprob = evaluate(model, test_loader, criterion, device, args.amp)
        print(f"\n  {'═'*58}\n  FINAL TEST  (epoch {best_epoch}, threshold={opt_threshold:.3f})")
        final_test_m = compute_metrics(tl, tp_, tprob, class_names, threshold=opt_threshold)
        print_metrics(final_test_m, class_names, "TEST", best_epoch)

    torch.save({"epoch": best_epoch, "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx}, output_dir / "effnet_last.pth")

    results = {
        "model": "EfficientNet-B4", "best_epoch": best_epoch,
        "optimal_threshold": opt_threshold,
        "val_metrics":  metrics_to_dict(final_val_m),
        "test_metrics": metrics_to_dict(final_test_m) if final_test_m else None,
        "imbalance_note": "1:3 ratio — use balanced_accuracy and MCC for fair comparison",
        "history": all_history, "args": vars(args),
    }
    rp = output_dir / "results.json"
    with open(rp, "w") as f: json.dump(results, f, indent=2)

    print(f"\n{'═'*62}")
    print(f"  EfficientNet-B4  |  best epoch: {best_epoch}  |  threshold: {opt_threshold:.3f}")
    print(f"  Val  AUC={final_val_m['auc_roc']:.4f}  BalAcc={final_val_m['balanced_accuracy']:.4f}  MCC={final_val_m['mcc']:.4f}")
    if final_test_m:
        print(f"  Test AUC={final_test_m['auc_roc']:.4f}  BalAcc={final_test_m['balanced_accuracy']:.4f}  MCC={final_test_m['mcc']:.4f}")
    print(f"  Results: {rp}\n{'═'*62}\n")


if __name__ == "__main__":
    main()