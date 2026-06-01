# deepfake_benchmark/utils/detect_dataset.py
"""
Запуск детекторов на датасете с manifest.json.

Поддерживает два режима:

  Режим А — через manifest.json (старый, рекомендуется для датасетов собранных dataset_build):
    python -m deepfake_benchmark.utils.detect_dataset \\
        --dataset_dir data/deepfake_dataset \\
        --split test \\
        --checkpoints xception:checkpoints_xception/xception_best.pth:0.700 \\
                      efficientnet:checkpoints_effnet/effnet_best.pth:0.588 \\
                      f3net:checkpoints_f3net/f3net_best.pth:0.516 \\
        --out data/results/metrics_test.json

  Режим Б — через папки real/ + fake/ (без manifest):
    python -m deepfake_benchmark.utils.detect_dataset \\
        --real_dir data/deepfake_dataset/test/real \\
        --fake_dir data/deepfake_dataset/test/fake \\
        --checkpoints xception:checkpoints_xception/xception_best.pth:0.700

  Режим В — через Python-конфиг (старый API, для совместимости):
    python -m deepfake_benchmark.utils.detect_dataset \\
        --config configs/dataset_celeba_vggface2.py \\
        --dataset_dir data/deepfake_dataset \\
        --split test

Формат --checkpoints:
    <name>:<weights_path>:<threshold>
    Threshold опционален (по умолчанию 0.5 → авто-подбор по F1).
    Пример: xception:checkpoints/xception_best.pth:0.700
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# Архитектура определяется по имени (можно переопределить через --arch)
_ARCH_BY_NAME = {
    "xception":    "xception",
    "efficientnet": "efficientnet",
    "effnet":      "efficientnet",
    "f3net":       "f3net",
}

_ARCH_DEFAULTS = {
    "xception":    {"img_size": 299, "mean": [0.5]*3, "std": [0.5]*3},
    "efficientnet": {"img_size": 380, "mean": [0.485, 0.456, 0.406],
                     "std": [0.229, 0.224, 0.225]},
    "f3net":       {"img_size": 299, "mean": [0.5]*3, "std": [0.5]*3},
}


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка модели
# ──────────────────────────────────────────────────────────────────────────────

def _detect_architecture(name: str, weights_path: Path) -> str:
    """Определяет архитектуру по имени детектора или имени файла."""
    name_lower = name.lower()
    for key, arch in _ARCH_BY_NAME.items():
        if key in name_lower:
            return arch
    # По имени файла
    stem = weights_path.stem.lower()
    for key, arch in _ARCH_BY_NAME.items():
        if key in stem:
            return arch
    raise ValueError(
        f"Cannot detect architecture for '{name}' ({weights_path.name}). "
        f"Add --arch {name}:xception (or efficientnet/f3net)."
    )


def _load_model(architecture: str, weights_path: Path, device):
    import torch
    ckpt  = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    class_to_idx: Dict[str, int] = ckpt.get("class_to_idx", {"fake": 0, "real": 1})

    if architecture == "xception":
        from deepfake_benchmark.models.xception_model import build_xception
        model = build_xception(num_classes=2, dropout_rate=0.5)
    elif architecture == "efficientnet":
        from deepfake_benchmark.models.efficientnet_model import build_efficientnet_b4
        model = build_efficientnet_b4(num_classes=2, dropout_rate=0.4, pretrained=False)
    elif architecture == "f3net":
        from deepfake_benchmark.models.f3net_model import build_f3net
        model = build_f3net(num_classes=2, dropout_rate=0.5)
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}")

    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    fake_class_idx = class_to_idx.get("fake", 0)
    return model, fake_class_idx


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка данных
# ──────────────────────────────────────────────────────────────────────────────

def _load_items_from_manifest(
    manifest_path: Path,
    split: str = "test",
    allowed_presets: Optional[Set[str]] = None,
) -> List[dict]:
    """Загружает записи из manifest.json."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if split not in manifest:
        raise KeyError(
            f"Split {split!r} not in manifest. Available: {list(manifest)}"
        )

    rows = []
    for row in manifest[split]:
        label = row.get("label")
        if label not in {"real", "fake"}:
            continue
        preset = row.get("preset")
        if label == "fake" and allowed_presets and preset not in allowed_presets:
            continue
        path = Path(row["path"]).expanduser().resolve()
        if not path.exists():
            logger.warning("Missing file: %s", path)
            continue
        rows.append({**row, "path": path, "label": label})
    return rows


def _load_items_from_dirs(real_dir: Path, fake_dir: Path, n_per_class: Optional[int]) -> List[dict]:
    """Загружает изображения напрямую из папок real/ и fake/."""
    import random
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def _collect(folder: Path, label: str) -> List[dict]:
        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if n_per_class:
            paths = random.Random(42).sample(paths, min(n_per_class, len(paths)))
        return [{"path": p, "label": label, "preset": "overall"} for p in paths]

    return _collect(real_dir, "real") + _collect(fake_dir, "fake")


# ──────────────────────────────────────────────────────────────────────────────
# Инференс
# ──────────────────────────────────────────────────────────────────────────────

def _run_inference(
    model,
    rows: List[dict],
    architecture: str,
    fake_class_idx: int,
    threshold: float,
    batch_size: int,
    device,
    use_amp: bool,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Возвращает (gt_labels, pred_labels, prob_fake).
    gt_labels: 0=real, 1=fake
    """
    import torch
    from torch.cuda.amp import autocast
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from torchvision import transforms

    d = _ARCH_DEFAULTS[architecture]
    transform = transforms.Compose([
        transforms.Resize((d["img_size"], d["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(d["mean"], d["std"]),
    ])

    class _DS(Dataset):
        def __init__(self, rows, transform):
            self.rows = rows; self.transform = transform
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            row = self.rows[i]
            img = Image.open(row["path"]).convert("RGB")
            return self.transform(img), 1 if row["label"] == "fake" else 0

    loader = DataLoader(_DS(rows, transform), batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=use_amp)

    all_gt, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_gt.extend(labels.numpy())
            all_probs.extend(probs[:, fake_class_idx])

    all_preds = [1 if p >= threshold else 0 for p in all_probs]
    return all_gt, all_preds, all_probs


# ──────────────────────────────────────────────────────────────────────────────
# Метрики (без sklearn)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_metrics(gt, preds, probs, threshold):
    import math
    n  = len(gt)
    tp = sum(p==1 and g==1 for p,g in zip(preds,gt))
    tn = sum(p==0 and g==0 for p,g in zip(preds,gt))
    fp = sum(p==1 and g==0 for p,g in zip(preds,gt))
    fn = sum(p==0 and g==1 for p,g in zip(preds,gt))
    eps = 1e-9
    prec = tp/(tp+fp+eps); rec = tp/(tp+fn+eps)
    spec = tn/(tn+fp+eps)
    f1   = 2*prec*rec/(prec+rec+eps)
    bacc = (rec+spec)/2
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc  = (tp*tn-fp*fn)/denom if denom else 0.0
    # AUC-ROC (трапеции)
    pairs = sorted(zip(probs, gt), key=lambda x: -x[0])
    n_pos = sum(gt); n_neg = n - n_pos
    auc = tpr = fpr = prev_tpr = prev_fpr = 0.0
    if n_pos > 0 and n_neg > 0:
        for score, label in pairs:
            if label==1: tpr += 1/n_pos
            else: fpr += 1/n_neg
            auc += (fpr-prev_fpr)*(tpr+prev_tpr)/2
            prev_tpr, prev_fpr = tpr, fpr
    return {
        "auc_roc": round(auc,4), "mcc": round(mcc,4),
        "balanced_accuracy": round(bacc,4), "f1": round(f1,4),
        "sensitivity": round(rec,4), "specificity": round(spec,4),
        "precision": round(prec,4), "accuracy": round((tp+tn)/n,4),
        "tp":tp, "tn":tn, "fp":fp, "fn":fn,
        "threshold": threshold, "n_samples": n,
    }


def _print_metrics(name: str, dataset: str, m: dict):
    print(f"\n  {'─'*60}")
    print(f"  {name} | {dataset}")
    print(f"  {'─'*60}")
    print(f"  AUC-ROC     : {m['auc_roc']:.4f}   (threshold-free)")
    print(f"  MCC         : {m['mcc']:.4f}   (best single metric for imbalance)")
    print(f"  Balanced Acc: {m['balanced_accuracy']:.4f}")
    print(f"  F1 (fake)   : {m['f1']:.4f}   threshold={m['threshold']:.3f}")
    print(f"  Sensitivity : {m['sensitivity']:.4f}   (TPR / recall fake)")
    print(f"  Specificity : {m['specificity']:.4f}   (TNR / recall real)")
    print(f"  Confusion   : TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")


# ──────────────────────────────────────────────────────────────────────────────
# Публичная функция (для использования из Benchmark)
# ──────────────────────────────────────────────────────────────────────────────

def run_detection(
    *,
    checkpoint_specs: List[Tuple[str, Path, float]],  # (name, path, threshold)
    rows: List[dict],
    dataset_name: str = "unknown",
    batch_size: int = 16,
    device_str: str = "cuda",
    arch_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, dict]:
    """
    Прогоняет несколько детекторов на rows (из manifest или из папок).

    Args:
        checkpoint_specs: список (name, weights_path, threshold)
        rows:             список dict с ключами path, label, preset
        dataset_name:     имя для отчёта
        batch_size:       размер батча
        device_str:       "cuda" или "cpu"
        arch_overrides:   {"my_model": "xception"} для переопределения архитектуры

    Returns:
        Dict[detector_name → metrics_dict]
    """
    import torch
    device  = torch.device(device_str if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    arch_overrides = arch_overrides or {}

    results = {}

    for name, weights_path, threshold in checkpoint_specs:
        architecture = arch_overrides.get(name) or _detect_architecture(name, weights_path)
        logger.info("[detect_dataset] Loading %s (%s)...", name, architecture)

        try:
            model, fake_class_idx = _load_model(architecture, weights_path, device)
        except Exception as e:
            logger.error("[detect_dataset] Failed to load %s: %s", name, e, exc_info=True)
            continue

        gt, preds, probs = _run_inference(
            model, rows, architecture, fake_class_idx,
            threshold, batch_size, device, use_amp,
        )
        m = _compute_metrics(gt, preds, probs, threshold)
        _print_metrics(name, dataset_name, m)
        results[name] = m

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Совместимость со старым API (для detect_dataset.py из utils)
# ──────────────────────────────────────────────────────────────────────────────

def run_detection_on_dataset(
    *,
    config,
    dataset_dir: Path,
    split: str = "test",
    presets=None,
) -> object:
    """
    Старый API: принимает BenchmarkConfig с настроенными детекторами через Registry.
    Оставлен для обратной совместимости.
    """
    from deepfake_benchmark.core.detector_manager import DetectorManager
    from deepfake_benchmark.core.metric_evaluator import MetricEvaluator
    from deepfake_benchmark.types import SampleItem

    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    allowed_presets = set(presets) if presets else None
    rows = _load_items_from_manifest(manifest_path, split, allowed_presets)

    items = [
        SampleItem(
            sample_id=f"combined_{Path(r['path']).stem}",
            media_path=Path(r["path"]),
            label=r["label"],
            generator="manifest",
            role="generated" if r["label"] == "fake" else None,
            dataset="combined",
            split=split,
            meta={k: v for k, v in r.items()
                  if k not in ("path", "label") and isinstance(v, str)},
        )
        for r in rows
    ]

    detector_mgr = DetectorManager(config)
    detector_mgr.setup()
    detections = detector_mgr.detect(items)

    evaluator = MetricEvaluator(config)
    return evaluator.evaluate(detections)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_checkpoint_spec(s: str) -> Tuple[str, Path, float]:
    """Парсит 'name:path:threshold' или 'name:path'."""
    parts = s.split(":")
    if len(parts) == 3:
        name, path, thr = parts
        return name, Path(path), float(thr)
    elif len(parts) == 2:
        return parts[0], Path(parts[1]), 0.5
    else:
        raise ValueError(f"Invalid checkpoint spec: {s!r}. Expected 'name:path' or 'name:path:threshold'")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m deepfake_benchmark.utils.detect_dataset",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Источник данных
    src = p.add_mutually_exclusive_group()
    src.add_argument("--dataset_dir", type=Path, default=None,
                     help="Папка с manifest.json (создаётся dataset_build.py)")
    src.add_argument("--real_dir", type=Path, default=None,
                     help="Папка с реальными изображениями (без manifest)")

    p.add_argument("--fake_dir",  type=Path, default=None,
                   help="Папка с фейками (нужна вместе с --real_dir)")
    p.add_argument("--split",     type=str, default="test",
                   choices=("train", "val", "test"),
                   help="Сплит из manifest.json (только для --dataset_dir)")
    p.add_argument("--presets",   nargs="+", default=None,
                   help="Фильтр пресетов фейков: easy medium hard")
    p.add_argument("--n_per_class", type=int, default=None,
                   help="Лимит изображений на класс (только для --real_dir)")

    # Детекторы
    p.add_argument("--checkpoints", nargs="+", required=False, default=None,
                   metavar="NAME:PATH[:THRESHOLD]",
                   help="Детекторы: 'xception:ckpt/best.pth:0.700 efficientnet:...'")
    p.add_argument("--arch", nargs="+", default=None,
                   metavar="NAME:ARCH",
                   help="Переопределить архитектуру: 'my_model:xception'")

    # Старый API (совместимость)
    p.add_argument("--config", type=Path, default=None,
                   help="Python-конфиг с make_config() — старый API")
    p.add_argument("--config_preset", type=str, default=None)

    # Прочее
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device",     type=str, default="cuda",
                   choices=("cuda", "cpu"))
    p.add_argument("--out", type=Path, default=None,
                   help="Путь для сохранения JSON с метриками")

    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args(argv)

    # ── Старый API ────────────────────────────────────────────────────────────
    if args.config and not args.checkpoints:
        import importlib.util
        spec   = importlib.util.spec_from_file_location(args.config.stem, args.config)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = module.make_config(args.config_preset) \
            if args.config_preset else module.make_config()

        report = run_detection_on_dataset(
            config=cfg,
            dataset_dir=Path(args.dataset_dir).resolve(),
            split=args.split,
            presets=args.presets,
        )
        out = args.out or (Path(cfg.results_root) / f"metrics_{args.split}.json")
        report.save_json(out)
        print(f"Saved: {out}")
        return 0

    # ── Новый API: --checkpoints ───────────────────────────────────────────────
    if not args.checkpoints:
        print("ERROR: specify --checkpoints or --config")
        return 1

    checkpoint_specs = [_parse_checkpoint_spec(s) for s in args.checkpoints]

    arch_overrides: Dict[str, str] = {}
    if args.arch:
        for spec in args.arch:
            name, arch = spec.split(":", 1)
            arch_overrides[name] = arch

    # Загружаем данные
    if args.dataset_dir:
        manifest_path = Path(args.dataset_dir) / "manifest.json"
        rows = _load_items_from_manifest(
            manifest_path, args.split,
            set(args.presets) if args.presets else None,
        )
        dataset_name = args.dataset_dir.name
    elif args.real_dir:
        if not args.fake_dir:
            print("ERROR: --fake_dir required with --real_dir")
            return 1
        rows = _load_items_from_dirs(args.real_dir, args.fake_dir, args.n_per_class)
        dataset_name = args.real_dir.parent.name
    else:
        print("ERROR: specify --dataset_dir or --real_dir")
        return 1

    print(f"\nDataset: {dataset_name}  ({sum(1 for r in rows if r['label']=='real')} real + "
          f"{sum(1 for r in rows if r['label']=='fake')} fake)")

    # Прогон
    results = run_detection(
        checkpoint_specs=checkpoint_specs,
        rows=rows,
        dataset_name=dataset_name,
        batch_size=args.batch_size,
        device_str=args.device,
        arch_overrides=arch_overrides,
    )

    # Сохраняем
    out = args.out or Path("data/results") / f"metrics_{dataset_name}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_name, "results": results}, f, indent=2)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())