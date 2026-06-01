from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class SplitBuildStats:
    n_real: int
    n_fake: int


def _iter_images(root: Path, patterns: Sequence[str] = ("*.jpg", "*.jpeg", "*.png")) -> Iterable[Path]:
    for pat in patterns:
        yield from root.glob(pat)


def _parse_target_stem(fake_path: Path) -> Optional[str]:
    """
    Expected fake filename format from this project:
      "<src_sample_id>_to_<tgt_sample_id>.jpg"
    We use target part to assign fake to the same split as its real target.
    """
    stem = fake_path.stem
    parts = stem.split("_to_")
    if len(parts) != 2:
        return None
    return parts[1]


_DATASET_PREFIXES = (
    # Common dataset names used inside this project as SampleItem.dataset / dataset_name
    "custom",
    "celeba",
    "celeba_hq",
    "vggface2",
    "lfw",
    "ffhq",
)


def _expand_stem_variants(stem: str) -> Set[str]:
    """
    SampleItem.sample_id in this project is built as f"{dataset_name}_{path.stem}".
    Therefore target part inside fake filename may look like:
      - "custom_00001"
      - "celeba_hq_00001"
    while real_dir filenames are typically:
      - "00001.jpg"

    Return a set of possible matching variants (original + stripped dataset prefix).
    """
    out = {stem}
    for ds in _DATASET_PREFIXES:
        prefix = f"{ds}_"
        if stem.startswith(prefix) and len(stem) > len(prefix):
            out.add(stem[len(prefix):])
    return out


def _collect_target_stems_from_fakes(
    fakes_dir: Path,
    presets: Sequence[str],
) -> Set[str]:
    """
    Scan fakes_dir/<preset>/<model>/**/*.jpg and collect target stems used in filenames.
    """
    stems: Set[str] = set()
    for preset in presets:
        preset_dir = fakes_dir / preset
        if not preset_dir.exists():
            continue
        for fake_path in preset_dir.rglob("*.jpg"):
            tgt = _parse_target_stem(fake_path)
            if tgt:
                stems.update(_expand_stem_variants(tgt))
    return stems


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def _make_unique_name(
    fake_path: Path,
    preset: str,
    model: str,
    used_names: set[str],
) -> str:
    """
    Deterministic collision-free naming for a flat fake/ folder.
    """
    base = f"{fake_path.stem}__{_safe_name(preset)}__{_safe_name(model)}"
    name = f"{base}{fake_path.suffix.lower()}"
    if name not in used_names:
        used_names.add(name)
        return name

    # Extremely rare: if the same (stem,preset,model) is encountered twice (e.g. duplicates on disk)
    i = 1
    while True:
        name_i = f"{base}__dup{i}{fake_path.suffix.lower()}"
        if name_i not in used_names:
            used_names.add(name_i)
            return name_i
        i += 1


def build_combined_dataset(
    *,
    real_dir: Path,
    fakes_dir: Path,
    output_dir: Path,
    split_ratios: Dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15},
    seed: int = 42,
    presets: Sequence[str] = ("easy", "medium", "hard"),
    copy: bool = True,
    restrict_real_to_used_targets: bool = False,
    max_real_images: Optional[int] = None,
    fill_real_to_max: bool = False,
) -> Dict[str, List[dict]]:
    """
    Build a unified dataset folder:

      output_dir/
        train/real, train/fake
        val/real,   val/fake
        test/real,  test/fake
        manifest.json

    Key properties:
    - Real images are split once; fakes are assigned to the split of their target.
    - All fakes are stored in ONE folder per split (flat).
    - Filename collisions are avoided by deterministic renaming:
        "<src>_to_<tgt>__<preset>__<model>.jpg"
    """
    real_dir = Path(real_dir)
    fakes_dir = Path(fakes_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    real_images = sorted(_iter_images(real_dir))
    if not real_images:
        raise FileNotFoundError(f"No images found in real_dir={real_dir}")

    all_real_images = real_images

    if restrict_real_to_used_targets:
        used_target_stems = _collect_target_stems_from_fakes(fakes_dir, presets)
        if not used_target_stems:
            raise FileNotFoundError(
                f"No fake targets found under fakes_dir={fakes_dir} for presets={list(presets)}"
            )
        real_images = [p for p in real_images if p.stem in used_target_stems]
        if not real_images:
            raise FileNotFoundError(
                "After filtering real_dir by targets used in fakes, no real images remained. "
                "Check that real_dir matches the target dataset used for generation."
            )

    if max_real_images is not None:
        if max_real_images <= 0:
            raise ValueError("max_real_images must be >= 1")
        rng.shuffle(real_images)
        real_images = real_images[: min(max_real_images, len(real_images))]
        real_images = sorted(real_images)
        if fill_real_to_max and len(real_images) < max_real_images:
            missing = max_real_images - len(real_images)
            pool = [p for p in all_real_images if p not in set(real_images)]
            rng.shuffle(pool)
            extra = pool[:missing]
            real_images = sorted([*real_images, *extra])

    total_ratio = sum(split_ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got {total_ratio}")
    if set(split_ratios.keys()) != {"train", "val", "test"}:
        raise ValueError("split_ratios must have keys: train, val, test")

    rng.shuffle(real_images)
    n = len(real_images)
    n_train = int(n * split_ratios["train"])
    n_val = int(n * split_ratios["val"])
    splits: Dict[str, List[Path]] = {
        "train": real_images[:n_train],
        "val": real_images[n_train : n_train + n_val],
        "test": real_images[n_train + n_val :],
    }

    real_stem_to_split = {img.stem: split for split, imgs in splits.items() for img in imgs}

    manifest: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    used_fake_names: Dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    def _transfer(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))

    # Copy reals
    for split, imgs in splits.items():
        split_real_dir = output_dir / split / "real"
        split_fake_dir = output_dir / split / "fake"
        split_real_dir.mkdir(parents=True, exist_ok=True)
        split_fake_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            dst = split_real_dir / img.name
            _transfer(img, dst)
            manifest[split].append({"path": str(dst), "label": "real"})

    # Copy fakes (assigned by target -> split)
    for preset in presets:
        preset_dir = fakes_dir / preset
        if not preset_dir.exists():
            continue

        # Under this project, model is the directory right under preset (preset/model/*.jpg).
        for model_dir in sorted([p for p in preset_dir.iterdir() if p.is_dir()]):
            model = model_dir.name
            for fake_path in model_dir.rglob("*.jpg"):
                target_stem = _parse_target_stem(fake_path)
                if not target_stem:
                    continue
                split = None
                for variant in _expand_stem_variants(target_stem):
                    split = real_stem_to_split.get(variant)
                    if split:
                        break
                if not split:
                    continue

                new_name = _make_unique_name(fake_path, preset=preset, model=model, used_names=used_fake_names[split])
                dst = output_dir / split / "fake" / new_name
                _transfer(fake_path, dst)
                manifest[split].append(
                    {
                        "path": str(dst),
                        "label": "fake",
                        "preset": preset,
                        "model": model,
                        "target_stem": target_stem,
                    }
                )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Convenience stdout-style summary for scripts
    stats: Dict[str, SplitBuildStats] = {}
    for split in ("train", "val", "test"):
        n_real = sum(1 for x in manifest[split] if x["label"] == "real")
        n_fake = sum(1 for x in manifest[split] if x["label"] == "fake")
        stats[split] = SplitBuildStats(n_real=n_real, n_fake=n_fake)

    return manifest


def _parse_split_ratios(values: Sequence[str]) -> Dict[str, float]:
    if len(values) != 3:
        raise ValueError("Expected exactly 3 values for --split (train val test)")
    train, val, test = (float(x) for x in values)
    return {"train": train, "val": val, "test": test}


def _load_benchmark_config_from_py(config_path: Path, preset: Optional[str]):
    config_path = Path(config_path)
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    make_config = getattr(module, "make_config", None)
    if make_config is None or not callable(make_config):
        raise AttributeError(f"{config_path} must define a callable make_config(...)")

    return make_config(preset) if preset is not None else make_config()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m deepfake_benchmark.utils.dataset_build",
        description="Build combined deepfake dataset (train/val/test) with collision-free fake filenames.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional: python config used for generation. If set, real targets are resolved via DatasetLoader (uses N_TARGETS/seed logic).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset passed to make_config(preset). Usually not needed for dataset build.",
    )
    parser.add_argument(
        "--real_dir",
        type=Path,
        default=None,
        help="Directory with real images (manual mode). Required if --config is not provided.",
    )
    parser.add_argument("--fakes_dir", type=Path, required=True, help="Directory with generated fakes (data/fakes).")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting real images.")
    parser.add_argument(
        "--split",
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=("0.70", "0.15", "0.15"),
        help="Split ratios (must sum to 1.0). Example: --split 0.7 0.15 0.15",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=("easy", "medium", "hard"),
        help="Which presets to include from fakes_dir (default: easy medium hard).",
    )
    parser.add_argument(
        "--restrict_real_to_used_targets",
        action="store_true",
        help="Use as real only those images whose stems appear as targets in fakes_dir.",
    )
    parser.add_argument(
        "--max_real",
        type=int,
        default=None,
        help="Limit number of real images (after optional target filtering). Example: --max_real 500",
    )
    parser.add_argument(
        "--fill_real_to_max",
        action="store_true",
        help="If --max_real is set and filtered real images are fewer, fill the remainder with random real images (may introduce real images without corresponding fakes).",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (default: copy).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    split_ratios = _parse_split_ratios(args.split)

    real_dir: Path
    if args.config is not None:
        from deepfake_benchmark.core.dataset_loader import DatasetLoader

        cfg = _load_benchmark_config_from_py(args.config, args.preset)
        loader = DatasetLoader(cfg)
        items = loader.load_all()
        real_paths = sorted({it.media_path.resolve() for it in items if it.role is None})
        if not real_paths:
            raise FileNotFoundError("No real target images found via DatasetLoader (role=None).")
        parents = {p.parent for p in real_paths}
        if len(parents) != 1:
            raise ValueError(
                "Config-mode expects real target images to be in a single directory. "
                f"Found {len(parents)} different parents."
            )
        real_dir = next(iter(parents))
    else:
        if args.real_dir is None:
            raise ValueError("Either --config or --real_dir must be provided.")
        real_dir = args.real_dir

    # Preflight: show how many targets are covered by fakes vs real_dir.
    try:
        all_real = sorted(_iter_images(real_dir))
        used_targets = _collect_target_stems_from_fakes(Path(args.fakes_dir), tuple(args.presets))
        real_stems = {p.stem for p in all_real}
        covered = real_stems & used_targets
        print(f"[dataset_build] real_dir images: {len(all_real)}")
        print(f"[dataset_build] unique targets found in fakes: {len(used_targets)}")
        print(f"[dataset_build] real images matching fake targets: {len(covered)}")
        if args.max_real is not None and args.restrict_real_to_used_targets and len(covered) < args.max_real:
            need = args.max_real - len(covered)
            print(f"[dataset_build] WARNING: cannot reach max_real={args.max_real} using only matched targets (missing {need}).")
            if not args.fill_real_to_max:
                print("[dataset_build] Hint: re-run with --fill_real_to_max (adds unpaired real) or generate more fakes to cover more targets.")
    except Exception as exc:
        print(f"[dataset_build] preflight skipped: {exc}")

    manifest = build_combined_dataset(
        real_dir=real_dir,
        fakes_dir=args.fakes_dir,
        output_dir=args.out_dir,
        split_ratios=split_ratios,
        seed=args.seed,
        presets=tuple(args.presets),
        copy=not args.move,
        restrict_real_to_used_targets=args.restrict_real_to_used_targets,
        max_real_images=args.max_real,
        fill_real_to_max=args.fill_real_to_max,
    )

    for split in ("train", "val", "test"):
        n_real = sum(1 for x in manifest[split] if x["label"] == "real")
        n_fake = sum(1 for x in manifest[split] if x["label"] == "fake")
        print(f"{split:5s}: {n_real:6d} real + {n_fake:6d} fake = {n_real + n_fake:6d}")
    print(f"Manifest saved: {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

