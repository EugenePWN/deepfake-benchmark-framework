"""
deepfake_benchmark/run.py
=========================
CLI точка входа. Парсит YAML -> BenchmarkConfig -> Benchmark.run().

Запуск:
    poetry run python -m deepfake_benchmark.run --config smoke_test.yaml
    poetry run python -m deepfake_benchmark.run --config smoke_test.yaml --dry_run
    poetry run python -m deepfake_benchmark.run --init smoke_test
    poetry run python -m deepfake_benchmark.run --init generate

Наследование конфигов через extends:
    # base.yaml
    mode: evaluate
    device: cuda

    # experiment.yaml
    extends: base.yaml
    generation:
      preset: hard
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# YAML утилиты
# ──────────────────────────────────────────────────────────────────────────────

def _merge_dicts(base: dict, override: dict) -> dict:
    """Рекурсивное слияние: override побеждает base, вложенные dict мержатся."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _find_project_root(config_path: Path) -> Path:
    """
    Ищет корень проекта поднимаясь вверх от папки конфига.
    Корень определяется по наличию pyproject.toml или .git.
    Если не найдено — возвращает CWD.
    """
    current = config_path.parent.resolve()
    for _ in range(10):  # не более 10 уровней вверх
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return Path.cwd()


def _resolve_path(raw_path: str, config_path: Path) -> Path:
    """
    Разрешает путь в следующем порядке приоритета:
    1. Абсолютный путь → возвращаем как есть.
    2. ${PROJECT_ROOT}/... → от корня проекта (pyproject.toml / .git).
    3. Существует относительно CWD → используем CWD.
    4. Иначе → относительно папки конфига.

    Используй ${PROJECT_ROOT} в YAML если конфиг лежит в подпапке (configs/),
    а пути данных задаются от корня проекта.
    """
    project_root = _find_project_root(config_path)

    # Подставляем ${PROJECT_ROOT}
    raw_path = raw_path.replace("${PROJECT_ROOT}", str(project_root))

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    # Относительно CWD — если файл/папка реально существует
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    # Относительно корня проекта
    root_candidate = (project_root / path).resolve()
    if root_candidate.exists():
        return root_candidate

    # Fallback — относительно папки конфига
    return (config_path.parent / path).resolve()


def _load_yaml_with_extends(path: Path, visited: Optional[Set[Path]] = None) -> dict:
    """
    Загружает YAML с поддержкой наследования через ключ 'extends'.
    Дочерний конфиг переопределяет поля базового (deep merge).
    """
    if visited is None:
        visited = set()
    resolved = path.resolve()
    if resolved in visited:
        raise ValueError(f"Cyclic extends detected: {resolved}")
    visited.add(resolved)

    with resolved.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    extends = raw.pop("extends", None)
    if extends is None:
        return raw

    base_path = _resolve_path(str(extends), resolved)
    base_raw = _load_yaml_with_extends(base_path, visited=visited)
    return _merge_dicts(base_raw, raw)


def _get_nested(d: dict, *keys: str) -> bool:
    """Проверяет наличие ключа в словаре по цепочке ключей."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return False
        cur = cur[k]
    return True


# ──────────────────────────────────────────────────────────────────────────────
# YAML -> BenchmarkConfig
# ──────────────────────────────────────────────────────────────────────────────

def load_config(yaml_path: Path, device_override: Optional[str] = None):
    """
    Загружает YAML (с поддержкой extends) и строит BenchmarkConfig.

    Маппинг секций:
      data:       -> LoaderConfig
      generation: -> GeneratorConfig
      output:     -> OutputConfig
      evaluation: -> EvalConfig
      splits:     -> split_ratios
      device:     -> BenchmarkConfig.device
      seed:       -> BenchmarkConfig.random_seed
    """
    from deepfake_benchmark.config import (
        BenchmarkConfig,
        DetectorEntryConfig,
        EvalConfig,
        EvalDatasetConfig,
        GeneratorConfig,
        LoaderConfig,
        OutputConfig,
    )

    raw = _load_yaml_with_extends(yaml_path)

    if device_override:
        raw["device"] = device_override

    mode = raw.get("mode", "generate")

    # ── data -> LoaderConfig ──────────────────────────────────────────────────
    d = raw.get("data", {})
    loader = LoaderConfig(
        source_datasets=d.get("source_datasets", []),
        real_data_root=(
            _resolve_path(d["real_data_root"], yaml_path)
            if "real_data_root" in d
            else "data/real"
        ),
        max_items_per_dataset=d.get("max_items_per_dataset"),
        external_source_dir=(
            _resolve_path(d["external_source_dir"], yaml_path)
            if "external_source_dir" in d
            else None
        ),
        identity_split=d.get("identity_split", True),
        auto_download=d.get("auto_download", False),
    )

    # ── generation -> GeneratorConfig ─────────────────────────────────────────
    g = raw.get("generation", {})
    generator = GeneratorConfig(
        preset=g.get("preset", "default"),
        native_args=g.get("native_args", {"execution_provider": "cuda"}),
        pairing_mode=g.get("pairing_mode", "one_for_all"),
        max_pairs=g.get("max_pairs"),
        output_structure=g.get("output_structure", "flat"),
        facefusion_dir=(
            _resolve_path(g["facefusion_dir"], yaml_path)
            if "facefusion_dir" in g
            else None
        ),
        facefusion_python=g.get("facefusion_python"),
        parallel=g.get("parallel", False),
        parallel_workers=g.get("parallel_workers", 2),
        skip_existing=g.get("skip_existing", True),
        subprocess_timeout=g.get("subprocess_timeout", 300),
        identity_aware_pairing=g.get("identity_aware_pairing", True),
        pairing_seed=g.get("pairing_seed", 42),
    )

    # ── output -> OutputConfig ────────────────────────────────────────────────
    o = raw.get("output", {})
    output = OutputConfig(
        structure=o.get("structure", "train_val_test"),
        fake_data_root=(
            _resolve_path(o["fake_data_root"], yaml_path)
            if "fake_data_root" in o
            else "data/fakes"
        ),
        results_root=(
            _resolve_path(o["results_root"], yaml_path)
            if "results_root" in o
            else "data/results"
        ),
        skip_dataset_build=o.get("skip_dataset_build", False),
        output_dir=(
            _resolve_path(o["output_dir"], yaml_path)
            if "output_dir" in o
            else None
        ),
        n_per_class=o.get("n_per_class"),
        copy_real=o.get("copy_real", True),
        real_source=o.get("real_source", "targets"),
    )

    # ── evaluation -> EvalConfig ──────────────────────────────────────────────
    ev = raw.get("evaluation", {})
    eval_config = None
    if ev:
        detectors: List[DetectorEntryConfig] = [
            DetectorEntryConfig(
                name=det["name"],
                architecture=det["architecture"],
                weights_path=_resolve_path(det["weights_path"], yaml_path),
                threshold=det.get("threshold", 0.5),
                img_size=det.get("img_size"),
                batch_size=det.get("batch_size", 16),
            )
            for det in ev.get("detectors", [])
        ]
        eval_datasets_list: List[EvalDatasetConfig] = [
            EvalDatasetConfig(
                name=ds["name"],
                real_dir=_resolve_path(ds["real_dir"], yaml_path),
                fake_dir=_resolve_path(ds["fake_dir"], yaml_path),
                n_per_class=ds.get("n_per_class"),
                mix_with=ds.get("mix_with", []),
            )
            for ds in ev.get("eval_datasets", [])
        ]
        results_root_for_eval = (
            _resolve_path(o["results_root"], yaml_path)
            if "results_root" in o
            else "data/results"
        )
        eval_config = EvalConfig(
            detectors=detectors,
            eval_datasets=eval_datasets_list,
            output_format=ev.get("output_format", "json"),
            save_plots=ev.get("save_plots", True),
            threshold_metric=ev.get("threshold_metric", "f1"),
            random_seed=raw.get("seed", 42),
            results_root=results_root_for_eval,
        )

    splits: Dict[str, float] = raw.get(
        "splits", {"train": 0.70, "val": 0.15, "test": 0.15}
    )

    return BenchmarkConfig(
        mode=mode,
        loader=loader,
        generator=generator,
        output=output,
        eval_config=eval_config,
        device=raw.get("device", "cuda"),
        random_seed=raw.get("seed", 42),
        split_ratios=splits,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dry run
# ──────────────────────────────────────────────────────────────────────────────

def dry_run(config, yaml_path: Optional[Path] = None) -> None:
    """Проверяет конфиг и выводит сводку без запуска пайплайна."""

    def _was_explicit(section: str, field: str) -> bool:
        """True если поле задано явно в YAML (не авто-определено)."""
        if yaml_path is None:
            return False
        try:
            raw = _load_yaml_with_extends(yaml_path)
            return field in raw.get(section, {})
        except Exception:
            return False

    sep = "=" * 62
    print(f"\n{sep}")
    print("  DRY RUN -- конфиг валиден")
    print(sep)
    print(f"  mode   : {config.mode}")
    print(f"  device : {config.device}")
    print(f"  seed   : {config.random_seed}")

    # ── Generation ────────────────────────────────────────────────────────────
    if config.mode in ("generate", "full"):
        g = config.generator
        ff_ready = g.facefusion_ready
        ff_src = (
            "from config"
            if _was_explicit("generation", "facefusion_dir")
            else "auto-detected"
        )
        py_explicit = _was_explicit("generation", "facefusion_python")
        if py_explicit:
            py_src = "from config"
        elif g.facefusion_python:
            py_src = "auto-detected"
        else:
            py_src = "sys.executable (fallback)"

        print("\n  GENERATION:")
        print(f"    preset        : {g.preset}")
        print(f"    max_pairs     : {g.max_pairs or 'unlimited'}")
        print(
            f"    facefusion_dir: {g.facefusion_dir}  "
            f"{'OK' if ff_ready else 'NOT FOUND'}  [{ff_src}]"
        )
        print(f"    python        : {g.resolved_python}  [{py_src}]")
        print(f"    skip_build    : {config.output.skip_dataset_build}")
        print(f"    structure     : {config.output.structure}")

        if not ff_ready:
            print(f"\n  [!] FaceFusion not found at: {g.facefusion_dir}")
            print("      Fix: poetry run python scripts/setup_facefusion.py")

    # ── Evaluation ────────────────────────────────────────────────────────────
    if config.mode in ("evaluate", "full") and config.eval_config:
        ev = config.eval_config
        is_full = config.mode == "full"

        print(f"\n  EVALUATION:")
        print(f"    detectors ({len(ev.detectors)}):")
        for det in ev.detectors:
            ok = det.weights_path.exists()
            status = "OK" if ok else "MISSING"
            print(
                f"      [{status}] {det.name} ({det.architecture}) "
                f"thr={det.threshold} -> {det.weights_path}"
            )

        print(f"    eval_datasets ({len(ev.eval_datasets)}):")
        any_missing = False
        for ds in ev.eval_datasets:
            r_ok = ds.real_dir.exists()
            f_ok = ds.fake_dir.exists()
            both_ok = r_ok and f_ok

            if both_ok:
                marker = "OK"
            elif is_full:
                marker = "pending"
            else:
                marker = "MISSING"

            print(f"      [{marker}] {ds.name}")

            if not r_ok:
                if is_full:
                    print(f"        real_dir: {ds.real_dir}")
                    print("        (will be created by generate step)")
                else:
                    print(f"        [!] real_dir NOT FOUND: {ds.real_dir}")
                any_missing = True

            if not f_ok:
                if is_full:
                    print(f"        fake_dir: {ds.fake_dir}")
                    print("        (will be created by generate step)")
                else:
                    print(f"        [!] fake_dir NOT FOUND: {ds.fake_dir}")
                any_missing = True

        if is_full and any_missing:
            print("\n  [i] eval_dataset paths marked [pending] will be created")
            print("      automatically during the generate step. This is expected.")

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Шаблоны конфигов
# ──────────────────────────────────────────────────────────────────────────────

TEMPLATES: Dict[str, str] = {

"smoke_test": """\
# smoke_test.yaml -- минимальный тест всего пайплайна
# Запуск: poetry run python -m deepfake_benchmark.run --config smoke_test.yaml

mode: full

data:
  real_data_root: "D:/datasets"
  source_datasets:
    - celeba_hq
  external_source_dir: "D:/datasets/vggface2/data/train"
  max_items_per_dataset: 30

generation:
  preset: easy
  max_pairs: 5
  skip_existing: true
  subprocess_timeout: 120
  native_args:
    execution_provider: cuda
  # facefusion_dir и facefusion_python не нужны после setup_facefusion.py

splits:
  train: 0.60
  val:   0.20
  test:  0.20

output:
  structure: train_val_test
  fake_data_root: "data/smoke_test/fakes"
  results_root:   "data/smoke_test/results"
  skip_dataset_build: false

evaluation:
  detectors:
    - name: xception
      architecture: xception
      weights_path: "checkpoints_xception/xception_best.pth"
      threshold: 0.700
      batch_size: 4
  eval_datasets:
    - name: smoke_test
      real_dir: "data/smoke_test/deepfake_dataset/test/real"
      fake_dir: "data/smoke_test/deepfake_dataset/test/fake"
  output_format: all
  save_plots: true

device: cuda
seed: 42
""",

"generate": """\
# generate.yaml -- генерация датасета для обучения детектора
# Запуск: poetry run python -m deepfake_benchmark.run --config generate.yaml

mode: generate

data:
  real_data_root: "D:/datasets"
  source_datasets:
    - celeba_hq
  external_source_dir: "D:/datasets/vggface2/data/train"
  max_items_per_dataset: 2000

generation:
  preset: medium
  max_pairs: 1500
  output_structure: preset_model
  skip_existing: true
  native_args:
    execution_provider: cuda

splits:
  train: 0.70
  val:   0.15
  test:  0.15

output:
  structure: train_val_test
  fake_data_root: "data/fakes"
  results_root:   "data/results"

device: cuda
seed: 42
""",

"evaluate": """\
# evaluate.yaml -- оценка готового детектора
# Запуск: poetry run python -m deepfake_benchmark.run --config evaluate.yaml

mode: evaluate

evaluation:
  detectors:
    - name: xception
      architecture: xception
      weights_path: "checkpoints_xception/xception_best.pth"
      threshold: 0.700
      batch_size: 16
    - name: efficientnet
      architecture: efficientnet
      weights_path: "checkpoints_effnet/effnet_best.pth"
      threshold: 0.588
      batch_size: 16
    - name: f3net
      architecture: f3net
      weights_path: "checkpoints_f3net/f3net_best.pth"
      threshold: 0.516
      batch_size: 12
  eval_datasets:
    - name: test_facefusion
      real_dir: "data/deepfake_dataset/test/real"
      fake_dir: "data/deepfake_dataset/test/fake"
  output_format: all
  save_plots: true
  threshold_metric: f1

output:
  results_root: "data/results"

device: cuda
seed: 42
""",

"full": """\
# full.yaml -- генерация + оценка
# Запуск: poetry run python -m deepfake_benchmark.run --config full.yaml

mode: full

data:
  real_data_root: "D:/datasets"
  source_datasets:
    - celeba_hq
  external_source_dir: "D:/datasets/vggface2/data/train"
  max_items_per_dataset: 1500

generation:
  preset: medium
  max_pairs: 1500
  output_structure: preset_model
  skip_existing: true
  native_args:
    execution_provider: cuda

splits:
  train: 0.70
  val:   0.15
  test:  0.15

output:
  structure: train_val_test
  fake_data_root: "data/fakes"
  results_root:   "data/results"

evaluation:
  detectors:
    - name: xception
      architecture: xception
      weights_path: "checkpoints_xception/xception_best.pth"
      threshold: 0.700
    - name: efficientnet
      architecture: efficientnet
      weights_path: "checkpoints_effnet/effnet_best.pth"
      threshold: 0.588
    - name: f3net
      architecture: f3net
      weights_path: "checkpoints_f3net/f3net_best.pth"
      threshold: 0.516
  eval_datasets:
    - name: generated_test
      real_dir: "data/deepfake_dataset/test/real"
      fake_dir: "data/deepfake_dataset/test/fake"
  output_format: all
  save_plots: true

device: cuda
seed: 42
""",
}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deepfake Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  poetry run python -m deepfake_benchmark.run --init smoke_test\n"
            "  poetry run python -m deepfake_benchmark.run --config smoke_test.yaml --dry_run\n"
            "  poetry run python -m deepfake_benchmark.run --config smoke_test.yaml\n"
            "  poetry run python -m deepfake_benchmark.run --config smoke_test.yaml --device cpu\n"
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to YAML config"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device: cuda | cpu"
    )
    parser.add_argument(
        "--init",
        type=str,
        default=None,
        choices=list(TEMPLATES),
        help="Create a config template",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for --init (default: <mode>.yaml)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Validate config without running",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(args.loglevel)

    # Создать шаблон
    if args.init:
        out = Path(args.output or f"{args.init}.yaml")
        out.write_text(TEMPLATES[args.init], encoding="utf-8")
        print(f"Created: {out}")
        print(f"Edit paths, then run:")
        print(f"  poetry run python -m deepfake_benchmark.run --config {out} --dry_run")
        return

    if not args.config:
        parser.print_help()
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    try:
        config = load_config(config_path, device_override=args.device)
    except Exception as exc:
        print(f"Config error: {exc}")
        sys.exit(1)

    if args.dry_run:
        dry_run(config, yaml_path=config_path)
        return

    from deepfake_benchmark.benchmark import Benchmark

    result = Benchmark(config).run()

    if result.get("status") == "success":
        print(f"\nDone. mode={result.get('mode')}")
        if "output_dir" in result:
            print(f"  Output: {result['output_dir']}")
        if result.get("mode") in ("evaluate", "full"):
            print(f"  Results: {config.output.results_root}")
    else:
        print(f"\nFailed: {result.get('message', 'unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()