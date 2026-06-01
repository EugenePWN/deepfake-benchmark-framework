# deepfake_benchmark/benchmark.py
"""
Оркестратор бенчмарка. Три режима:

  mode=generate  → генерация фейков + сборка датасета через dataset_build
  mode=evaluate  → оценка детектора через detect_dataset
  mode=full      → generate + evaluate

Реальный пайплайн:
  DatasetGenerator  → data/fakes/<preset>/<model>/
  dataset_build.py  → data/deepfake_dataset/ (train/val/test + manifest.json)
  train_*.py        → checkpoints/
  detect_dataset.py → data/results/metrics.json
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

from .config import BenchmarkConfig, EvalDatasetConfig, MixedEvalDatasetConfig
from .core.dataset_loader import DatasetLoader
from .core.dataset_generator import DatasetGenerator
from .core.detector_manager import DetectorManager
from .core.metric_evaluator import EvaluationReport, MetricEvaluator
from .core.reporter import Reporter
from .types import SampleItem

logger = logging.getLogger(__name__)


class Benchmark:
    """
    Оркестратор бенчмарка.

    Режимы (config.mode):
      "generate"  — генерирует фейки через FaceFusion, затем вызывает
                    dataset_build для создания train/val/test структуры.
                    Результат:  data/fakes/<preset>/<model>/
                                data/deepfake_dataset/train|val|test + manifest.json

      "evaluate"  — прогоняет детекторы на готовом датасете (через manifest.json
                    или напрямую через real_dir + fake_dir).
                    Результат:  data/results/metrics.json + отчёты

      "full"      — generate + evaluate за один запуск.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

        self._loader:       Optional[DatasetLoader]    = None
        self._generator:    Optional[DatasetGenerator] = None
        self._detector_mgr: Optional[DetectorManager]  = None
        self._evaluator:    Optional[MetricEvaluator]   = None
        self._reporter:     Optional[Reporter]          = None

    # ------------------------------------------------------------------
    # Ленивые свойства
    # ------------------------------------------------------------------

    @property
    def evaluator(self) -> MetricEvaluator:
        if self._evaluator is None:
            self._evaluator = MetricEvaluator(self.config)
        return self._evaluator

    @property
    def reporter(self) -> Reporter:
        if self._reporter is None:
            self._reporter = Reporter(self.config)
        return self._reporter

    @property
    def loader(self) -> DatasetLoader:
        if self._loader is None:
            self._loader = DatasetLoader(self.config)
        return self._loader

    @property
    def generator(self) -> DatasetGenerator:
        if self._generator is None:
            self._generator = DatasetGenerator(self.config)
        return self._generator

    @property
    def detector_mgr(self) -> DetectorManager:
        if self._detector_mgr is None:
            self._detector_mgr = DetectorManager(config=self.config)
            self._detector_mgr.setup()
        return self._detector_mgr

    # ------------------------------------------------------------------
    # Точка входа
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, object]:
        mode = getattr(self.config, "mode", "generate")
        logger.info("[Benchmark] mode=%s", mode)

        if mode == "generate":
            return self._run_generate()
        elif mode == "evaluate":
            return self._run_evaluate()
        elif mode == "full":
            return self._run_full()
        else:
            raise ValueError(
                f"Unknown mode: {mode!r}. Use 'generate', 'evaluate' or 'full'."
            )

    # ------------------------------------------------------------------
    # Pipeline A: generate
    # ------------------------------------------------------------------

    def _run_generate(self) -> Dict[str, object]:
        """
        Шаг 1: Загрузка реальных изображений.
        Шаг 2: Генерация фейков через FaceFusion → data/fakes/
        Шаг 3: Сборка датасета через dataset_build → data/deepfake_dataset/
        """
        # Шаг 1 — загрузка
        logger.info("[Benchmark] Stage 1: Loading real data...")
        real_items = self.loader.load_all()
        if not real_items:
            return {"status": "error", "message": "no_real_data"}

        sources = [i for i in real_items if i.role in ("source", None)]
        targets = [i for i in real_items if i.role in ("target", None)]
        if not sources or not targets:
            return {
                "status": "error", "message": "insufficient_data",
                "sources": len(sources), "targets": len(targets),
            }

        # Шаг 2 — генерация
        logger.info("[Benchmark] Stage 2: Generating fakes (preset=%s)...",
                    self.config.generator.preset)
        fake_root = Path(self.config.fake_data_root).resolve()
        fake_items = self.generator.generate(sources, targets, fake_root)
        logger.info("[Benchmark] Generated %d fake items.", len(fake_items))

        if not fake_items:
            logger.warning("[Benchmark] No fakes generated.")

        # Шаг 3 — сборка датасета (dataset_build)
        # Пропускается если output.skip_dataset_build=True.
        output_cfg = self.config.output
        if output_cfg.skip_dataset_build:
            logger.info(
                "[Benchmark] skip_dataset_build=True — fakes in %s. "
                "Run dataset_build.py manually.", fake_root,
            )
            build_result = {"status": "skipped", "reason": "skip_dataset_build=True"}
        else:
            build_result = self._build_dataset(
                real_items=real_items,
                fake_root=fake_root,
                output_structure=output_cfg.structure,
            )

        return {
            "status":       "success",
            "mode":         "generate",
            "counts":       {"real": len(real_items), "fake": len(fake_items)},
            "preset":       self.config.generator.preset,
            "fakes_dir":    str(fake_root),
            "dataset_build": build_result,
        }

    def _build_dataset(
        self,
        real_items: List[SampleItem],
        fake_root:  Path,
        output_structure: str = "train_val_test",
    ) -> Dict[str, object]:
        """
        Вызывает dataset_build.build_combined_dataset() для формирования
        итоговой структуры train/val/test.

        output_structure:
          "train_val_test" → обычная структура для обучения детектора
          "flat_eval"      → flat real/ + fake/ для валидации готовой модели
        """
        try:
            from .utils.dataset_build import build_combined_dataset
        except ImportError:
            logger.warning(
                "[Benchmark] dataset_build not available — skipping dataset assembly. "
                "Run: python -m deepfake_benchmark.utils.dataset_build manually."
            )
            return {"status": "skipped", "reason": "dataset_build not importable"}

        # Определяем папку с реальными изображениями
        # real_items могут быть из разных папок — берём родителей
        real_paths = sorted({i.media_path.resolve() for i in real_items
                             if i.role in (None, "target")})
        if not real_paths:
            return {"status": "skipped", "reason": "no target real images"}

        # Если все реальные в одной папке — используем её напрямую
        parents = {p.parent for p in real_paths}
        if len(parents) == 1:
            real_dir = next(iter(parents))
        else:
            # Несколько папок — создаём временную или используем первую
            logger.warning("[Benchmark] Real images from multiple dirs — using first")
            real_dir = next(iter(parents))

        out_cfg   = getattr(self.config, "output", None)
        splits    = getattr(self.config, "split_ratios",
                            {"train": 0.70, "val": 0.15, "test": 0.15})
        seed      = getattr(self.config, "random_seed", 42)
        presets   = self._get_active_presets()

        if output_structure == "flat_eval":
            # Плоская структура для валидации
            output_dir = Path(
                getattr(out_cfg, "output_dir", "data/eval_dataset")
                if out_cfg else "data/eval_dataset"
            )
            n_per_class = getattr(out_cfg, "n_per_class", None) if out_cfg else None
            # flat_eval: берём только test (100%)
            flat_splits = {"train": 0.0, "val": 0.0, "test": 1.0}
            logger.info("[Benchmark] Building flat eval dataset → %s", output_dir)
        else:
            # Датасет кладём рядом с results_root: data/smoke_test/deepfake_dataset/
            # Явный путь берём из fake_data_root — он всегда задан пользователем.
            output_dir  = Path(self.config.output.fake_data_root).parent / "deepfake_dataset"
            flat_splits = splits
            n_per_class = None
            logger.info("[Benchmark] Building train/val/test dataset → %s", output_dir)

        restrict_real = getattr(out_cfg, "real_source", "targets") == "targets" \
            if out_cfg else True

        try:
            manifest = build_combined_dataset(
                real_dir=real_dir,
                fakes_dir=fake_root,
                output_dir=output_dir,
                split_ratios=flat_splits,
                seed=seed,
                presets=presets,
                copy=True,
                restrict_real_to_used_targets=restrict_real,
                max_real_images=n_per_class,
            )
            stats = {
                split: {
                    "real": sum(1 for x in rows if x["label"] == "real"),
                    "fake": sum(1 for x in rows if x["label"] == "fake"),
                }
                for split, rows in manifest.items()
            }
            logger.info("[Benchmark] Dataset built: %s", stats)
            return {"status": "success", "output_dir": str(output_dir), "stats": stats}
        except Exception as e:
            logger.error("[Benchmark] dataset_build failed: %s", e, exc_info=True)
            return {"status": "error", "reason": str(e)}

    def _get_active_presets(self) -> tuple:
        """
        Возвращает кортеж активных пресетов для dataset_build.
        Всегда возвращает только тот пресет который реально использовался
        при генерации — dataset_build будет искать фейки именно в этой подпапке.
        """
        return (self.config.generator.preset,)

    # ------------------------------------------------------------------
    # Pipeline B: evaluate
    # ------------------------------------------------------------------

    def _run_evaluate(self) -> Dict[str, object]:
        """
        Прогоняет детекторы на готовых датасетах.
        Поддерживает два источника:
          1. EvalDatasetConfig (real_dir + fake_dir) — через DetectorManager.run_all()
          2. manifest.json — через detect_dataset.run_detection_on_dataset()
        """
        eval_cfg = getattr(self.config, "eval_config", None)
        if not eval_cfg:
            return {"status": "error", "message": "eval_config required for mode=evaluate"}

        all_reports: List[EvaluationReport] = []

        # Поддержка mix_with: объединяем датасеты если указано
        datasets_to_eval = eval_cfg.mixed_datasets()

        for ds_cfg in datasets_to_eval:
            logger.info("[Benchmark] Evaluating on: %s", ds_cfg.name)

            # MixedEvalDatasetConfig — несколько источников объединяются
            if isinstance(ds_cfg, MixedEvalDatasetConfig):
                items = []
                for src in ds_cfg.sources:
                    src_items = self._load_eval_items(src)
                    items.extend(src_items)
                logger.info("[Benchmark] Mixed dataset %s: %d items total",
                            ds_cfg.name, len(items))
            else:
                items = self._load_eval_items(ds_cfg)

            if not items:
                logger.warning("[Benchmark] No items in %s — skipping", ds_cfg.name)
                continue

            for item in items:
                item.meta.setdefault("preset", "overall")

            detections = self.detector_mgr.run_all(items)
            if not detections:
                continue

            report = self.evaluator.evaluate(detections, dataset_name=ds_cfg.name)
            all_reports.append(report)

        merged  = self._merge_reports(all_reports)
        out_dir = self.reporter.generate(merged, run_name="eval")

        return {
            "status":     "success",
            "mode":       "evaluate",
            "n_datasets": len(all_reports),
            "output_dir": str(out_dir),
            "metrics":    merged.to_dict(),
        }

    def _load_eval_items(self, ds_cfg: EvalDatasetConfig) -> List[SampleItem]:
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        seed = getattr(self.config, "random_seed", 42)

        def _collect(folder: Path, label: str) -> List[SampleItem]:
            if not folder.exists():
                logger.warning("[Benchmark] Not found: %s", folder)
                return []
            paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
            if ds_cfg.n_per_class:
                rng = random.Random(seed)
                paths = rng.sample(paths, min(ds_cfg.n_per_class, len(paths)))
            return [
                SampleItem(
                    sample_id=f"{ds_cfg.name}_{p.stem}",
                    media_path=p, label=label,
                    generator="none" if label == "real" else "external",
                    dataset=ds_cfg.name, split="test", meta={},
                )
                for p in paths
            ]

        real_items = _collect(ds_cfg.real_dir, "real")
        fake_items = _collect(ds_cfg.fake_dir, "fake")
        logger.info("[Benchmark] %s: %d real + %d fake",
                    ds_cfg.name, len(real_items), len(fake_items))
        return real_items + fake_items

    # ------------------------------------------------------------------
    # Pipeline C: full
    # ------------------------------------------------------------------

    def _run_full(self) -> Dict[str, object]:
        gen = self._run_generate()
        if gen.get("status") != "success":
            return gen
        ev  = self._run_evaluate()
        return {"status": "success", "mode": "full", "generate": gen, "evaluate": ev}

    # ------------------------------------------------------------------
    # Вспомогательные
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_reports(reports: List[EvaluationReport]) -> EvaluationReport:
        merged = EvaluationReport()
        for r in reports:
            for m in r.all_metrics():
                merged.add(m)
        return merged

    def generate_only(self) -> List[SampleItem]:
        """Только генерация фейков без сборки датасета."""
        real_items = self.loader.load_all()
        if not real_items:
            return []
        sources = [i for i in real_items if i.role in ("source", None)]
        targets = [i for i in real_items if i.role in ("target", None)]
        if not sources or not targets:
            return []
        return self.generator.generate(
            sources, targets, Path(self.config.fake_data_root)
        )

    def evaluate_on_dirs(
        self,
        real_dir:     Path,
        fake_dir:     Path,
        dataset_name: str  = "custom",
        n_per_class:  Optional[int] = None,
        run_name:     str  = "eval",
    ) -> EvaluationReport:
        """
        Shortcut: оценить детекторы на готовых папках без полного конфига.

        Пример:
            report = bench.evaluate_on_dirs(
                real_dir=Path("data/deepfake_dataset/test/real"),
                fake_dir=Path("data/deepfake_dataset/test/fake"),
            )
        """
        ds_cfg  = EvalDatasetConfig(
            name=dataset_name, real_dir=real_dir,
            fake_dir=fake_dir, n_per_class=n_per_class,
        )
        items   = self._load_eval_items(ds_cfg)
        for item in items:
            item.meta.setdefault("preset", "overall")
        detections = self.detector_mgr.run_all(items)
        report     = self.evaluator.evaluate(detections, dataset_name=dataset_name)
        self.reporter.generate(report, run_name=run_name)
        return report