# deepfake_benchmark/core/detector_manager.py
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from ..config import BenchmarkConfig, DetectorEntryConfig, EvalConfig, MixedEvalDatasetConfig
from ..types import SampleItem
from .detectors.base_detector import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

# Дефолты по архитектуре
_ARCH_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "xception":    {"img_size": 299, "mean": [0.5]*3,               "std": [0.5]*3},
    "efficientnet": {"img_size": 380, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "f3net":       {"img_size": 299, "mean": [0.5]*3,               "std": [0.5]*3},
}


# ──────────────────────────────────────────────────────────────────────────────
# Реестр кастомных детекторов
# ──────────────────────────────────────────────────────────────────────────────

class DetectorRegistry:
    """
    Реестр для кастомных детекторов (BaseDetector-подклассы).
    Регистрируй через декоратор: @DetectorRegistry.register
    """
    _registry: Dict[str, Type[BaseDetector]] = {}

    @classmethod
    def register(cls, detector_cls: Type[BaseDetector]) -> Type[BaseDetector]:
        if not detector_cls.name:
            raise ValueError(f"Cannot register {detector_cls.__name__}: empty `name`.")
        cls._registry[detector_cls.name] = detector_cls
        return detector_cls

    @classmethod
    def build(cls, name: str, threshold: float = 0.5, **kwargs: Any) -> BaseDetector:
        if name not in cls._registry:
            raise KeyError(f"Detector {name!r} not registered. Available: {cls.available()}")
        return cls._registry[name](threshold=threshold, **kwargs)

    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls._registry.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка весов по архитектуре
# Импорт через прямое имя модуля — модели лежат рядом с train-скриптами,
# не внутри пакета. При переносе в models/ заменить на относительный импорт.
# ──────────────────────────────────────────────────────────────────────────────

def _load_weights(
    architecture: str, weights_path: Path, device: torch.device
) -> Tuple[nn.Module, Dict[str, int], int]:
    """
    Загружает модель из checkpoint.
    Возвращает (model, class_to_idx, fake_class_idx).
    """
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # Убираем _orig_mod. от torch.compile
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    class_to_idx: Dict[str, int] = ckpt.get("class_to_idx", {"fake": 0, "real": 1})
    fake_class_idx = class_to_idx.get("fake", 0)

    # Стратегия импорта:
    # 1. Пробуем из пакета (если модели перенесены в deepfake_benchmark/models/)
    # 2. Fallback на прямой import из корня проекта (текущее расположение)
    if architecture == "xception":
        model = _import_model("xception_model", "build_xception",
                              num_classes=2, dropout_rate=0.5)
    elif architecture == "efficientnet":
        model = _import_model("efficientnet_model", "build_efficientnet_b4",
                              num_classes=2, dropout_rate=0.4, pretrained=False)
    elif architecture == "f3net":
        model = _import_model("f3net_model", "build_f3net",
                              num_classes=2, dropout_rate=0.5)
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}. "
                         f"Supported: xception, efficientnet, f3net")

    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, class_to_idx, fake_class_idx


def _find_project_root() -> Path:
    """Ищет корень проекта по pyproject.toml или .git."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[3]  # fallback: 3 уровня вверх от core/


def _import_model(module_name: str, builder_name: str, **kwargs) -> nn.Module:
    """
    Импортирует build-функцию модели. Порядок поиска:
    1. deepfake_benchmark.models.<module_name>  (если модели в пакете)
    2. <module_name> с project_root в sys.path  (если модели в корне проекта)
    """
    import importlib

    # Попытка 1: из пакета deepfake_benchmark.models
    try:
        mod = importlib.import_module(f"deepfake_benchmark.models.{module_name}")
        builder = getattr(mod, builder_name)
        return builder(**kwargs)
    except (ModuleNotFoundError, AttributeError):
        pass

    # Попытка 2: корень проекта добавляем в sys.path и импортируем напрямую
    project_root = str(_find_project_root())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        mod = importlib.import_module(module_name)
        builder = getattr(mod, builder_name)
        return builder(**kwargs)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(
            f"Cannot import {builder_name} from '{module_name}'.\n"
            f"Tried:\n"
            f"  1. deepfake_benchmark.models.{module_name}\n"
            f"  2. {module_name} (with project root {project_root} in sys.path)\n"
            f"Error: {e}\n"
            f"Fix: place {module_name}.py in deepfake_benchmark/models/ "
            f"or in the project root ({project_root})."
        ) from e


# ──────────────────────────────────────────────────────────────────────────────
# Dataset для инференса
# ──────────────────────────────────────────────────────────────────────────────

class _InferenceDataset(Dataset):
    def __init__(self, items: List[SampleItem], transform: transforms.Compose):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img  = Image.open(item.media_path).convert("RGB")
        return self.transform(img), idx


# ──────────────────────────────────────────────────────────────────────────────
# DetectorManager
# ──────────────────────────────────────────────────────────────────────────────

class DetectorManager:
    """
    Оркестратор инференса детекторов.

    Два API:
      Новый (рекомендуется): DetectorEntryConfig в EvalConfig.
        Загружает .pth файлы напрямую, без Registry.
      Старый (совместимость): DetectorRegistry.
        Кастомные BaseDetector-подклассы.
    """

    def __init__(
        self,
        config:      Optional[BenchmarkConfig] = None,
        eval_config: Optional[EvalConfig]      = None,
        device:      Optional[torch.device]    = None,
    ) -> None:
        self._bench_config = config
        self._eval_config  = eval_config or (config.eval_config if config else None)

        # Устройство из конфига или автоматически
        device_str = getattr(config, "device", "cuda") if config else "cuda"
        self.device = device or torch.device(
            device_str if torch.cuda.is_available() else "cpu"
        )
        self.use_amp = self.device.type == "cuda"

        # Новый API: name → (model, DetectorEntryConfig, fake_class_idx)
        self._models: Dict[str, tuple] = {}
        # Старый API: name → BaseDetector
        self._registry_detectors: Dict[str, BaseDetector] = {}

    def setup(self) -> None:
        """Загружает все детекторы."""
        if self._eval_config:
            for entry in self._eval_config.detectors:
                self._load_entry(entry)
        if self._bench_config and self._bench_config.detectors:
            for name in self._bench_config.detectors:
                self._load_from_registry(name)

        if not self._models and not self._registry_detectors:
            logger.warning("[DetectorManager] No detectors loaded")
        else:
            logger.info("[DetectorManager] Ready: %s", self.loaded_detectors)

    def _load_entry(self, entry: DetectorEntryConfig) -> None:
        if not entry.weights_path.exists():
            logger.error("[DetectorManager] Weights not found: %s — skipping %s",
                         entry.weights_path, entry.name)
            return
        try:
            model, class_to_idx, fake_class_idx = _load_weights(
                entry.architecture, entry.weights_path, self.device
            )
            self._models[entry.name] = (model, entry, fake_class_idx)
            logger.info("[DetectorManager] Loaded '%s' (%s) thr=%.3f",
                        entry.name, entry.architecture, entry.threshold)
        except Exception as e:
            logger.error("[DetectorManager] Failed '%s': %s", entry.name, e, exc_info=True)

    def _load_from_registry(self, name: str) -> None:
        cfg          = (self._bench_config.detector_configs or {}).get(name, {}).copy()
        threshold    = cfg.pop("threshold", 0.5)
        weights_path = Path(cfg.pop("weights_path")) if "weights_path" in cfg else None
        try:
            det = DetectorRegistry.build(name, threshold=threshold)
            det.load(weights_path=weights_path, **cfg)
            self._registry_detectors[name] = det
            logger.info("[DetectorManager] Registry loaded '%s'", name)
        except Exception as e:
            logger.error("[DetectorManager] Registry failed '%s': %s", name, e, exc_info=True)

    # ------------------------------------------------------------------
    # Новый API: run_on_items / run_all
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_on_items(
        self, items: List[SampleItem], detector_name: str
    ) -> List[DetectionResult]:
        if detector_name not in self._models:
            raise KeyError(f"'{detector_name}' not loaded. Call setup() first.")

        model, entry, fake_class_idx = self._models[detector_name]
        d = _ARCH_DEFAULTS.get(entry.architecture, _ARCH_DEFAULTS["xception"])
        img_size = entry.img_size or d["img_size"]

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(d["mean"], d["std"]),
        ])

        dataset = _InferenceDataset(items, transform)
        loader  = DataLoader(dataset, batch_size=entry.batch_size,
                             shuffle=False, num_workers=4, pin_memory=self.use_amp)

        results: List[DetectionResult] = []
        for batch_imgs, batch_indices in loader:
            batch_imgs = batch_imgs.to(self.device, non_blocking=True)
            with autocast("cuda", enabled=self.use_amp):
                logits = model(batch_imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for prob_row, item_idx in zip(probs, batch_indices.numpy()):
                item      = items[item_idx]
                prob_fake = float(prob_row[fake_class_idx])
                label     = "fake" if prob_fake >= entry.threshold else "real"
                results.append(DetectionResult(
                    sample_id=item.sample_id,
                    detector=detector_name,
                    score=prob_fake,
                    label=label,
                    gt_label=item.label,
                    meta={"architecture": entry.architecture,
                          **item.meta},
                ))
        return results

    def run_all(
        self, items: List[SampleItem]
    ) -> Dict[str, List[DetectionResult]]:
        """Прогоняет все загруженные детекторы (новый API)."""
        output: Dict[str, List[DetectionResult]] = {}
        for name in self._models:
            t0 = time.perf_counter()
            results = self.run_on_items(items, name)
            elapsed = time.perf_counter() - t0
            logger.info("[DetectorManager] '%s': %d items in %.1fs (%.1f img/s)",
                        name, len(results), elapsed,
                        len(results) / elapsed if elapsed > 0 else 0)
            output[name] = results
        return output

    # ------------------------------------------------------------------
    # Старый API: detect (Registry-детекторы)
    # ------------------------------------------------------------------

    def detect(self, items: Sequence[SampleItem]) -> Dict[str, List[DetectionResult]]:
        """Старый API для совместимости с detect_dataset.py."""
        if not self._registry_detectors:
            logger.warning("[DetectorManager] No registry detectors. Use run_all().")
            return {}
        items = list(items)
        batch_size = getattr(self._bench_config, "detector_batch_size", 32)
        results: Dict[str, List[DetectionResult]] = {}
        for name, det in self._registry_detectors.items():
            t0 = time.perf_counter()
            all_res: List[DetectionResult] = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                try:
                    all_res.extend(det.predict_batch(batch))
                except Exception as e:
                    logger.error("[DetectorManager] '%s' error: %s", name, e)
                    for item in batch:
                        all_res.append(DetectionResult(
                            sample_id=item.sample_id, detector=name,
                            score=float("nan"), label="real",
                            gt_label=item.label, meta={"error": str(e)},
                        ))
            results[name] = all_res
            logger.info("[DetectorManager] '%s': %d items in %.1fs",
                        name, len(all_res), time.perf_counter() - t0)
        return results

    @property
    def loaded_detectors(self) -> List[str]:
        return list(self._models) + list(self._registry_detectors)