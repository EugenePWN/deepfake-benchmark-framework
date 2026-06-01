# deepfake_benchmark/core/detectors/base_detector.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...types import SampleItem


@dataclass
class DetectionResult:
    """
    Результат инференса одного детектора на одном изображении.

    Поля:
      sample_id   — ID изображения (из SampleItem.sample_id)
      detector    — имя детектора (из BaseDetector.name)
      score       — вероятность того, что изображение является фейком [0.0, 1.0]
      label       — предсказанный класс: "fake" или "real" (по порогу threshold)
      gt_label    — истинная метка из SampleItem.label ("fake" / "real")
      meta        — дополнительные данные (время инференса, промежуточные активации и т.д.)
    """
    sample_id: str
    detector: str
    score: float                        # P(fake) ∈ [0.0, 1.0]
    label: str                          # "fake" | "real"  (после применения порога)
    gt_label: str                       # истинная метка из датасета
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_correct(self) -> bool:
        return self.label == self.gt_label

    @property
    def binary_pred(self) -> int:
        """1 = fake, 0 = real — удобно для sklearn."""
        return 1 if self.label == "fake" else 0

    @property
    def binary_gt(self) -> int:
        return 1 if self.gt_label == "fake" else 0


class BaseDetector(ABC):
    """
    Абстрактный базовый класс для всех детекторов дипфейков.

    Контракт:
      - Каждый детектор реализует predict_one() или predict_batch().
      - predict_batch() по умолчанию вызывает predict_one() в цикле —
        переопределяй для батч-оптимизации (GPU-инференс, ONNX и т.д.)
      - load() вызывается один раз при регистрации в DetectorManager.
      - threshold задаёт порог для бинаризации score → label.

    Как добавить новый детектор:
      1. Создай файл deepfake_benchmark/core/detectors/my_detector.py
      2. Унаследуй от BaseDetector
      3. Реализуй name, load(), predict_one()
      4. Зарегистрируй в DetectorRegistry (detector_manager.py)

    Пример:
      class XceptionDetector(BaseDetector):
          name = "xception"

          def load(self, weights_path: Path, **kwargs) -> None:
              self.model = build_xception(...)
              self.model.load_state_dict(torch.load(weights_path))
              self.model.eval()

          def predict_one(self, item: SampleItem) -> DetectionResult:
              img = preprocess(item.media_path)
              with torch.no_grad():
                  score = torch.softmax(self.model(img), dim=1)[0, 1].item()
              return self._make_result(item, score)
    """

    # Имя детектора — уникальный идентификатор, используется как ключ в реестре.
    # ОБЯЗАТЕЛЬНО переопределить в подклассе.
    name: str = ""

    def __init__(self, threshold: float = 0.5) -> None:
        if not self.name:
            raise ValueError(
                f"{self.__class__.__name__} must define a non-empty class attribute `name`"
            )
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.threshold = threshold
        self._loaded = False

    # ------------------------------------------------------------------
    # Абстрактные методы — обязательны к реализации
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, weights_path: Optional[Path] = None, **kwargs: Any) -> None:
        """
        Загружает веса модели и подготавливает детектор к инференсу.
        Вызывается один раз из DetectorManager перед запуском detect().
        После успешной загрузки должен установить self._loaded = True.
        """

    @abstractmethod
    def predict_one(self, item: SampleItem) -> DetectionResult:
        """
        Запускает инференс на одном изображении.
        Возвращает DetectionResult с заполненными score и label.
        """

    # ------------------------------------------------------------------
    # Батч-инференс (переопределяй для оптимизации)
    # ------------------------------------------------------------------

    def predict_batch(self, items: List[SampleItem]) -> List[DetectionResult]:
        """
        Батч-инференс. По умолчанию — последовательный вызов predict_one().
        Переопредели для GPU-батчинга или ONNX-инференса.
        """
        return [self.predict_one(item) for item in items]

    # ------------------------------------------------------------------
    # Вспомогательные методы для подклассов
    # ------------------------------------------------------------------

    def _make_result(
        self,
        item: SampleItem,
        score: float,
        meta: Optional[Dict[str, Any]] = None,
    ) -> DetectionResult:
        """
        Фабричный метод — строит DetectionResult из score.
        Применяет порог и заполняет gt_label из item.label.
        Используй в predict_one() вместо ручного создания DetectionResult.
        """
        score = float(max(0.0, min(1.0, score)))   # зажимаем в [0, 1]
        label = "fake" if score >= self.threshold else "real"
        # Пробрасываем item.meta в результат, чтобы оценщик мог группировать
        # по preset/split и другим полям, которые уже есть в SampleItem.
        merged_meta = dict(item.meta or {})
        if meta:
            merged_meta.update(meta)

        return DetectionResult(
            sample_id=item.sample_id,
            detector=self.name,
            score=score,
            label=label,
            gt_label=item.label,
            meta=merged_meta,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, threshold={self.threshold})"