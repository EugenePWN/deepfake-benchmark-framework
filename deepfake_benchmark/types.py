# deepfake_benchmark/types.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SampleRole = Optional[Literal["source", "target", "generated"]]


class SampleItem(BaseModel):
    """
    Единый формат одного примера (real или fake), который проходит через весь пайплайн.

    Используется:
      - DatasetLoader      (real, label="real", generator="none")
      - DatasetGenerator   (fake, label="fake", generator="facefusion", meta={...})
      - DetectorManager    (инференс на media_path)
      - MetricEvaluator    (расчёт метрик по label и предикциям)
      - Reporter           (формирование отчётов)

    Поля:
      sample_id   — уникальный идентификатор примера (обычно stem имени файла)
      media_path  — абсолютный путь к изображению/видеофрейму
      label       — "real" или "fake"
      generator   — имя генератора ("none" для реальных, "facefusion" и т.д. для фейков)
      source_id   — sample_id донора лица (только для фейков)
      target_id   — sample_id целевого изображения (только для фейков)
      role        — роль в пайплайне: "source" | "target" | "generated" | None
      dataset     — имя исходного датасета ("celeba_hq", "vggface2", ...)
      split       — принадлежность к сплиту: "train" | "val" | "test" | None
      meta        — произвольные метаданные (пресет, параметры генерации и т.д.)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_id: str
    media_path: Path
    label: Literal["real", "fake"]

    generator: str = "none"
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    role: SampleRole = None
    dataset: str
    split: Optional[Literal["train", "val", "test"]] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("media_path", mode="before")
    @classmethod
    def coerce_media_path(cls, v: Any) -> Path:
        return Path(v)

    @field_validator("sample_id")
    @classmethod
    def validate_sample_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("sample_id must be a non-empty string")
        return v.strip()

    # ------------------------------------------------------------------
    # Удобные свойства для использования в DetectorManager / MetricEvaluator
    # ------------------------------------------------------------------

    @property
    def is_fake(self) -> bool:
        return self.label == "fake"

    @property
    def is_real(self) -> bool:
        return self.label == "real"

    @property
    def binary_label(self) -> int:
        """0 = real, 1 = fake — удобно для sklearn/torch метрик."""
        return 1 if self.label == "fake" else 0