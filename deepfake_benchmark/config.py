# deepfake_benchmark/config.py
from __future__ import annotations

import sys

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BASE_DIR = Path(__file__).parent.parent.resolve()

SUPPORTED_DATASETS   = frozenset({"celeba", "celeba_hq", "vggface2", "lfw", "ffhq", "custom"})
SUPPORTED_PROVIDERS  = frozenset({"cuda", "cpu", "coreml", "directml", "openvino"})
SUPPORTED_SWAP_MODELS = frozenset({
    "inswapper_128", "inswapper_128_fp16",
    "simswap_256", "simswap_512_unofficial",
    "ghost_256_unet_1", "ghost_256_unet_2", "ghost_256_unet_3",
    "blendface_256", "uniface_256",
})


# ──────────────────────────────────────────────────────────────────────────────
# PostProcessConfig
# ──────────────────────────────────────────────────────────────────────────────

class PostProcessConfig(BaseModel):
    """
    Постобработка сгенерированного изображения через Pillow.
    Моделирует деградацию при реальном распространении фейков.

    Порядок применения: crop → resize → blur → jitter → noise → JPEG.

    Fields:
      jpeg_quality     1–95.  None = без сжатия.
                       95 ≈ без потерь · 82 = Instagram · 68 = WhatsApp
      resize_factor    Ресайз вниз и обратно. 0.75 = Telegram. None = нет.
      gaussian_blur_r  Радиус Гауссова размытия. 0 = нет. 1–2 = лёгкий.
      add_noise_std    σ аддитивного Гауссова шума. 0 = нет. 4–6 = лёгкий.
      color_jitter     ±factor для яркости/контраста/насыщенности. 0 = нет.
      random_crop_ratio  Кроп центра [0.5–1.0]. 1.0 = без кропа.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    jpeg_quality:      Optional[int]   = Field(default=None, ge=1, le=95)
    resize_factor:     Optional[float] = Field(default=None, gt=0.0, le=1.0)
    gaussian_blur_r:   int             = Field(default=0,   ge=0, le=20)
    add_noise_std:     float           = Field(default=0.0, ge=0.0, le=50.0)
    color_jitter:      float           = Field(default=0.0, ge=0.0, le=1.0)
    random_crop_ratio: float           = Field(default=1.0, ge=0.5, le=1.0)

    def is_identity(self) -> bool:
        return (
            self.jpeg_quality is None
            and self.resize_factor is None
            and self.gaussian_blur_r == 0
            and self.add_noise_std == 0.0
            and self.color_jitter == 0.0
            and self.random_crop_ratio == 1.0
        )


# ──────────────────────────────────────────────────────────────────────────────
# ModelMixEntry + GenerationPlan
# ──────────────────────────────────────────────────────────────────────────────

class ModelMixEntry(BaseModel):
    """Одна swap-модель в смеси пресета с её весом и постобработкой."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name:   str
    weight:       float = Field(default=1.0, gt=0.0)
    post_process: PostProcessConfig = Field(default_factory=PostProcessConfig)
    ff_args:      Dict[str, Any]   = Field(default_factory=dict)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if v not in SUPPORTED_SWAP_MODELS:
            raise ValueError(f"Unknown swap model: {v!r}. Supported: {SUPPORTED_SWAP_MODELS}")
        return v


class GenerationPlan(BaseModel):
    """Полное описание пресета генерации: модели + параметры FaceFusion + постобработка."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_mix: List[ModelMixEntry] = Field(default_factory=list)

    # Параметры детектора лиц
    face_detector_model: str   = "retinaface"
    face_detector_score: float = Field(default=0.7, ge=0.0, le=1.0)
    face_selector_mode:  str   = "one"

    # Параметры маски
    face_mask_types:   List[str] = Field(default_factory=lambda: ["box"])
    face_mask_blur:    float     = Field(default=0.3, ge=0.0, le=1.0)
    face_mask_padding: List[int] = Field(default_factory=lambda: [0, 0, 0, 0])

    # Face enhancer
    face_enhancer_model: Optional[str] = None
    face_enhancer_blend: int = Field(default=80, ge=0, le=100)

    # Постобработка по умолчанию (применяется если не задана в ModelMixEntry)
    default_post_process: PostProcessConfig = Field(default_factory=PostProcessConfig)

    @model_validator(mode="after")
    def validate_weights(self) -> "GenerationPlan":
        if self.model_mix:
            total = sum(e.weight for e in self.model_mix)
            if abs(total) < 1e-9:
                raise ValueError("model_mix weights sum to zero")
        return self

    def normalized_weights(self) -> List[float]:
        total = sum(e.weight for e in self.model_mix)
        return [e.weight / total for e in self.model_mix]

    def assign_models_to_pairs(
        self, n_pairs: int, seed: Optional[int] = 42
    ) -> List["ModelMixEntry"]:
        """
        Детерминированно распределяет n_pairs пар по моделям согласно весам.

        Алгоритм наибольшего остатка (метод Хэйра):
          - Гарантирует что итоговое распределение максимально точно
            соответствует заданным весам.
          - При фиксированном seed результат воспроизводим.
          - Каждой паре назначается ровно одна ModelMixEntry.

        Пример: weights=[0.6, 0.4], n_pairs=5
          → [model_A, model_A, model_A, model_B, model_B]
          (3 пары для A, 2 для B — перемешаны случайно)

        Returns:
            List[ModelMixEntry] длиной n_pairs.
        """
        import math
        import random as _random

        if not self.model_mix:
            raise ValueError("GenerationPlan.model_mix is empty")

        if len(self.model_mix) == 1:
            return [self.model_mix[0]] * n_pairs

        weights = self.normalized_weights()

        # Шаг 1: базовые целые квоты (floor)
        quotas = [int(w * n_pairs) for w in weights]
        remainders = [(w * n_pairs - q, i) for i, (w, q) in enumerate(zip(weights, quotas))]

        # Шаг 2: распределяем остаток по наибольшим дробным частям
        leftover = n_pairs - sum(quotas)
        for _, i in sorted(remainders, reverse=True)[:leftover]:
            quotas[i] += 1

        # Шаг 3: составляем список и перемешиваем детерминированно
        assignments: List[ModelMixEntry] = []
        for entry, quota in zip(self.model_mix, quotas):
            assignments.extend([entry] * quota)

        rng = _random.Random(seed)
        rng.shuffle(assignments)
        return assignments

    def to_ff_args(self) -> Dict[str, Any]:
        """
        Возвращает общие параметры FaceFusion для этого плана
        (детектор лиц, маска, enhancer) без swap-модели.
        """
        args: Dict[str, Any] = {
            "face_detector_model":  self.face_detector_model,
            "face_detector_score":  str(self.face_detector_score),
            "face_selector_mode":   self.face_selector_mode,
            "face_mask_types":      " ".join(self.face_mask_types),
            "face_mask_blur":       str(self.face_mask_blur),
            "face_mask_padding":    " ".join(str(p) for p in self.face_mask_padding),
        }
        if self.face_enhancer_model:
            args["face_enhancer_model"] = self.face_enhancer_model
            args["face_enhancer_blend"] = str(self.face_enhancer_blend)
        return args


def _make_default_plans() -> Dict[str, GenerationPlan]:
    return {
        "default": GenerationPlan(
            model_mix=[ModelMixEntry(model_name="inswapper_128", weight=1.0)],
        ),
        "easy": GenerationPlan(
            model_mix=[ModelMixEntry(
                model_name="inswapper_128", weight=1.0,
                post_process=PostProcessConfig(),
            )],
            face_detector_score=0.9,
            face_mask_types=["box"],
            face_mask_blur=0.1,
        ),
        "medium": GenerationPlan(
            model_mix=[
                ModelMixEntry(model_name="inswapper_128", weight=0.6,
                              post_process=PostProcessConfig(jpeg_quality=82, color_jitter=0.15)),
                ModelMixEntry(model_name="simswap_256",   weight=0.4,
                              post_process=PostProcessConfig(jpeg_quality=75, resize_factor=0.85)),
            ],
            face_detector_score=0.7,
            face_mask_types=["box", "occlusion"],
            face_mask_blur=0.3,
            face_mask_padding=[3, 3, 3, 3],
        ),
        "hard": GenerationPlan(
            model_mix=[
                ModelMixEntry(model_name="inswapper_128_fp16", weight=0.40,
                              ff_args={"face_enhancer_model": "gfpgan_1.4", "face_enhancer_blend": "80"},
                              post_process=PostProcessConfig(jpeg_quality=72, resize_factor=0.75,
                                                            gaussian_blur_r=1, color_jitter=0.2)),
                ModelMixEntry(model_name="simswap_512_unofficial", weight=0.35,
                              ff_args={"face_enhancer_model": "codeformer", "face_enhancer_blend": "80"},
                              post_process=PostProcessConfig(jpeg_quality=65, add_noise_std=4.0,
                                                            random_crop_ratio=0.92)),
                ModelMixEntry(model_name="ghost_256_unet_2", weight=0.25,
                              post_process=PostProcessConfig(jpeg_quality=60, resize_factor=0.6,
                                                            gaussian_blur_r=2, add_noise_std=6.0,
                                                            color_jitter=0.3)),
            ],
            face_detector_score=0.5,
            face_mask_types=["box", "occlusion", "region"],
            face_mask_blur=0.6,
            face_mask_padding=[5, 5, 5, 5],
            face_enhancer_model="gfpgan_1.4",
            face_enhancer_blend=80,
        ),
        "ultra_hard": GenerationPlan(
            model_mix=[
                ModelMixEntry(
                    model_name="inswapper_128_fp16", weight=1.0,
                    ff_args={"face_enhancer_model": "codeformer", "face_enhancer_blend": "88"},
                    post_process=PostProcessConfig(jpeg_quality=94, color_jitter=0.03),
                ),
            ],
            face_detector_score=0.6,
            face_mask_types=["region", "occlusion"],
            face_mask_blur=0.6,
            face_mask_padding=[4, 4, 4, 4],
            face_enhancer_model="codeformer",
            face_enhancer_blend=88,
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# LoaderConfig
# ──────────────────────────────────────────────────────────────────────────────

class LoaderConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_datasets:        List[str] = Field(default_factory=list)
    real_data_root:         Path      = BASE_DIR / "data" / "real"
    max_items_per_dataset:  Optional[int] = None
    external_source_dir:    Optional[Path] = None
    custom_target_dir:      Optional[Path] = None
    use_loader:             bool = True
    identity_split:         bool = True
    auto_download:          bool = False

    @field_validator("source_datasets")
    @classmethod
    def validate_datasets(cls, v: List[str]) -> List[str]:
        unknown = set(v) - SUPPORTED_DATASETS
        if unknown:
            raise ValueError(f"Unknown datasets: {unknown}. Supported: {SUPPORTED_DATASETS}")
        return v

    @field_validator("real_data_root", "external_source_dir", "custom_target_dir", mode="before")
    @classmethod
    def coerce_path(cls, v: Any) -> Optional[Path]:
        return Path(v) if v is not None else None


# ──────────────────────────────────────────────────────────────────────────────
# FaceFusion auto-detection
# ──────────────────────────────────────────────────────────────────────────────

def _find_facefusion_dir() -> Path:
    """
    Ищет папку FaceFusion по стандартным местам в порядке приоритета:
      1. <project_root>/facefusion/          — после setup_facefusion.py
      2. ~/facefusion/                        — ручная установка в домашней папке
      3. Текущая рабочая директория/facefusion/

    Возвращает первый найденный вариант с facefusion.py внутри,
    или <project_root>/facefusion/ если ничего не найдено
    (чтобы валидатор дал понятную ошибку а не молча возвращал неверный путь).
    """
    candidates = [
        BASE_DIR / "facefusion",
        Path.home() / "facefusion",
        Path.cwd() / "facefusion",
    ]
    for p in candidates:
        if (p / "facefusion.py").exists():
            return p
    # Не найдено — возвращаем дефолт, валидатор выдаст предупреждение
    return BASE_DIR / "facefusion"


def _find_facefusion_python() -> Optional[str]:
    """
    Ищет Python-окружение FaceFusion по стандартным именам conda-окружений.
    Проверяет как Windows (.exe), так и Linux/macOS пути.

    Стандартные имена окружений: facefusion, ff, deepfake
    Стандартные пути conda: ~/miniconda3, ~/anaconda3, ~/mambaforge

    Возвращает None если не найдено — тогда используется sys.executable.
    """
    import platform

    conda_roots = [
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
        Path.home() / "mambaforge",
        Path.home() / "miniforge3",
        Path("C:/ProgramData/miniconda3"),
        Path("C:/ProgramData/anaconda3"),
    ]
    env_names = ["facefusion", "ff", "deepfake", "facefusion_env"]
    is_win = platform.system() == "Windows"

    for root in conda_roots:
        for env_name in env_names:
            if is_win:
                candidate = root / "envs" / env_name / "python.exe"
            else:
                candidate = root / "envs" / env_name / "bin" / "python"
            if candidate.exists():
                return str(candidate)

    # Проверяем venv рядом с проектом
    for env_name in env_names:
        for venv_root in [BASE_DIR, Path.cwd()]:
            if is_win:
                candidate = venv_root / env_name / "Scripts" / "python.exe"
            else:
                candidate = venv_root / env_name / "bin" / "python"
            if candidate.exists():
                return str(candidate)

    return None  # используем sys.executable как fallback


# ──────────────────────────────────────────────────────────────────────────────
# GeneratorConfig
# ──────────────────────────────────────────────────────────────────────────────

class GeneratorConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    preset: Literal["easy", "medium", "hard", "ultra_hard", "default"] = "default"
    plans:  Dict[str, GenerationPlan] = Field(default_factory=_make_default_plans)

    native_args: Dict[str, Any] = Field(
        default_factory=lambda: {"execution_provider": "cuda"}
    )
    pairing_mode:            Literal["one_for_all", "all_vs_all", "external_sources"] = "one_for_all"
    max_pairs:               Optional[int] = Field(default=None, ge=1)
    include_meta:            bool = True
    output_structure:        Literal["flat", "preset_model"] = "flat"

    facefusion_dir:    Optional[Path] = None
    facefusion_python: Optional[str]  = None

    parallel:                bool = False
    parallel_workers:        int  = Field(default=2, ge=1, le=32)
    pairing_seed:            Optional[int] = 42
    identity_aware_pairing:  bool = True
    skip_existing:           bool = True
    subprocess_timeout:      int  = Field(default=300, ge=10, le=3600)

    @field_validator("facefusion_dir", mode="before")
    @classmethod
    def coerce_ff_path(cls, v: Any) -> Optional[Path]:
        return Path(v) if v is not None else None

    @model_validator(mode="after")
    def resolve_facefusion_paths(self) -> "GeneratorConfig":
        """
        Авто-определение путей FaceFusion если не заданы явно.
        Запускается после валидации всех полей.
        """
        if self.facefusion_dir is None:
            self.facefusion_dir = _find_facefusion_dir()

        if self.facefusion_python is None:
            found = _find_facefusion_python()
            if found:
                self.facefusion_python = found
            # else: 

        return self

    @property
    def resolved_python(self) -> str:
        """Python для запуска FaceFusion. Fallback — текущий интерпретатор."""
        return self.facefusion_python or sys.executable

    @property
    def resolved_base_args(self) -> Dict[str, Any]:
        """
        Базовые аргументы FaceFusion из native_args без execution_provider
        (execution_provider добавляется отдельно в _build_tasks).
        """
        return {
            k: v for k, v in self.native_args.items()
            if k != "execution_provider"
        }

    @property
    def facefusion_ready(self) -> bool:
        """True если facefusion.py найден в facefusion_dir."""
        return (self.facefusion_dir / "facefusion.py").exists()

    @property
    def active_plan(self) -> GenerationPlan:
        if self.preset not in self.plans:
            raise KeyError(f"Preset {self.preset!r} not in plans: {list(self.plans)}")
        return self.plans[self.preset]


# ──────────────────────────────────────────────────────────────────────────────
# OutputConfig — параметры сборки и сохранения датасета
# ──────────────────────────────────────────────────────────────────────────────

class OutputConfig(BaseModel):
    """
    Параметры вывода для mode=generate.

    structure:
      "train_val_test" — собираем train/val/test для обучения детектора
      "flat_eval"      — flat real/ + fake/ для валидации готовой модели

    skip_dataset_build:
      True  → только генерируем фейки в fake_data_root, без сборки датасета.
               Удобно когда хочется сначала проверить качество фейков визуально.
      False → генерируем + автоматически запускаем dataset_build (по умолчанию).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    structure:          Literal["train_val_test", "flat_eval"] = "train_val_test"
    fake_data_root:     Path = BASE_DIR / "data" / "fakes"
    results_root:       Path = BASE_DIR / "data" / "results"
    skip_dataset_build: bool = False

    # Только для structure="flat_eval"
    output_dir:    Optional[Path] = None  
    n_per_class:   Optional[int]  = None  
    copy_real:     bool           = True
    real_source:   Literal["targets", "loader"] = "targets"

    @field_validator("fake_data_root", "results_root", "output_dir", mode="before")
    @classmethod
    def coerce_paths(cls, v: Any) -> Optional[Path]:
        return Path(v) if v is not None else None


# ──────────────────────────────────────────────────────────────────────────────
# DetectorEntryConfig, EvalDatasetConfig, EvalConfig
# ──────────────────────────────────────────────────────────────────────────────

class DetectorEntryConfig(BaseModel):
    """Один детектор в режиме evaluate."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name:         str
    architecture: Literal["xception", "efficientnet", "f3net"]
    weights_path: Path
    threshold:    float          = Field(default=0.5, ge=0.0, le=1.0)
    img_size:     Optional[int]  = None
    batch_size:   int            = Field(default=16, ge=1)

    @field_validator("weights_path", mode="before")
    @classmethod
    def coerce_weights_path(cls, v: Any) -> Path:
        return Path(v)


class EvalDatasetConfig(BaseModel):
    """
    Один тестовый датасет для режима evaluate.

    mix_with: список имён других датасетов из той же EvalConfig для объединения.
    Если указан — реальные и фейки из этих датасетов смешиваются в один набор
    и оцениваются совместно. Полезно для оценки на смешанных условиях.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name:        str
    real_dir:    Path
    fake_dir:    Path
    n_per_class: Optional[int]  = Field(default=None, ge=1)
    mix_with:    List[str]      = Field(default_factory=list)

    @field_validator("real_dir", "fake_dir", mode="before")
    @classmethod
    def coerce_paths(cls, v: Any) -> Path:
        return Path(v)


class EvalConfig(BaseModel):
    """Конфигурация для режима evaluate."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    detectors:        List[DetectorEntryConfig] = Field(default_factory=list)
    eval_datasets:    List[EvalDatasetConfig]   = Field(default_factory=list)
    output_format:    Literal["json", "csv", "html", "all"] = "json"
    save_plots:       bool = True
    threshold_metric: Literal["f1", "balanced_acc", "mcc"] = "f1"
    random_seed:      int  = 42
    results_root:     Path = BASE_DIR / "data" / "results"

    @field_validator("results_root", mode="before")
    @classmethod
    def coerce_results_root(cls, v: Any) -> Path:
        return Path(v)

    @field_validator("detectors")
    @classmethod
    def unique_detector_names(cls, v: List[DetectorEntryConfig]) -> List[DetectorEntryConfig]:
        names = [d.name for d in v]
        if len(names) != len(set(names)):
            raise ValueError(f"Detector names must be unique, got duplicates: {names}")
        return v

    def mixed_datasets(self) -> List["EvalDatasetConfig"]:
        """
        Возвращает список датасетов с учётом mix_with.
        Датасеты с одинаковым mix_with объединяются в один виртуальный датасет.
        """
        by_name = {ds.name: ds for ds in self.eval_datasets}
        processed, seen = [], set()
        for ds in self.eval_datasets:
            if ds.name in seen:
                continue
            if not ds.mix_with:
                processed.append(ds)
            else:
                # Группируем ds + все mix_with в один MixedEvalDatasetConfig
                partners = [ds] + [by_name[n] for n in ds.mix_with if n in by_name]
                mixed_name = "+".join(d.name for d in partners)
                mixed = MixedEvalDatasetConfig(
                    name=mixed_name,
                    sources=partners,
                    n_per_class=ds.n_per_class,
                )
                processed.append(mixed)
                seen.update(d.name for d in partners)
            seen.add(ds.name)
        return processed


class MixedEvalDatasetConfig(BaseModel):
    """Виртуальный датасет из нескольких источников (для mix_with)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name:        str
    sources:     List[EvalDatasetConfig]
    n_per_class: Optional[int] = None
    real_dir:    Optional[Path] = None   # не используется, но нужен для совместимости
    fake_dir:    Optional[Path] = None


# ──────────────────────────────────────────────────────────────────────────────
# BenchmarkConfig — единственный, главный конфиг
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkConfig(BaseModel):
    """
    Главный конфиг бенчмарка. Один класс, без наследования от себя.

    mode:
      "generate"  — генерация фейков + сборка датасета (output.skip_dataset_build
                    позволяет пропустить сборку)
      "evaluate"  — оценка готового детектора на готовых данных
      "full"      — generate + evaluate

    Поля:
      loader       — откуда брать реальные изображения
      generator    — параметры FaceFusion и пресеты
      output       — куда сохранять и как собирать датасет
      eval_config  — детекторы и датасеты для оценки
      device       — устройство для инференса ("cuda" | "cpu")
      split_ratios — разбивка на train/val/test
      random_seed  — для воспроизводимости
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: Literal["generate", "evaluate", "full"] = "generate"

    loader:    LoaderConfig    = Field(default_factory=LoaderConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    output:    OutputConfig    = Field(default_factory=OutputConfig)

    eval_config: Optional[EvalConfig] = None

    device:      Literal["cuda", "cpu"] = "cuda"
    random_seed: int = 42

    split_ratios: Dict[str, float] = Field(
        default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15}
    )

    # Shortcut-поля для обратной совместимости со старым кодом
    # (используются в dataset_build и других местах)
    @property
    def fake_data_root(self) -> Path:
        return self.output.fake_data_root

    @property
    def results_root(self) -> Path:
        return self.output.results_root

    # Старый API: detectors как список строк (Registry)
    detectors:          List[str]              = Field(default_factory=list)
    detector_configs:   Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    detector_batch_size: int                   = Field(default=32, ge=1)

    @field_validator("split_ratios")
    @classmethod
    def validate_split_ratios(cls, v: Dict[str, float]) -> Dict[str, float]:
        required = {"train", "val", "test"}
        if missing := required - set(v.keys()):
            raise ValueError(f"split_ratios missing keys: {missing}")
        if abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must sum to 1.0, got {sum(v.values()):.4f}")
        return v

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "BenchmarkConfig":
        if self.mode in ("evaluate", "full") and self.eval_config is None:
            raise ValueError(
                f"mode='{self.mode}' requires eval_config with detectors and datasets."
            )
        if self.mode in ("evaluate", "full") and self.eval_config:
            if not self.eval_config.detectors:
                raise ValueError("eval_config.detectors cannot be empty")
            if not self.eval_config.eval_datasets:
                raise ValueError("eval_config.eval_datasets cannot be empty")
        return self