# deepfake_benchmark/core/dataset_loader.py
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional

from ..config import BenchmarkConfig, SUPPORTED_DATASETS
from ..types import SampleItem

logger = logging.getLogger(__name__)

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# Датасеты с реализованным авто-скачиванием.
BUILTIN_DATASETS = frozenset({"lfw", "ffhq", "utkface", "celeba_hq"})

# HuggingFace-идентификаторы для авто-скачивания
_HF_DATASETS = {
    "ffhq":      {"repo": "bitmind/ffhq-256",                       "split": "train", "image_key": "image"},
    "utkface":   {"repo": "Subh775/UTKFace_demographics_V1",        "split": "train", "image_key": "image"},
    "celeba_hq": {"repo": "korexyz/celeba-hq-256x256",             "split": "train", "image_key": "image"},
}

_LEGACY_LOCAL_SUBDIR = "local"


def _has_images(folder: Path) -> bool:
    if not folder.exists():
        return False
    return any(
        p.suffix.lower() in IMAGE_EXTS
        for p in folder.rglob("*")
        if p.is_file()
    )


class DatasetLoader:
    """
    Загрузчик реальных изображений для бенчмарка.

    Три сценария:
      1. Локальные данные:  real_data_root/<name>/images/*.jpg
      2. Авто-скачивание:   auto_download: true в конфиге
         - lfw       → sklearn.datasets.fetch_lfw_people
         - ffhq      → HuggingFace bitmind/ffhq-256
         - utkface   → HuggingFace Subh775/UTKFace_demographics_V1
         - celeba_hq → HuggingFace korexyz/celeba-hq-256
      3. Внешние доноры:    external_source_dir → role="source"
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self._cfg = config.loader
        self._seed = config.random_seed
        self._root = Path(self._cfg.real_data_root).expanduser().resolve()
        self._limit: Optional[int] = self._cfg.max_items_per_dataset
        self._auto_download: bool = getattr(self._cfg, "auto_download", False)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def load_all(self) -> List[SampleItem]:
        if not self._cfg.use_loader:
            logger.info("[DatasetLoader] use_loader=False — skipping")
            return []

        all_items: List[SampleItem] = []

        for name in self._cfg.source_datasets:
            items = self._load_dataset(name)
            logger.info("[DatasetLoader] %r → %d items", name, len(items))
            all_items.extend(items)

        if self._cfg.external_source_dir:
            ext_items = self._load_external_sources()
            logger.info("[DatasetLoader] external_sources → %d items", len(ext_items))
            all_items.extend(ext_items)

        logger.info("[DatasetLoader] total real items: %d", len(all_items))
        return all_items

    # ------------------------------------------------------------------
    # Роутинг
    # ------------------------------------------------------------------

    def _load_dataset(self, name: str) -> List[SampleItem]:
        name_lower = name.lower()

        # 1. Стандартный путь: real_data_root/<name>/images/
        standard_folder = self._root / name_lower / "images"
        if _has_images(standard_folder):
            return self._load_from_folder(name_lower, standard_folder, role=None)

        # 2. Пробуем без подпапки images (если пользователь положил прямо в <name>/)
        flat_folder = self._root / name_lower
        if _has_images(flat_folder) and flat_folder != standard_folder:
            return self._load_from_folder(name_lower, flat_folder, role=None)

        # 3. Legacy путь: real_data_root/local/<name>/images/
        legacy_folder = self._root / _LEGACY_LOCAL_SUBDIR / name_lower / "images"
        if _has_images(legacy_folder):
            logger.debug("[DatasetLoader] %r: using legacy path %s", name, legacy_folder)
            return self._load_from_folder(name_lower, legacy_folder, role=None)

        # 4. Авто-скачивание если включено
        if self._auto_download and name_lower in BUILTIN_DATASETS:
            return self._download_and_load(name_lower, standard_folder)

        # 5. Ничего не нашли
        if name_lower in BUILTIN_DATASETS:
            logger.warning(
                "[DatasetLoader] %r: no images found. "
                "Set auto_download: true in config to download automatically, "
                "or place images in %s",
                name, standard_folder,
            )
        else:
            logger.warning(
                "[DatasetLoader] %r: no images found in %s. "
                "Place images manually.",
                name, standard_folder,
            )
        return []

    # ------------------------------------------------------------------
    # Авто-скачивание
    # ------------------------------------------------------------------

    def _download_and_load(self, name: str, images_dir: Path) -> List[SampleItem]:
        logger.info(
            "[DatasetLoader] %r: auto_download=True, downloading into %s ...",
            name, images_dir,
        )

        if name == "lfw":
            self._download_lfw(images_dir)
        elif name in _HF_DATASETS:
            self._download_from_hf(name, images_dir)
        else:
            logger.warning("[DatasetLoader] no downloader for %r", name)
            return []

        if _has_images(images_dir):
            return self._load_from_folder(name, images_dir, role=None)

        logger.warning("[DatasetLoader] download of %r failed or produced no images", name)
        return []

    def _download_from_hf(self, name: str, images_dir: Path) -> None:
        """Скачивает датасет с HuggingFace через библиотеку datasets."""
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            logger.error(
                "[DatasetLoader] pip install datasets  — required for auto_download. "
                "Install: pip install datasets --break-system-packages"
            )
            return

        info = _HF_DATASETS[name]
        repo = info["repo"]
        split = info["split"]
        image_key = info["image_key"]

        logger.info("[DatasetLoader] downloading %s from HuggingFace (%s)...", name, repo)
        try:
            ds = load_dataset(repo, split=split)
        except Exception as exc:
            logger.error("[DatasetLoader] download failed for %r: %s", name, exc)
            return

        images_dir.mkdir(parents=True, exist_ok=True)
        n = len(ds) if not self._limit else min(self._limit, len(ds))

        logger.info("[DatasetLoader] saving %d images to %s ...", n, images_dir)
        for i, sample in enumerate(ds.select(range(n))):
            img = sample[image_key]
            _save_pil(img, images_dir / f"{i:06d}.jpg")

        logger.info("[DatasetLoader] %s: saved %d images to %s", name, n, images_dir)

    def _download_lfw(self, images_dir: Path) -> None:
        """Скачивает LFW через sklearn."""
        try:
            import numpy as np
            from PIL import Image as PILImage
            from sklearn.datasets import fetch_lfw_people  # type: ignore
        except ImportError:
            logger.error(
                "[DatasetLoader] pip install scikit-learn pillow numpy  — required for LFW download"
            )
            return

        logger.info("[DatasetLoader] downloading LFW via sklearn...")
        data = fetch_lfw_people(color=True, resize=1.0, funneled=True)
        arr = data.images
        images_dir.mkdir(parents=True, exist_ok=True)
        n = arr.shape[0] if not self._limit else min(self._limit, arr.shape[0])

        for i in range(n):
            img_arr = arr[i]
            if img_arr.dtype != np.uint8:
                maxv = float(img_arr.max())
                if maxv <= 1.0:
                    img_arr = (img_arr * 255.0).round().astype(np.uint8)
                else:
                    img_arr = np.clip(img_arr, 0, 255).round().astype(np.uint8)
            if img_arr.ndim == 2:
                img_arr = np.stack([img_arr] * 3, axis=-1)
            elif img_arr.shape[2] == 1:
                img_arr = np.repeat(img_arr, 3, axis=2)
            _save_pil(PILImage.fromarray(img_arr, mode="RGB"), images_dir / f"{i:06d}.jpg")

        logger.info("[DatasetLoader] LFW: saved %d images to %s", n, images_dir)

    # ------------------------------------------------------------------
    # Загрузка из папки
    # ------------------------------------------------------------------

    def _load_from_folder(
        self,
        dataset_name: str,
        folder: Path,
        role: Optional[str],
    ) -> List[SampleItem]:
        if not folder.exists():
            logger.warning("[DatasetLoader] folder not found: %s", folder)
            return []

        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

        if not files:
            logger.warning("[DatasetLoader] no images in %s", folder)
            return []

        if self._limit is not None:
            rng = random.Random(self._seed)
            files = rng.sample(files, min(self._limit, len(files)))
            files.sort()

        logger.info("[DatasetLoader] %r: %d files from %s", dataset_name, len(files), folder)

        items: List[SampleItem] = []
        for path in files:
            sample_id = f"{dataset_name}_{path.stem}"
            items.append(SampleItem(
                sample_id=sample_id,
                media_path=path,
                label="real",
                generator="none",
                source_id=None,
                target_id=None,
                role=role,
                dataset=dataset_name,
                split=None,
                meta={},
            ))
        return items

    def _load_external_sources(self) -> List[SampleItem]:
        ext_dir = self._cfg.external_source_dir
        if not ext_dir:
            return []
        ext_dir = Path(ext_dir).expanduser().resolve()
        if not ext_dir.exists():
            logger.warning("[DatasetLoader] external_source_dir not found: %s", ext_dir)
            return []
        return self._load_from_folder("external", ext_dir, role="source")


# ------------------------------------------------------------------
# Вспомогательная функция сохранения
# ------------------------------------------------------------------

def _save_pil(img, path: Path) -> None:
    from PIL import Image as PILImage
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(path, format="JPEG", quality=95, optimize=True)