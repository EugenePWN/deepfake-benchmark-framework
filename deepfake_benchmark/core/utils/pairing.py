# deepfake_benchmark/core/utils/pairing.py
from __future__ import annotations

import logging
import random
from typing import List, Literal, Optional, Sequence, Tuple

from ...types import SampleItem

logger = logging.getLogger(__name__)

# Тип одной пары
Pair = Tuple[SampleItem, SampleItem]

# Поддерживаемые режимы — зеркало GeneratorConfig.pairing_mode
PairingMode = Literal["one_for_all", "all_vs_all", "external_sources"]


def build_pairs(
    sources: Sequence[SampleItem],
    targets: Sequence[SampleItem],
    mode: PairingMode = "one_for_all",
    seed: Optional[int] = None,
    identity_aware: bool = True,
) -> List[Pair]:
    """
    Строит список пар (source, target) по заданному режиму.

    ИСПРАВЛЕНО: оригинальная функция принимала один аргумент real_items и сама
    фильтровала источники/цели по role. Это противоречило сигнатуре вызова в
    DatasetGenerator.generate(), который передаёт уже разделённые sources и targets:
        build_pairs(sources, targets, mode=self.cfg.pairing_mode)
    Теперь функция принимает два явных списка — это соответствует вызывающему коду.

    Args:
        sources:        изображения-источники лиц (доноры)
        targets:        целевые изображения (куда вставляем лицо)
        mode:           стратегия построения пар
        seed:           зерно для воспроизводимого перемешивания (None = без перемешивания)
        identity_aware: пропускать пары, у которых source и target из одной идентичности
                        (предотвращает dataset bias — важно для честного бенчмарка)

    Режимы:
        one_for_all     — один первый source × все targets (быстро, для отладки)
        all_vs_all      — каждый source × каждый target (максимальное покрытие)
        external_sources — то же что all_vs_all, но семантически источники внешние
                           (например VGGFace2), а targets — из основного датасета

    Returns:
        Список пар (source, target). Пустой список если данных недостаточно.
    """
    sources = list(sources)
    targets = list(targets)

    if not sources:
        logger.warning("[build_pairs] sources list is empty (mode=%s)", mode)
        return []
    if not targets:
        logger.warning("[build_pairs] targets list is empty (mode=%s)", mode)
        return []

    logger.debug(
        "[build_pairs] mode=%s | sources=%d | targets=%d | identity_aware=%s",
        mode, len(sources), len(targets), identity_aware,
    )

    if mode == "one_for_all":
        pairs = _one_for_all(sources, targets)
    elif mode in ("all_vs_all", "external_sources"):
        # external_sources семантически отличается от all_vs_all только смыслом источников,
        # но логика построения пар идентична — каждый source × каждый target.
        pairs = _all_vs_all(sources, targets)
    else:
        logger.error("[build_pairs] Unknown pairing mode: %s", mode)
        return []

    # Фильтр по идентичности — предотвращает dataset bias
    if identity_aware:
        before = len(pairs)
        pairs = _filter_same_identity(pairs)
        removed = before - len(pairs)
        if removed:
            logger.info(
                "[build_pairs] identity_aware: removed %d same-identity pairs (%d remain)",
                removed, len(pairs),
            )

    # Воспроизводимое перемешивание — важно для бенчмарка
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(pairs)

    logger.info("[build_pairs] Built %d pairs (mode=%s)", len(pairs), mode)
    return pairs


# ------------------------------------------------------------------
# Внутренние стратегии
# ------------------------------------------------------------------

def _one_for_all(sources: List[SampleItem], targets: List[SampleItem]) -> List[Pair]:
    """Первый source против всех targets."""
    primary = sources[0]
    pairs = [(primary, tgt) for tgt in targets
             if tgt.media_path.resolve() != primary.media_path.resolve()]
    if len(pairs) < len(targets):
        logger.debug("[build_pairs] one_for_all: skipped 1 self-pair")
    return pairs


def _all_vs_all(sources: List[SampleItem], targets: List[SampleItem]) -> List[Pair]:
    """Каждый source против каждого target, без self-пар."""
    pairs: List[Pair] = []
    for src in sources:
        src_path = src.media_path.resolve()
        for tgt in targets:
            if tgt.media_path.resolve() == src_path:
                continue  # пропускаем self-пары
            pairs.append((src, tgt))
    return pairs


def _filter_same_identity(pairs: List[Pair]) -> List[Pair]:
    """
    Удаляет пары, у которых source и target имеют одинаковый identity-префикс.

    Соглашение об именовании sample_id: "<identity_id>_<frame_or_img_number>"
    Например: "n000001_0001" и "n000001_0002" — одна идентичность "n000001".
    Если разделитель "_" отсутствует, используется весь sample_id как identity.

    Это соглашение совпадает с форматом VGGFace2 и CelebA-HQ.
    """
    def get_identity(item: SampleItem) -> str:
        """
        Извлекает identity из sample_id.

        Поддерживаемые форматы:
          - "<identity>_<frame>"                    -> identity = "<identity>"
          - "<dataset>_<identity>_<frame>"          -> identity = "<identity>"
            (dataset может содержать "_" как в "celeba_hq")
          - "<identity>"                            -> identity = "<identity>"
        """
        sid = item.sample_id
        dataset = (item.dataset or "").strip()
        prefix = f"{dataset}_"

        # sample_id в loader формируется как "<dataset>_<stem>".
        # Если префикс датасета присутствует — убираем его перед выделением identity.
        core_id = sid[len(prefix):] if dataset and sid.startswith(prefix) else sid

        if "_" not in core_id:
            return core_id
        return core_id.rsplit("_", 1)[0]

    return [
        (src, tgt) for src, tgt in pairs
        if get_identity(src) != get_identity(tgt)
    ]