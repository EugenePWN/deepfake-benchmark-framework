# deepfake_benchmark/core/dataset_generator.py
from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from ..config import BenchmarkConfig, GenerationPlan, ModelMixEntry, PostProcessConfig
from ..types import SampleItem
from .utils.pairing import build_pairs

logger = logging.getLogger(__name__)

_Pair = Tuple[SampleItem, SampleItem]


def _resolve_facefusion_python(facefusion_python: Optional[Any]) -> str:
    """
    Разрешает GeneratorConfig.facefusion_python в путь к исполняемому python.
    """
    if facefusion_python is None:
        return sys.executable
    p = Path(facefusion_python).expanduser()
    if p.is_file():
        return str(p.resolve())
    if p.is_dir():
        candidates = [
            p / "python.exe",
            p / "Scripts" / "python.exe",
            p / "bin" / "python",
            p / "bin" / "python3",
        ]
        for c in candidates:
            if c.is_file():
                resolved = str(c.resolve())
                logger.info("[DatasetGenerator] facefusion_python directory → %s", resolved)
                return resolved
        raise FileNotFoundError(
            f"facefusion_python указывает на каталог {p}, но python не найден "
            f"(ожидались: python.exe, Scripts\\\\python.exe, bin/python). "
            f"Укажи полный путь к python.exe этого окружения."
        )
    return str(p.resolve())


def _apply_post_process(img_path: Path, cfg: PostProcessConfig) -> None:
    """Постобработка сгенерированного изображения. Все операции — in-place."""
    if cfg.is_identity:
        return
    try:
        from PIL import Image, ImageFilter, ImageEnhance
        import random as _random
    except ImportError:
        logger.warning("[PostProcess] Pillow not installed — skipping")
        return

    img = Image.open(img_path).convert("RGB")
    orig_size = img.size

    # 1. Random crop → resize back
    if cfg.random_crop_ratio < 1.0:
        w, h = orig_size
        new_w, new_h = int(w * cfg.random_crop_ratio), int(h * cfg.random_crop_ratio)
        left, top = (w - new_w) // 2, (h - new_h) // 2
        img = img.crop((left, top, left + new_w, top + new_h)).resize(orig_size, Image.LANCZOS)

    # 2. Resize down + back (имитация мессенджера)
    if cfg.resize_factor and cfg.resize_factor < 1.0:
        sw = max(1, int(orig_size[0] * cfg.resize_factor))
        sh = max(1, int(orig_size[1] * cfg.resize_factor))
        img = img.resize((sw, sh), Image.LANCZOS).resize(orig_size, Image.LANCZOS)

    # 3. Gaussian blur
    if cfg.gaussian_blur_r > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=cfg.gaussian_blur_r))

    # 4. Color jitter
    if cfg.color_jitter > 0.0:
        rng = _random.Random()
        for Enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            factor = 1.0 + rng.uniform(-cfg.color_jitter, cfg.color_jitter)
            img = Enhancer(img).enhance(max(0.1, factor))

    # 5. Gaussian noise
    if cfg.add_noise_std > 0.0:
        try:
            import numpy as np
            arr = np.array(img, dtype=np.float32)
            arr = np.clip(arr + np.random.normal(0, cfg.add_noise_std, arr.shape), 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        except ImportError:
            logger.debug("[PostProcess] numpy not available — skipping noise")

    # 6. JPEG compression (последней)
    if cfg.jpeg_quality is not None:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=cfg.jpeg_quality, optimize=True)
        buf.seek(0)
        img = Image.open(buf).copy()

    img.save(img_path, format="JPEG", quality=95, optimize=True)


def _worker_generate(
    src_path: str,
    tgt_path: str,
    out_path: str,
    ff_dir: str,
    python_exec: str,
    resolved_args: Dict[str, Any],
    post_process_dict: Dict[str, Any],
    include_meta: bool,
    src_id: str,
    tgt_id: str,
    dataset: str,
    model_name: str,
    preset: str,
    skip_existing: bool,
    timeout: int,
) -> Optional[Dict[str, Any]]:
    """Воркер: один вызов FaceFusion + постобработка."""
    out = Path(out_path)

    if skip_existing and out.exists():
        return _make_result(out_path, src_id, tgt_id, dataset, model_name, preset,
                            resolved_args, include_meta)

    cli = [python_exec, "facefusion.py", "headless-run",
           "--source-path", src_path, "--target-path", tgt_path, "--output-path", out_path]
    for k, v in resolved_args.items():
        if v is None:
            continue
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                continue
            cli.append(flag)
            cli.extend(str(item) for item in v)
        else:
            cli.extend([flag, str(v)])

    env = os.environ.copy()
    py_path = Path(python_exec).resolve()
    conda_root = py_path.parent if py_path.parent.name.lower() != "scripts" else py_path.parent.parent
    lib_bin = conda_root / "Library" / "bin"
    if lib_bin.exists():
        env["PATH"] = str(lib_bin) + os.pathsep + env.get("PATH", "")

    try:
        proc = subprocess.run(cli, cwd=ff_dir, capture_output=True, text=True,
                              env=env, timeout=timeout)
        if proc.returncode != 0:
            logger.warning("[Worker] FaceFusion failed src=%s: %s",
                           src_id, proc.stderr.strip()[-500:])
            return None
        if not out.exists():
            logger.warning("[Worker] FaceFusion exited 0 but output missing: %s", out_path)
            return None

        pp_cfg = PostProcessConfig(**post_process_dict)
        if not pp_cfg.is_identity:
            _apply_post_process(out, pp_cfg)

        return _make_result(out_path, src_id, tgt_id, dataset, model_name, preset,
                            resolved_args, include_meta)

    except subprocess.TimeoutExpired:
        logger.warning("[Worker] timeout (%ds) src=%s tgt=%s", timeout, src_id, tgt_id)
        _safe_unlink(out_path)
        return None
    except Exception as exc:
        logger.error("[Worker] unexpected error src=%s: %s", src_id, exc, exc_info=True)
        return None


def _make_result(out_path, src_id, tgt_id, dataset, model_name, preset,
                 resolved_args, include_meta):
    meta = {}
    if include_meta:
        meta = {
            "swap_model": model_name,
            "preset": preset,
            "ff_args": {k: v for k, v in resolved_args.items() if k != "execution_provider"},
        }
    return {"out_path": out_path, "meta": meta, "src_id": src_id, "tgt_id": tgt_id,
            "dataset": dataset, "model_name": model_name, "preset": preset}


def _safe_unlink(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


class DatasetGenerator:
    """
    Генерирует фейковые изображения через FaceFusion.

    Ключевые возможности:
      - Смесь моделей с весами (70% inswapper + 30% simswap)
      - Детальные пресеты (easy/medium/hard) через GenerationPlan
      - Постобработка после FaceFusion (JPEG, ресайз, шум, blur, color jitter)
      - output_structure="preset_model" — data/fakes/<preset>/<model_name>/

    Логика разрешения аргументов на пару:
      base_args + plan.to_ff_args() + entry.ff_args + face_swapper_model + execution_provider
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.cfg = config.generator
        self.ff_dir = Path(self.cfg.facefusion_dir).resolve()
        self.ff_py = self.ff_dir / "facefusion.py"
        self.python = _resolve_facefusion_python(self.cfg.facefusion_python)
        self._seed = config.random_seed

        if not self.ff_py.exists():
            raise FileNotFoundError(
                f"facefusion.py not found at {self.ff_py}. "
                "Проверь GeneratorConfig.facefusion_dir."
            )

    def generate(
        self,
        sources: Sequence[SampleItem],
        targets: Sequence[SampleItem],
        output_dir: Path,
    ) -> List[SampleItem]:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        pairs: List[_Pair] = build_pairs(
            sources, targets,
            mode=self.cfg.pairing_mode,
            seed=self.cfg.pairing_seed,
            identity_aware=self.cfg.identity_aware_pairing,
        )
        if not pairs:
            logger.warning("[DatasetGenerator] No valid pairs (mode=%s, src=%d, tgt=%d)",
                           self.cfg.pairing_mode, len(sources), len(targets))
            return []

        # Two possible goals:
        # - "successes": generate N successful fakes (may repeat targets)
        # - "target_coverage": generate at least one successful fake for each target
        # For external_sources pairing, target_coverage is the fairest default.
        goal_value = self.cfg.max_pairs
        goal_mode = "target_coverage" if (goal_value and self.cfg.pairing_mode == "external_sources") else "successes"
        desired_targets: Optional[Set[str]] = None
        if goal_mode == "target_coverage" and goal_value:
            # Targets passed into generate() are already limited by LoaderConfig.max_items_per_dataset.
            # We want coverage over this set (or at least goal_value if someone passes a smaller list).
            desired_targets = {t.sample_id for t in targets}

        plan = self.cfg.active_plan
        assignments = plan.assign_models_to_pairs(len(pairs), seed=self._seed)
        provider = self._resolve_provider()
        use_parallel = self.cfg.parallel and provider == "cpu"

        # Логируем распределение моделей
        from collections import Counter
        dist = Counter(e.model_name for e in assignments)
        logger.info(
            "[DatasetGenerator] preset=%s | candidate_pairs=%d | distribution=%s | provider=%s | mode=%s",
            self.cfg.preset, len(pairs), dict(dist), provider,
            "parallel" if use_parallel else "sequential",
        )
        if goal_value:
            if goal_mode == "target_coverage":
                total_targets = len(desired_targets or [])
                logger.info(
                    "[DatasetGenerator] goal: cover targets_total=%d (need >=1 fake per target)",
                    total_targets,
                )
            else:
                logger.info(
                    "[DatasetGenerator] goal: need %d successful fakes (generator will keep trying new pairs until goal is reached)",
                    goal_value,
                )

        tasks = self._build_tasks(pairs, assignments, plan, output_dir, provider)
        if use_parallel:
            # Parallel mode is only used on CPU in this project; keep the simpler "successes" goal there.
            return self._run_parallel(tasks, target_successes=goal_value if goal_mode == "successes" else None)
        return self._run_sequential(
            tasks,
            target_successes=goal_value if goal_mode == "successes" else None,
            desired_targets=desired_targets if goal_mode == "target_coverage" else None,
        )

    def _resolve_provider(self) -> str:
        target = self.cfg.native_args.get("execution_provider", "cuda")
        if target == "cuda" and not torch.cuda.is_available():
            logger.warning("[DatasetGenerator] CUDA unavailable — falling back to CPU")
            return "cpu"
        return target

    def _resolve_output_path(self, output_dir: Path, src: SampleItem,
                              tgt: SampleItem, model_name: str) -> Path:
        filename = f"{src.sample_id}_to_{tgt.sample_id}.jpg"
        if self.cfg.output_structure == "preset_model":
            return output_dir / self.cfg.preset / model_name / filename
        return output_dir / filename

    def _build_tasks(self, pairs, assignments, plan: GenerationPlan,
                     output_dir: Path, provider: str) -> List[tuple]:
        base_args = self.cfg.resolved_base_args
        plan_args = plan.to_ff_args()
        tasks = []
        for (src, tgt), entry in zip(pairs, assignments):
            out_path = self._resolve_output_path(output_dir, src, tgt, entry.model_name)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            resolved = {
                **base_args,
                **plan_args,
                **entry.ff_args,
                "face_swapper_model": entry.model_name,
                "execution_provider": provider,
            }

            pp = (entry.post_process
                  if not entry.post_process.is_identity
                  else plan.default_post_process)

            tasks.append((
                str(src.media_path), str(tgt.media_path), str(out_path),
                str(self.ff_dir), self.python,
                resolved, pp.model_dump(),
                self.cfg.include_meta,
                src.sample_id, tgt.sample_id, src.dataset,
                entry.model_name, self.cfg.preset,
                self.cfg.skip_existing, self.cfg.subprocess_timeout,
            ))
        return tasks

    def _run_sequential(
        self,
        tasks: List[tuple],
        target_successes: Optional[int] = None,
        desired_targets: Optional[Set[str]] = None,
    ) -> List[SampleItem]:
        fakes, total = [], len(tasks)
        covered_targets: Set[str] = set()
        if desired_targets is not None:
            logger.info(
                "[DatasetGenerator] progress: targets_covered=0, targets_total=%d",
                len(desired_targets),
            )
        elif target_successes:
            logger.info("[DatasetGenerator] progress: generated=0, remaining_to_goal=%d", target_successes)

        for idx, task in enumerate(tasks, 1):
            # task layout (see _build_tasks): ... src_id, tgt_id, ...
            tgt_id = task[9]
            if desired_targets is not None and tgt_id in covered_targets:
                continue

            if idx % 50 == 0 or idx == total:
                if desired_targets is not None:
                    remaining = max(len(desired_targets) - len(covered_targets), 0)
                    logger.info(
                        "[DatasetGenerator] attempts=%d/%d | targets_covered=%d | remaining_targets=%d",
                        idx, total, len(covered_targets), remaining,
                    )
                elif target_successes:
                    remaining = max(target_successes - len(fakes), 0)
                    logger.info(
                        "[DatasetGenerator] attempts=%d/%d | generated=%d | remaining_to_goal=%d",
                        idx, total, len(fakes), remaining,
                    )
                else:
                    logger.info("[DatasetGenerator] %d/%d processed", idx, total)
            if res := _worker_generate(*task):
                fakes.append(self._to_sample_item(res))
                if desired_targets is not None:
                    covered_targets.add(res["tgt_id"])
                    if desired_targets.issubset(covered_targets):
                        logger.info(
                            "[DatasetGenerator] goal reached: targets_covered=%d/%d (after %d attempts)",
                            len(covered_targets), len(desired_targets), idx,
                        )
                        break
                elif target_successes and len(fakes) >= target_successes:
                    logger.info(
                        "[DatasetGenerator] goal reached: requested=%d, generated=%d, remaining_to_goal=0 (after %d attempts)",
                        target_successes, len(fakes), idx,
                    )
                    break
        if desired_targets is not None:
            remaining = max(len(desired_targets) - len(covered_targets), 0)
            logger.info(
                "[DatasetGenerator] result: targets_covered=%d/%d, remaining_targets=%d",
                len(covered_targets), len(desired_targets), remaining,
            )
            if remaining > 0:
                logger.warning(
                    "[DatasetGenerator] goal not reached: missing_targets=%d (all candidate pairs exhausted)",
                    remaining,
                )
        elif target_successes:
            remaining = max(target_successes - len(fakes), 0)
            logger.info(
                "[DatasetGenerator] result: requested=%d, generated=%d, remaining_to_goal=%d",
                target_successes, len(fakes), remaining,
            )
        else:
            logger.info("[DatasetGenerator] done: %d/%d succeeded", len(fakes), total)
        if target_successes and len(fakes) < target_successes:
            logger.warning(
                "[DatasetGenerator] goal not reached: requested=%d, generated=%d, need_to_add=%d (all candidate pairs exhausted)",
                target_successes, len(fakes), target_successes - len(fakes),
            )
        return fakes

    def _run_parallel(self, tasks: List[tuple], target_successes: Optional[int] = None) -> List[SampleItem]:
        fakes = []
        if target_successes:
            logger.info("[DatasetGenerator] progress: generated=0, remaining_to_goal=%d", target_successes)
        workers = min(self.cfg.parallel_workers, os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker_generate, *t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    if res := future.result():
                        fakes.append(self._to_sample_item(res))
                        if target_successes and len(fakes) >= target_successes:
                            logger.info(
                                "[DatasetGenerator] goal reached: requested=%d, generated=%d, remaining_to_goal=0",
                                target_successes, len(fakes),
                            )
                            for pending in futures:
                                pending.cancel()
                            break
                except Exception as exc:
                    logger.error("[DatasetGenerator] future error: %s", exc, exc_info=True)
        if target_successes:
            remaining = max(target_successes - len(fakes), 0)
            logger.info(
                "[DatasetGenerator] result: requested=%d, generated=%d, remaining_to_goal=%d",
                target_successes, len(fakes), remaining,
            )
        else:
            logger.info("[DatasetGenerator] parallel done: %d/%d succeeded", len(fakes), len(tasks))
        if target_successes and len(fakes) < target_successes:
            logger.warning(
                "[DatasetGenerator] goal not reached: requested=%d, generated=%d, need_to_add=%d (all candidate pairs exhausted)",
                target_successes, len(fakes), target_successes - len(fakes),
            )
        return fakes

    @staticmethod
    def _to_sample_item(res: dict) -> SampleItem:
        return SampleItem(
            sample_id=Path(res["out_path"]).stem,
            media_path=Path(res["out_path"]),
            label="fake",
            generator=f"facefusion/{res['model_name']}",
            source_id=res["src_id"],
            target_id=res["tgt_id"],
            role="generated",
            dataset=res["dataset"],
            meta=res.get("meta", {}),
        )