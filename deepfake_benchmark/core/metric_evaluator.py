# deepfake_benchmark/core/metric_evaluator.py
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import BenchmarkConfig, EvalConfig
from .detectors.base_detector import DetectionResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Датакласс результатов одного детектора на одном датасете
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectorMetrics:
    """
    Полный набор метрик одного детектора на одном датасете/пресете.

    threshold_used — порог, при котором вычислены precision/recall/F1/conf.matrix.
    Метрики без порога (auc_roc, auc_pr) вычисляются по всему диапазону.
    """
    detector: str
    dataset: str
    preset: str
    threshold_used: float

    # Threshold-free
    auc_roc:    float = 0.0
    auc_pr:     float = 0.0    

    # При threshold_used
    accuracy:          float = 0.0
    balanced_accuracy: float = 0.0
    mcc:               float = 0.0
    f1:                float = 0.0
    f1_weighted:       float = 0.0
    precision:         float = 0.0
    recall:            float = 0.0   
    specificity:       float = 0.0  
    fpr:               float = 0.0
    fnr:               float = 0.0

    # Confusion matrix
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    # Статистика
    n_samples: int = 0
    n_errors:  int = 0   # NaN-скоры (сбои инференса)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def summary_line(self) -> str:
        return (
            f"AUC={self.auc_roc:.4f}  MCC={self.mcc:.4f}  "
            f"BalAcc={self.balanced_accuracy:.4f}  F1={self.f1:.4f}  "
            f"Sens={self.recall:.4f}  Spec={self.specificity:.4f}  "
            f"thr={self.threshold_used:.3f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Сводный отчёт
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationReport:
    """
    Сводный отчёт по всем детекторам и датасетам.

    results[detector][dataset][preset] = DetectorMetrics
    """
    results: Dict[str, Dict[str, Dict[str, DetectorMetrics]]] = field(default_factory=dict)

    def add(self, m: DetectorMetrics) -> None:
        self.results \
            .setdefault(m.detector, {}) \
            .setdefault(m.dataset, {}) \
            [m.preset] = m

    def get(self, detector: str, dataset: str, preset: str = "overall") -> Optional[DetectorMetrics]:
        return self.results.get(detector, {}).get(dataset, {}).get(preset)

    def all_metrics(self) -> List[DetectorMetrics]:
        out = []
        for dets in self.results.values():
            for dsets in dets.values():
                out.extend(dsets.values())
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            det: {
                ds: {preset: m.to_dict() for preset, m in presets.items()}
                for ds, presets in datasets.items()
            }
            for det, datasets in self.results.items()
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("[EvaluationReport] JSON → %s", path)

    def save_csv(self, path: Path) -> None:
        import csv
        path.parent.mkdir(parents=True, exist_ok=True)
        all_m = self.all_metrics()
        if not all_m:
            return
        keys = list(all_m[0].to_dict().keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for m in all_m:
                writer.writerow(m.to_dict())
        logger.info("[EvaluationReport] CSV → %s", path)

    def print_summary(self) -> None:
        header = (
            f"  {'Detector':<20} {'Dataset':<18} {'Preset':<10} "
            f"{'AUC':>6} {'MCC':>6} {'BalAcc':>7} {'F1':>6} "
            f"{'Sens':>6} {'Spec':>6} {'Thr':>5}"
        )
        sep = "  " + "─" * (len(header) - 2)
        logger.info("[EvaluationReport] Summary:\n%s\n%s", header, sep)
        for m in self.all_metrics():
            logger.info(
                "  %-20s %-18s %-10s %6.4f %6.4f %7.4f %6.4f %6.4f %6.4f %5.3f",
                m.detector, m.dataset, m.preset,
                m.auc_roc, m.mcc, m.balanced_accuracy,
                m.f1, m.recall, m.specificity, m.threshold_used,
            )


# ──────────────────────────────────────────────────────────────────────────────
# MetricEvaluator
# ──────────────────────────────────────────────────────────────────────────────

class MetricEvaluator:
    """
    Вычисляет метрики по результатам инференса DetectorManager.

    Входные данные:
      Dict[detector_name → List[DetectionResult]]

    Выходные данные:
      EvaluationReport

    Группировка:
      По умолчанию все результаты → preset="overall".
      Если DetectionResult.meta["preset"] задан → дополнительная группировка
      (позволяет сравнивать easy / medium / hard в одном отчёте).

    Пороги:
      Если threshold=0.5 (дефолт) и задан threshold_metric →
      порог подбирается по ВСЕЙ выборке (grid search 0.2–0.7).
      Для честной оценки лучше указывать порог явно из обучения.
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        eval_config: Optional[EvalConfig] = None,
    ) -> None:
        self._eval_config = eval_config or (config.eval_config if config else None)
        self._threshold_metric = (
            self._eval_config.threshold_metric if self._eval_config else "f1"
        )

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        detections: Dict[str, List[DetectionResult]],
        dataset_name: str = "unknown",
    ) -> EvaluationReport:
        """
        Вычисляет метрики по всем детекторам.

        Args:
            detections:   {detector_name: [DetectionResult, ...]}
            dataset_name: название датасета для отчёта
        """
        report = EvaluationReport()

        if not detections:
            logger.warning("[MetricEvaluator] Empty detections — nothing to evaluate")
            return report

        for detector_name, results in detections.items():
            if not results:
                logger.warning("[MetricEvaluator] No results for '%s'", detector_name)
                continue

            logger.info("[MetricEvaluator] Evaluating '%s' on '%s' (%d results)...",
                        detector_name, dataset_name, len(results))

            # Группируем по пресету
            groups = self._group_by_preset(results)

            for preset, group in groups.items():
                # Определяем порог
                threshold = self._resolve_threshold(detector_name, group)

                m = self._compute(detector_name, dataset_name, preset, group, threshold)
                report.add(m)
                logger.info("  %s / %s / %s → %s", detector_name, dataset_name, preset,
                            m.summary_line)

        return report

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _resolve_threshold(self, detector_name: str, results: List[DetectionResult]) -> float:
        """
        Если порог не задан явно (=0.5 по умолчанию) — подбираем по grid search.
        Если задан в eval_config.detectors — используем его.
        """
        # Ищем явный порог для этого детектора в eval_config
        if self._eval_config:
            for entry in self._eval_config.detectors:
                if entry.name == detector_name and entry.threshold != 0.5:
                    return entry.threshold

        # Авто-подбор по сетке
        valid = [r for r in results if not math.isnan(r.score)]
        if not valid:
            return 0.5

        gt     = [r.binary_gt   for r in valid]
        scores = [r.score       for r in valid]
        return _find_optimal_threshold(gt, scores, metric=self._threshold_metric)

    @staticmethod
    def _group_by_preset(results: List[DetectionResult]) -> Dict[str, List[DetectionResult]]:
        groups: Dict[str, List[DetectionResult]] = {}
        for r in results:
            preset = r.meta.get("preset", "overall")
            groups.setdefault(preset, []).append(r)
        return groups

    def _compute(
        self,
        detector: str,
        dataset: str,
        preset: str,
        results: List[DetectionResult],
        threshold: float,
    ) -> DetectorMetrics:
        valid   = [r for r in results if not math.isnan(r.score)]
        n_errors = len(results) - len(valid)

        if not valid:
            return DetectorMetrics(
                detector=detector, dataset=dataset, preset=preset,
                threshold_used=threshold, n_samples=len(results), n_errors=n_errors,
            )

        gt     = [r.binary_gt for r in valid]
        scores = [r.score     for r in valid]
        preds  = [1 if s >= threshold else 0 for s in scores]

        n  = len(valid)
        tp = sum(p == 1 and g == 1 for p, g in zip(preds, gt))
        tn = sum(p == 0 and g == 0 for p, g in zip(preds, gt))
        fp = sum(p == 1 and g == 0 for p, g in zip(preds, gt))
        fn = sum(p == 0 and g == 1 for p, g in zip(preds, gt))

        eps = 1e-9
        acc  = (tp + tn) / n if n else 0.0
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)   # sensitivity / TPR
        spec = tn / (tn + fp + eps)   # TNR
        f1   = _f1(prec, rec)
        f1_w = _f1_weighted(gt, preds)
        mcc  = _mcc(tp, tn, fp, fn)
        bacc = (rec + spec) / 2.0
        fpr  = fp / (fp + tn + eps)
        fnr  = fn / (fn + tp + eps)

        auc_roc = _roc_auc(gt, scores)
        auc_pr  = _pr_auc(gt, scores)

        return DetectorMetrics(
            detector=detector, dataset=dataset, preset=preset,
            threshold_used=threshold,
            auc_roc=auc_roc, auc_pr=auc_pr,
            accuracy=acc, balanced_accuracy=bacc,
            mcc=mcc, f1=f1, f1_weighted=f1_w,
            precision=prec, recall=rec, specificity=spec,
            fpr=fpr, fnr=fnr,
            tp=tp, tn=tn, fp=fp, fn=fn,
            n_samples=n, n_errors=n_errors,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Математика — без зависимости от sklearn
# ──────────────────────────────────────────────────────────────────────────────

def _find_optimal_threshold(
    gt: List[int], scores: List[float],
    metric: str = "f1",
    lo: float = 0.2, hi: float = 0.7, steps: int = 50,
) -> float:
    best_score, best_t = -1.0, 0.5
    for t in (lo + (hi - lo) * i / steps for i in range(steps + 1)):
        preds = [1 if s >= t else 0 for s in scores]
        if metric == "f1":
            tp = sum(p == 1 and g == 1 for p, g in zip(preds, gt))
            fp = sum(p == 1 and g == 0 for p, g in zip(preds, gt))
            fn = sum(p == 0 and g == 1 for p, g in zip(preds, gt))
            score = _f1(tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9))
        elif metric == "balanced_acc":
            tp = sum(p == 1 and g == 1 for p, g in zip(preds, gt))
            tn = sum(p == 0 and g == 0 for p, g in zip(preds, gt))
            fp = sum(p == 1 and g == 0 for p, g in zip(preds, gt))
            fn = sum(p == 0 and g == 1 for p, g in zip(preds, gt))
            score = (tp / (tp + fn + 1e-9) + tn / (tn + fp + 1e-9)) / 2
        elif metric == "mcc":
            score = _mcc_from_preds(preds, gt)
        else:
            score = sum(p == g for p, g in zip(preds, gt)) / len(gt)
        if score > best_score:
            best_score, best_t = score, t
    return best_t


def _f1(precision: float, recall: float) -> float:
    d = precision + recall
    return 2 * precision * recall / d if d > 0 else 0.0


def _f1_weighted(gt: List[int], preds: List[int]) -> float:
    classes = set(gt)
    n = len(gt)
    total = 0.0
    for c in classes:
        weight = sum(g == c for g in gt) / n
        tp = sum(p == c and g == c for p, g in zip(preds, gt))
        fp = sum(p == c and g != c for p, g in zip(preds, gt))
        fn = sum(p != c and g == c for p, g in zip(preds, gt))
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        total += weight * _f1(prec, rec)
    return total


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def _mcc_from_preds(preds: List[int], gt: List[int]) -> float:
    tp = sum(p == 1 and g == 1 for p, g in zip(preds, gt))
    tn = sum(p == 0 and g == 0 for p, g in zip(preds, gt))
    fp = sum(p == 1 and g == 0 for p, g in zip(preds, gt))
    fn = sum(p == 0 and g == 1 for p, g in zip(preds, gt))
    return _mcc(tp, tn, fp, fn)


def _roc_auc(gt: List[int], scores: List[float]) -> float:
    if len(set(gt)) < 2:
        return 0.5
    pairs = sorted(zip(scores, gt), key=lambda x: -x[0])
    n_pos, n_neg = sum(gt), len(gt) - sum(gt)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    auc = tp = fp = 0.0
    prev_tpr = prev_fpr = 0.0
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr, prev_fpr = tpr, fpr
    return float(auc)


def _pr_auc(gt: List[int], scores: List[float]) -> float:
    """Average Precision (PR-AUC) для класса fake (positive=1)."""
    if len(set(gt)) < 2:
        return 0.0
    pairs = sorted(zip(scores, gt), key=lambda x: -x[0])
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    n_pos = sum(gt)
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        prec   = tp / (tp + fp)
        recall = tp / n_pos
        ap += prec * (recall - prev_recall)
        prev_recall = recall
    return float(ap)