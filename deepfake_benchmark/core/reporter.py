# deepfake_benchmark/core/reporter.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from ..config import BenchmarkConfig, EvalConfig
from .metric_evaluator import DetectorMetrics, EvaluationReport

logger = logging.getLogger(__name__)


class Reporter:
    """
    Формирует итоговые отчёты из EvaluationReport.

    Поддерживаемые форматы:
      json  — metrics.json  
      csv   — metrics.csv
      html  — report.html 
      all   — все три формата

    При save_plots=True дополнительно создаёт:
      roc_curves.png      — ROC-кривые всех детекторов
      metrics_bar.png     — bar chart AUC / MCC / BalAcc / F1
      confusion_grid.png  — сетка confusion matrices
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        eval_config: Optional[EvalConfig] = None,
    ) -> None:
        self._eval_config  = eval_config or (config.eval_config if config else None)
        output_format      = "json"
        save_plots         = True
        results_root       = Path("data/results")

        if self._eval_config:
            output_format = self._eval_config.output_format
            save_plots    = self._eval_config.save_plots
            results_root  = self._eval_config.results_root
        elif config:
            output_format = "json"
            results_root  = config.output.results_root

        self.output_format = output_format
        self.save_plots    = save_plots
        self.results_root  = Path(results_root)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def generate(
        self,
        report: EvaluationReport,
        run_name: str = "eval",
    ) -> Path:
        """
        Сохраняет отчёт в results_root/<run_name>/.
        Возвращает путь к папке.
        """
        out_dir = self.results_root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        report.save_json(out_dir / "metrics.json")

        if self.output_format in ("csv", "all"):
            report.save_csv(out_dir / "metrics.csv")

        if self.output_format in ("html", "all"):
            self._save_html(report, out_dir / "report.html")

        if self.save_plots:
            self._save_plots(report, out_dir)

        # Краткая сводка в консоль
        self._print_console_summary(report)

        logger.info("[Reporter] Reports saved to: %s", out_dir)
        return out_dir

    # ------------------------------------------------------------------
    # HTML-отчёт
    # ------------------------------------------------------------------

    def _save_html(self, report: EvaluationReport, path: Path) -> None:
        all_m = report.all_metrics()
        if not all_m:
            return

        rows = ""
        for m in sorted(all_m, key=lambda x: (x.dataset, x.detector, x.preset)):
            rows += f"""
            <tr>
              <td>{m.detector}</td>
              <td>{m.dataset}</td>
              <td>{m.preset}</td>
              <td class="num">{m.auc_roc:.4f}</td>
              <td class="num">{m.mcc:.4f}</td>
              <td class="num">{m.balanced_accuracy:.4f}</td>
              <td class="num">{m.f1:.4f}</td>
              <td class="num">{m.recall:.4f}</td>
              <td class="num">{m.specificity:.4f}</td>
              <td class="num">{m.precision:.4f}</td>
              <td class="num">{m.threshold_used:.3f}</td>
              <td class="num">{m.n_samples}</td>
              <td class="num">{m.tp}/{m.tn}/{m.fp}/{m.fn}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Deepfake Benchmark — Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 32px; color: #1a1a2e; background: #f5f5f5; }}
  h1   {{ color: #16213e; }}
  table {{ border-collapse: collapse; width: 100%; background: white;
           box-shadow: 0 2px 8px rgba(0,0,0,.1); border-radius: 8px; overflow: hidden; }}
  th   {{ background: #16213e; color: white; padding: 10px 14px;
          text-align: left; font-size: 13px; }}
  td   {{ padding: 9px 14px; border-bottom: 1px solid #eee; font-size: 13px; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:hover td {{ background: #f0f4ff; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 11px; font-weight: 600; }}
  .good  {{ background: #d4edda; color: #155724; }}
  .ok    {{ background: #fff3cd; color: #856404; }}
  .bad   {{ background: #f8d7da; color: #721c24; }}
  .meta  {{ color: #666; font-size: 12px; margin-top: 8px; }}
</style>
</head>
<body>
<h1>🔍 Deepfake Benchmark — Evaluation Report</h1>
<p class="meta">Сгенерировано автоматически | {len(all_m)} записей</p>

<table>
  <thead>
    <tr>
      <th>Detector</th><th>Dataset</th><th>Preset</th>
      <th>AUC-ROC</th><th>MCC</th><th>Balanced Acc</th>
      <th>F1</th><th>Sensitivity</th><th>Specificity</th>
      <th>Precision</th><th>Threshold</th>
      <th>N</th><th>TP/TN/FP/FN</th>
    </tr>
  </thead>
  <tbody>{rows}
  </tbody>
</table>

<h2 style="margin-top:32px">Интерпретация метрик</h2>
<ul>
  <li><b>AUC-ROC</b> — threshold-free, основная метрика. &gt;0.97 отлично, &gt;0.90 хорошо.</li>
  <li><b>MCC</b> — лучшая одиночная метрика при дисбалансе классов. ∈ [-1, 1].</li>
  <li><b>Balanced Accuracy</b> — среднее recall по классам, честно при дисбалансе.</li>
  <li><b>Sensitivity</b> = TPR = Recall(fake) — доля найденных фейков.</li>
  <li><b>Specificity</b> = TNR = Recall(real) — доля правильно распознанных реальных.</li>
</ul>
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
        logger.info("[Reporter] HTML → %s", path)

    # ------------------------------------------------------------------
    # Графики
    # ------------------------------------------------------------------

    def _save_plots(self, report: EvaluationReport, out_dir: Path) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("[Reporter] matplotlib не установлен — пропускаем графики")
            return

        all_m = report.all_metrics()
        if not all_m:
            return

        detectors = sorted(set(m.detector for m in all_m))
        datasets  = sorted(set(m.dataset  for m in all_m))
        presets   = sorted(set(m.preset   for m in all_m))

        _COLORS = [
            "#4C72B0", "#DD8452", "#55A868", "#C44E52",
            "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
        ]
        color_map = {d: _COLORS[i % len(_COLORS)] for i, d in enumerate(detectors)}

        # ── ROC-кривых нет (нет сырых scores в report) ───────────────────
        # ROC строится в cross_pipeline_eval.py где есть сырые вероятности.
        # Здесь строим только агрегированные bar charts.

        # ── Bar chart: AUC, MCC, BalAcc, F1 ─────────────────────────────
        metrics_to_plot = ["auc_roc", "mcc", "balanced_accuracy", "f1"]
        metric_labels   = ["AUC-ROC", "MCC", "Balanced Acc", "F1"]

        # Один subplot на датасет × пресет
        combos = [(ds, pr) for ds in datasets for pr in presets
                  if any(m.dataset == ds and m.preset == pr for m in all_m)]

        n_combos = len(combos)
        fig, axes = plt.subplots(
            n_combos, len(metrics_to_plot),
            figsize=(5 * len(metrics_to_plot), 4 * max(n_combos, 1)),
            squeeze=False,
        )

        x = np.arange(len(detectors))
        width = 0.6

        for row, (ds, pr) in enumerate(combos):
            for col, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
                ax = axes[row][col]
                vals = []
                for det in detectors:
                    m = report.get(det, ds, pr)
                    vals.append(getattr(m, metric, 0.0) if m else 0.0)

                bars = ax.bar(x, vals, width,
                              color=[color_map[d] for d in detectors], alpha=0.8)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

                ax.set_xticks(x)
                ax.set_xticklabels(detectors, rotation=12, ha="right", fontsize=9)
                ax.set_ylim(0.0, 1.06)
                ax.set_title(f"{label}\n{ds} / {pr}", fontsize=10, fontweight="bold")
                ax.set_ylabel("Score")
                ax.grid(axis="y", alpha=0.3)

        plt.suptitle("Deepfake Detection Benchmark — Metrics", fontsize=13,
                     fontweight="bold", y=1.01)
        plt.tight_layout()
        p = out_dir / "metrics_bar.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info("[Reporter] Bar chart → %s", p)

        # ── Confusion matrix grid ─────────────────────────────────────────
        n_det   = len(detectors)
        n_combo = len(combos)
        fig, axes = plt.subplots(
            n_det, n_combo,
            figsize=(4 * max(n_combo, 1), 3.5 * max(n_det, 1)),
            squeeze=False,
        )

        for ri, det in enumerate(detectors):
            for ci, (ds, pr) in enumerate(combos):
                ax = axes[ri][ci]
                m = report.get(det, ds, pr)
                if m is None:
                    ax.axis("off")
                    continue
                cm_data = np.array([[m.tn, m.fp], [m.fn, m.tp]])
                im = ax.imshow(cm_data, cmap="Blues")
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xticklabels(["Pred: Real", "Pred: Fake"], fontsize=8)
                ax.set_yticklabels(["True: Real", "True: Fake"], fontsize=8)
                ax.set_title(f"{det}\n{ds}/{pr}", fontsize=8)
                for (r, c), val in np.ndenumerate(cm_data):
                    ax.text(c, r, str(val), ha="center", va="center",
                            fontsize=11, color="black")

        plt.suptitle("Confusion Matrices", fontsize=12, fontweight="bold")
        plt.tight_layout()
        p = out_dir / "confusion_grid.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info("[Reporter] Confusion grid → %s", p)

    # ------------------------------------------------------------------
    # Консольная сводка
    # ------------------------------------------------------------------

    def _print_console_summary(self, report: EvaluationReport) -> None:
        all_m = sorted(
            report.all_metrics(),
            key=lambda m: (m.dataset, m.preset, m.detector),
        )
        if not all_m:
            return

        print(f"\n{'═'*82}")
        print("  BENCHMARK EVALUATION REPORT")
        print(f"{'═'*82}")
        hdr = (f"  {'Detector':<20} {'Dataset':<16} {'Preset':<10} "
               f"{'AUC':>6} {'MCC':>6} {'BalAcc':>7} {'F1':>6} "
               f"{'Sens':>6} {'Spec':>6} {'Thr':>5}")
        print(hdr)
        print("  " + "─" * 78)

        prev_group = None
        for m in all_m:
            group = (m.dataset, m.preset)
            if group != prev_group and prev_group is not None:
                print()
            prev_group = group
            print(
                f"  {m.detector:<20} {m.dataset:<16} {m.preset:<10} "
                f"{m.auc_roc:>6.4f} {m.mcc:>6.4f} "
                f"{m.balanced_accuracy:>7.4f} {m.f1:>6.4f} "
                f"{m.recall:>6.4f} {m.specificity:>6.4f} "
                f"{m.threshold_used:>5.3f}"
            )
        print(f"{'═'*82}\n")