"""Generate a comparison from actual saved predictions, without invented scores."""

import json
from pathlib import Path

from . import data  # Configure local matplotlib cache before importing pyplot.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve


def write_report(output, results, probabilities, labels, intervals, selected):
    output = Path(output)
    lines = ["# Civil Comments experiment results", "",
             "All three checkpoints use the same deduplicated, stratified test subset.",
             "The chosen model and operating thresholds were locked using validation before test evaluation.", "",
             "| Model | Operating point | AP | ROC-AUC | Toxic P | Toxic R | Toxic F1 | Macro F1 | Accuracy |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, points in results.items():
        for point, item in points.items():
            values = [item[k] for k in ("average_precision", "roc_auc", "toxic_precision", "toxic_recall", "toxic_f1", "macro_f1", "accuracy")]
            lines.append(f"| {name} | {point} (t={item['threshold']:.4f}) | " + " | ".join(f"{x:.4f}" for x in values) + " |")
    lines += ["", "AP is average precision, a non-interpolated summary of the precision–recall curve.",
              "The fixed-0.5 rows compare model changes at a common operating point. The validation-threshold rows",
              "compare the same threshold-selection procedure applied separately to every model.", "",
              f"Hyperparameter-search winner: **{selected['origin']}**, chosen by selection AP = {selected['selection_average_precision']:.4f}.",
              "The untuned configuration is retained as an incumbent; the search may keep it if none of the sampled candidates improves selection AP.", "",
              "## Changes relative to the frozen baseline", ""]
    baseline = results["frozen"]["validation_threshold"]
    for name in ("untuned", "tuned"):
        metrics = results[name]["validation_threshold"]
        ap_delta = metrics["average_precision"] - baseline["average_precision"]
        f1_delta = metrics["toxic_f1"] - baseline["toxic_f1"]
        ci = intervals[name]
        ap_ci, f1_ci = ci["average_precision_difference_95_ci"], ci["toxic_f1_difference_95_ci"]
        lines.append(f"- {name}: ΔAP = {ap_delta:+.4f} (95% paired bootstrap CI [{ap_ci[0]:+.4f}, {ap_ci[1]:+.4f}]); "
                     f"Δtoxic F1 = {f1_delta:+.4f} (CI [{f1_ci[0]:+.4f}, {f1_ci[1]:+.4f}]).")
    lines += ["", "A confidence interval spanning zero does not establish an improvement. These intervals condition on the",
              "fitted models and fixed thresholds; they do not include retraining variance. This is a one-seed, reduced-data experiment.",
              "A real-world deployment claim requires independent, representative data and repeated training seeds.", "",
              "![Precision–recall comparison](precision_recall.png)", "",
              "![Confusion matrices](confusion_matrices.png)", "",
              "![Training selection curves](selection_curves.png)", "",
              "Run `python -m toxicity.explain --run <this-directory>` for sourced real-world predictions and LIME explanations."]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probability in probabilities.items():
        precision, recall, _ = precision_recall_curve(labels, probability)
        ap = results[name]["fixed_0_5"]["average_precision"]
        ax.plot(recall, precision, label=f"{name}: AP={ap:.3f}")
    ax.axhline(float(np.mean(labels)), color="gray", linestyle="--", label="Test toxic prevalence")
    ax.set(xlabel="Toxic recall", ylabel="Toxic precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "precision_recall.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, points) in zip(axes, results.items()):
        matrix = np.asarray(points["validation_threshold"]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
        ax.set(title=name, xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1],
               xticklabels=["not toxic", "toxic"], yticklabels=["not toxic", "toxic"])
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    histories = {name: json.loads((output / f"{name}_training.json").read_text()) for name in ("frozen", "untuned")}
    histories.update({f"trial {t['trial']}": t for t in json.loads((output / "search_trials.json").read_text())})
    for name, result in histories.items():
        ax.plot([h["epoch"] for h in result["history"]], [h["selection_average_precision"] for h in result["history"]],
                marker="o", label=name)
    ax.set(xlabel="Epoch", ylabel="Selection average precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "selection_curves.png", dpi=160)
    plt.close(fig)
