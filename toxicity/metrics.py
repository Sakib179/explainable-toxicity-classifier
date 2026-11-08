"""Ranking metrics, validation-only operating thresholds, and paired uncertainty."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)


def select_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    # A deterministic tie-break: choose the higher threshold (fewer flags).
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def evaluate(labels, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    return {
        "threshold": float(threshold), "rows": len(labels), "toxic_rows": int(np.sum(labels)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "toxic_precision": float(precision_score(labels, predicted, zero_division=0)),
        "toxic_recall": float(recall_score(labels, predicted, zero_division=0)),
        "toxic_f1": float(f1_score(labels, predicted, zero_division=0)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "per_class": classification_report(labels, predicted, labels=[0, 1],
                                            target_names=["not toxic", "toxic"],
                                            output_dict=True, zero_division=0),
    }


def bootstrap_differences(labels, reference, candidate, reference_threshold, candidate_threshold,
                          repetitions=500, seed=42):
    """Paired stratified percentile intervals, conditional on this fitted pair.

    Resample the same indices for both models; thresholds remain fixed. These
    intervals do not estimate retraining/seed variance or deployment prevalence shift.
    """
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == c) for c in (0, 1)]
    ap, f1 = [], []
    for _ in range(repetitions):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        y, a, b = labels[idx], reference[idx], candidate[idx]
        ap.append(average_precision_score(y, b) - average_precision_score(y, a))
        f1.append(f1_score(y, b >= candidate_threshold, zero_division=0) -
                  f1_score(y, a >= reference_threshold, zero_division=0))
    return {"repetitions": repetitions, "method": "paired stratified percentile bootstrap",
            "difference_direction": "candidate minus frozen baseline",
            "average_precision_difference_95_ci": np.quantile(ap, [0.025, 0.975]).tolist(),
            "toxic_f1_difference_95_ci": np.quantile(f1, [0.025, 0.975]).tolist()}
