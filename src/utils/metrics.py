import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

LOGGER = logging.getLogger("AdaGAD-HNC")


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def compute_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 100) -> float:
    idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[idx]
    return float(y_true_sorted[:k].sum() / max(y_true.sum(), 1))


def compute_f1(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    idx = np.argmax(f1s)
    return {
        "f1": float(f1s[idx]),
        "threshold": float(thresholds[idx] if idx < len(thresholds) else thresholds[-1]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }


def compute_all_metrics(
    y_true: np.ndarray, y_score: np.ndarray, recall_ks: list[int] | None = None
) -> dict[str, Any]:
    if recall_ks is None:
        recall_ks = [10, 50, 100]
    metrics = {
        "auc": compute_auc(y_true, y_score),
        "auprc": compute_auprc(y_true, y_score),
    }
    for k in recall_ks:
        metrics[f"recall@{k}"] = compute_recall_at_k(y_true, y_score, k)
    metrics.update(compute_f1(y_true, y_score))
    return metrics
