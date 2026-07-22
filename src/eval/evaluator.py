import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.metrics import compute_all_metrics

LOGGER = logging.getLogger("AdaGAD-HNC")


class Evaluator:
    def __init__(self, recall_ks: list[int] | None = None):
        if recall_ks is None:
            recall_ks = [10, 50, 100]
        self.recall_ks = recall_ks

    def evaluate(self, y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
        return compute_all_metrics(y_true, y_score, recall_ks=self.recall_ks)

    def evaluate_and_save(self, y_true, y_score, out_path: Path):
        metrics = self.evaluate(y_true, y_score)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        LOGGER.info(f"Saved metrics to {out_path}")
        return metrics


def summarize_multi_seed(results: list[dict], out_path: Path | None = None) -> dict[str, Any]:
    keys = results[0].keys()
    summary = {}
    for k in keys:
        vals = [r[k] for r in results if k in r and isinstance(r[k], (int, float))]
        if vals:
            summary[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    if out_path:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
    return summary
