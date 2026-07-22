import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger("AdaGAD-HNC")


def validate_graph_data(data: Any) -> None:
    if not hasattr(data, "x") or data.x is None:
        raise ValueError("Data object missing node features 'x'")
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("Data object missing edge_index")

    if data.x.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {tuple(data.x.shape)}")
    if data.x.shape[0] == 0:
        raise ValueError("Empty node feature matrix")
    if data.edge_index.ndim != 2 or data.edge_index.shape[0] != 2:
        raise ValueError("edge_index must be 2 x E")

    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        raise ValueError("NaN or Inf detected in node features")
    if torch.isnan(data.edge_index).any() or torch.isinf(data.edge_index).any():
        raise ValueError("NaN or Inf detected in edge_index")

    if not hasattr(data, "y") or data.y is None:
        raise ValueError("Data object missing anomaly labels 'y'")
    if data.y.numel() != data.x.shape[0]:
        raise ValueError("Label length does not match number of nodes")


def validate_positive_ratio(y: np.ndarray, min_ratio: float = 0.01) -> None:
    ratio = float(y.mean())
    if ratio < min_ratio:
        raise ValueError(f"Anomaly ratio {ratio:.4f} is too small (< {min_ratio})")
    if ratio > 0.99:
        raise ValueError(f"Anomaly ratio {ratio:.4f} is too large (> 0.99)")


def export_dataset_summary(stats: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    LOGGER.info(f"Exported dataset summary to {path}")
