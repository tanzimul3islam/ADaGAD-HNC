import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset

LOGGER = logging.getLogger("AdaGAD-HNC")


@dataclass
class AnomalyStats:
    num_nodes: int
    num_edges: int
    num_features: int
    num_anomalies: int
    num_normal: int
    anomaly_ratio: float


class AnomalyDataset(Dataset):
    def __init__(self, root: str, name: str, transform=None, pre_transform=None):
        self.root = Path(root)
        self.name = name.lower()
        super().__init__(self.root, transform, pre_transform)
        self._validate_and_load()

    def _validate_and_load(self) -> None:
        processed = Path(self.processed_paths[0])
        if not processed.exists():
            raise FileNotFoundError(
                f"Processed file {processed} not found. Run prepare_data.py first."
            )
        data = torch.load(processed, weights_only=False)
        if isinstance(data, tuple):
            data, self.stats = data
        else:
            self.stats = self.summary() if hasattr(self, "data") else None

        if not isinstance(data, Data):
            raise TypeError(f"Expected torch_geometric.data.Data, got {type(data)}")

        if data.x is None or data.x.shape[0] == 0:
            raise ValueError("Empty node features.")
        if data.edge_index is None or data.edge_index.numel() == 0:
            raise ValueError("Empty edge index.")

        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            raise ValueError("NaN or Inf in node features.")
        if torch.isnan(data.edge_index).any() or torch.isinf(data.edge_index).any():
            raise ValueError("NaN or Inf in edge_index.")

        self.data = data

    @property
    def raw_file_names(self):
        return [f"{self.name}.pt"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        pass

    def len(self) -> int:
        return self.data.num_nodes

    def get(self, idx: int) -> Data:
        return self.data

    @property
    def num_nodes(self) -> int:
        return self.data.num_nodes

    @property
    def num_edges(self) -> int:
        return self.data.edge_index.size(1) // 2

    @property
    def num_features(self) -> int:
        return self.data.num_features

    @property
    def anomaly_labels(self) -> torch.Tensor:
        return self.data.y

    def summary(self) -> dict:
        y = self.data.y.cpu().numpy()
        num_anom = int(y.sum())
        return {
            "dataset": self.name,
            "num_nodes": int(self.data.num_nodes),
            "num_edges": int(self.data.edge_index.size(1) // 2),
            "num_features": int(self.data.num_features),
            "num_anomalies": num_anom,
            "num_normal": int(len(y) - num_anom),
            "anomaly_ratio": float(num_anom / len(y)),
        }
