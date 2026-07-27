import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scipy.io as sio
import torch
from torch_geometric.data import Data as GeoData
from torch_geometric.utils import dense_to_sparse, to_undirected

LOGGER = logging.getLogger("AdaGAD-HNC")

DATASET_DIR = Path("dataset")
RAW_DATASET_DIR = Path("raw_dataset")


def prepare_mat_dataset(
    name: str,
    data_dir: str = "data",
    anomaly_ratio: float = 0.05,
    seed: int = 42,
    mat_key: str = "Label",
) -> Dict:
    mat_path = _find_mat_file(name)
    if mat_path is None:
        raise FileNotFoundError(
            f"MAT file for '{name}' not found. "
            f"Place it in '{DATASET_DIR}/' or '{RAW_DATASET_DIR}/'."
        )

    data = _load_mat(mat_path, name=name, anomaly_key=mat_key)
    data.y = data.y.long()

    out_dir = Path(data_dir) / name.lower() / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "data.pt"
    torch.save(data, save_path)
    summary = _summary(data, name)
    LOGGER.info(f"Saved {name} to {save_path}")
    return summary


def _find_mat_file(name: str) -> Optional[Path]:
    candidates = [
        DATASET_DIR / f"{name}.mat",
        RAW_DATASET_DIR / name / f"{name}.mat",
        RAW_DATASET_DIR / f"{name}.mat",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_mat(path: Path, name: str, anomaly_key: str = "Label") -> GeoData:
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)

    def _get(key):
        if key not in mat:
            raise KeyError(f"'{key}' not found in {path}. Available keys: {list(mat.keys())}")
        return mat[key]

    network = _get("Network")
    attributes = _get("Attributes")
    labels = _get(anomaly_key)

    if hasattr(network, "toarray"):
        adj = network.toarray()
    else:
        adj = np.array(network)

    if hasattr(attributes, "toarray"):
        x = attributes.toarray()
    else:
        x = np.array(attributes)

    adj = np.array(adj, dtype=np.float32)
    x = np.array(x, dtype=np.float32)

    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got {adj.shape}")
    if x.shape[0] != adj.shape[0]:
        raise ValueError(f"Feature matrix rows ({x.shape[0]}) != nodes ({adj.shape[0]})")

    x = torch.tensor(x, dtype=torch.float32)
    y = np.array(labels).flatten()

    if y.size != x.shape[0]:
        raise ValueError(f"Label length ({y.size}) != number of nodes ({x.shape[0]})")

    y = (y > 0).astype(np.int64)

    edge_index, _ = dense_to_sparse(torch.tensor(adj, dtype=torch.float32))
    edge_index = to_undirected(edge_index, num_nodes=x.shape[0])

    data = GeoData(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.long))
    data.num_nodes = x.shape[0]
    data.num_features = x.shape[1]

    if "Class" in mat:
        data.y_class = torch.tensor(np.array(mat["Class"]).flatten(), dtype=torch.long)

    return data


def _summary(data: GeoData, name: str) -> Dict:
    y = data.y.cpu().numpy()
    return {
        "dataset": name,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1) // 2),
        "num_features": int(data.num_features),
        "num_anomalies": int(y.sum()),
        "anomaly_ratio": float(y.mean()),
    }


PREPARERS = {
    "amazon": lambda **kw: prepare_mat_dataset("Amazon", **kw),
    "yelphotel": lambda **kw: prepare_mat_dataset("YelpHotel", **kw),
    "yelpnyc": lambda **kw: prepare_mat_dataset("YelpNYC", **kw),
    "yelpres": lambda **kw: prepare_mat_dataset("YelpRes", **kw),
}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--anomaly-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    name = args.dataset.lower()
    if name not in PREPARERS:
        raise ValueError(f"Unknown dataset: {name}")
    result = PREPARERS[name](
        data_dir=args.data_dir,
        anomaly_ratio=args.anomaly_ratio,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
