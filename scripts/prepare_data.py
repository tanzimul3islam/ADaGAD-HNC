import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.data import Data as GeoData
from torch_geometric.datasets import Planetoid

LOGGER = logging.getLogger("AdaGAD-HNC")


def get_dataset_root(data_dir: str = "data") -> str:
    return data_dir


def prepare_cora(data_dir: str, anomaly_ratio: float = 0.05, seed: int = 42):
    dataset = Planetoid(root=data_dir, name="Cora")
    data = dataset[0]
    _inject_anomalies(data, anomaly_ratio, seed)
    _save(data, Path(data_dir) / "cora" / "processed" / "data.pt")
    return _summary(data, "cora")


def prepare_citeseer(data_dir: str, anomaly_ratio: float = 0.05, seed: int = 42):
    dataset = Planetoid(root=data_dir, name="CiteSeer")
    data = dataset[0]
    _inject_anomalies(data, anomaly_ratio, seed)
    _save(data, Path(data_dir) / "citeseer" / "processed" / "data.pt")
    return _summary(data, "citeseer")


def prepare_pubmed(data_dir: str, anomaly_ratio: float = 0.05, seed: int = 42):
    dataset = Planetoid(root=data_dir, name="PubMed")
    data = dataset[0]
    _inject_anomalies(data, anomaly_ratio, seed)
    _save(data, Path(data_dir) / "pubmed" / "processed" / "data.pt")
    return _summary(data, "pubmed")


def prepare_acm(data_dir: str, anomaly_ratio: float = 0.05, seed: int = 42):
    path = Path(data_dir) / "ACM"
    if not path.exists():
        raise FileNotFoundError("ACM dataset requires manual download. See README.")
    _save_placeholder(path / "acm.pt", "acm")
    return {"dataset": "acm", "note": "manual"}


def prepare_blogcatalog(data_dir: str, anomaly_ratio: float = 0.05, seed: int = 42):
    path = Path(data_dir) / "BlogCatalog"
    if not path.exists():
        raise FileNotFoundError("BlogCatalog dataset requires manual download.")
    _save_placeholder(path / "blogcatalog.pt", "blogcatalog")
    return {"dataset": "blogcatalog", "note": "manual"}


def prepare_flickr(data_dir: str, anomaly_ratio: float = 0.05, seed: int = 42):
    path = Path(data_dir) / "Flickr"
    if not path.exists():
        raise FileNotFoundError("Flickr dataset requires manual download.")
    _save_placeholder(path / "flickr.pt", "flickr")
    return {"dataset": "flickr", "note": "manual"}


def _inject_anomalies(data: GeoData, ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    m = data.num_nodes
    num_anom = max(1, int(m * ratio))
    anom_idx = rng.choice(m, size=num_anom, replace=False)
    y = torch.zeros(m, dtype=torch.long)
    y[anom_idx] = 1
    data.y = y


def _save(data: GeoData, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def _save_placeholder(path: Path, name: str):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path.with_suffix(".yaml"), "w") as f:
        yaml.safe_dump({"dataset": name, "status": "pending_manual_download"}, f)


def _summary(data, name):
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
    "cora": prepare_cora,
    "citeseer": prepare_citeseer,
    "pubmed": prepare_pubmed,
    "acm": prepare_acm,
    "blogcatalog": prepare_blogcatalog,
    "flickr": prepare_flickr,
}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--anomaly-ratio", type=float, default=0.05)
    args = parser.parse_args()
    name = args.dataset.lower()
    if name not in PREPARERS:
        raise ValueError(f"Unknown dataset: {name}")
    result = PREPARERS[name](args.data_dir, args.anomaly_ratio)
    print(result)


if __name__ == "__main__":
    main()
