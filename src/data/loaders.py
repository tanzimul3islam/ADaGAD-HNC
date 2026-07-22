import logging

from torch_geometric.loader import DataLoader, NeighborLoader

from src.data.dataset import AnomalyDataset

LOGGER = logging.getLogger("AdaGAD-HNC")


class FullGraphLoader:
    def __init__(self, dataset: AnomalyDataset, batch_size: int = 1, num_workers: int = 0):
        self.dataset = dataset
        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    def __iter__(self):
        yield from self.loader

    def __len__(self):
        return len(self.loader)


class SubgraphLoader:
    def __init__(
        self,
        dataset: AnomalyDataset,
        num_neighbors: list | None = None,
        batch_size: int = 512,
        num_workers: int = 0,
        shuffle: bool = True,
    ):
        data = dataset.data
        self.loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors if num_neighbors is not None else [15, 10, 5],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            input_nodes=None,
        )

    def __iter__(self):
        yield from self.loader

    def __len__(self):
        return len(self.loader)


def get_runner(dataset: AnomalyDataset, config: dict):
    return get_loader(dataset, config)


def get_loader(dataset: AnomalyDataset, config: dict):
    loader_type = config.get("loader", "full")
    if loader_type == "full":
        return FullGraphLoader(dataset, batch_size=config.get("batch_size", 1))
    if loader_type == "subgraph":
        return SubgraphLoader(
            dataset,
            num_neighbors=config.get("num_neighbors"),
            batch_size=config.get("batch_size", 512),
        )
    raise ValueError(f"Unknown loader type: {loader_type}")
