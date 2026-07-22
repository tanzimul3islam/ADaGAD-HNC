import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class NormalizeFeatures(BaseTransform):
    def __call__(self, data: Data) -> Data:
        x = data.x.float()
        row_mean = x.mean(dim=1, keepdim=True)
        row_std = x.std(dim=1, keepdim=True)
        row_std[row_std == 0] = 1.0
        data.x = (x - row_mean) / row_std
        return data


class SelectAnomalyLabel(BaseTransform):
    def __init__(self, attr: str = "y"):
        self.attr = attr

    def __call__(self, data: Data) -> Data:
        if not hasattr(data, self.attr):
            raise AttributeError(f"Data object has no attribute {self.attr}")
        data.y = getattr(data, self.attr)
        return data


class AddDegreeFeatures(BaseTransform):
    def __call__(self, data: Data) -> Data:
        from torch_geometric.utils import degree

        deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        deg = deg.view(-1, 1).float()
        data.x = torch.cat([data.x, deg], dim=1)
        return data


def get_transform(name: str | None = None):
    if name is None:
        return None
    name = name.lower()
    if name == "normalize":
        return NormalizeFeatures()
    if name == "add_degree":
        return AddDegreeFeatures()
    raise ValueError(f"Unknown transform: {name}")
