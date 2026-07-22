import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import to_dense_batch

from src.utils.registry import READOUT_REGISTRY


class MeanReadout(nn.Module):
    def forward(self, x, batch=None):
        if batch is None:
            return x.mean(dim=0, keepdim=True)
        dense, mask = to_dense_batch(x, batch)
        return dense.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)


class MaxReadout(nn.Module):
    def forward(self, x, batch=None):
        if batch is None:
            return x.max(dim=0, keepdim=True)[0]
        dense, mask = to_dense_batch(x, batch)
        dense[~mask] = float("-inf")
        return dense.max(dim=1)[0]


class AttentionReadout(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x, batch=None):
        if batch is None:
            weights = F.softmax(self.attn(x), dim=0)
            return (weights * x).sum(dim=0, keepdim=True)
        dense, mask = to_dense_batch(x, batch)
        attn = self.attn(dense)
        attn = attn.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = F.softmax(attn, dim=1)
        return (weights * dense).sum(dim=1)


@READOUT_REGISTRY.register("mean")
class RegisteredMeanReadout(MeanReadout):
    pass


@READOUT_REGISTRY.register("max")
class RegisteredMaxReadout(MaxReadout):
    pass


@READOUT_REGISTRY.register("attention")
class RegisteredAttentionReadout(AttentionReadout):
    pass


def build_readout(type: str, hidden_dim: int = 256):
    cls = READOUT_REGISTRY.get(type)
    return cls(hidden_dim=hidden_dim) if type == "attention" else cls()
