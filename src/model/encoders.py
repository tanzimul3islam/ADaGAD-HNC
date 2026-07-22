import logging

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from src.utils.registry import ENCODER_REGISTRY

LOGGER = logging.getLogger("AdaGAD-HNC")


class GCNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.residual = residual

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            h = conv(x, edge_index)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual and x.shape[-1] == h.shape[-1]:
                h = h + x
            x = h
        return x


class GATEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        heads: int = 8,
        residual: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        out_dim = hidden_dim * heads if heads > 1 else hidden_dim
        self.layers.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(out_dim, hidden_dim, heads=heads, dropout=dropout))
        self.layers.append(GATConv(out_dim, hidden_dim, heads=1, dropout=dropout, concat=False))
        self.dropout = dropout
        self.residual = residual
        self._out_dim = hidden_dim

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            h = conv(x, edge_index)
            if i < len(self.layers) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual and x.shape[-1] == h.shape[-1]:
                h = h + x
            x = h
        return x


class SAGEEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.residual = residual

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            h = conv(x, edge_index)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual and x.shape[-1] == h.shape[-1]:
                h = h + x
            x = h
        return x


@ENCODER_REGISTRY.register("gcn")
class RegisteredGCNEncoder(GCNEncoder):
    pass


@ENCODER_REGISTRY.register("gat")
class RegisteredGATEncoder(GATEncoder):
    pass


@ENCODER_REGISTRY.register("sage")
class RegisteredSAGEEncoder(SAGEEncoder):
    pass


def build_encoder(
    type: str,
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float = 0.1,
    heads: int = 8,
    residual: bool = True,
):
    cls = ENCODER_REGISTRY.get(type)
    return cls(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        heads=heads,
        residual=residual,
    )
