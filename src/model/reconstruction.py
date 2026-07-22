from typing import Any

import torch
from torch import nn


class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2, out_dim: int | None = None):
        super().__init__()
        out_dim = out_dim or hidden_dim
        layers: list[Any] = []
        for _i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h):
        return self.mlp(h)


class GraphDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.decoder = MLPDecoder(hidden_dim, num_layers, out_dim=hidden_dim)
        self.pred = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())

    def forward(self, h, edge_index):
        h = self.decoder(h)
        rows, cols = edge_index
        edge_emb = torch.cat([h[rows], h[cols]], dim=-1)
        return self.pred(edge_emb).squeeze(-1)
