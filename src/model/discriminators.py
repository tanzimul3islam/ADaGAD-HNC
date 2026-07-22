from typing import Any

from torch import nn


class Discriminator(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        layers: list[Any] = []
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            out_dim = 1 if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, h):
        return self.mlp(h).squeeze(-1)
