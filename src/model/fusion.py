from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2, entropy_reg_weight: float = 0.1):
        super().__init__()
        layers: list[Any] = []
        layers.append(nn.Linear(4, hidden_dim))
        layers.append(nn.ReLU())
        for _i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 3))
        self.fusion_mlp = nn.Sequential(*layers)
        self.entropy_reg_weight = entropy_reg_weight

    def forward(
        self, context_score, patch_score, recon_score, uncertainty: torch.Tensor | None = None
    ):
        if uncertainty is None:
            uncertainty = torch.zeros_like(context_score)
        h = torch.stack([context_score, patch_score, recon_score, uncertainty], dim=-1)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        w_logits = self.fusion_mlp(h)
        w = F.softmax(w_logits, dim=-1)
        entropy = -(w * (w.clamp_min(1e-9).log())).sum(dim=-1).mean()
        score = (w * torch.stack([context_score, patch_score, recon_score], dim=-1)).sum(dim=-1)
        return score, w, entropy
