import torch
import torch.nn.functional as F
from torch import nn


class ReconstructionLoss(nn.Module):
    def __init__(self, feature_weight: float = 1.0, structure_weight: float = 1.0):
        super().__init__()
        self.feature_weight = feature_weight
        self.structure_weight = structure_weight

    def forward(self, x_recon, x, edge_pred, edge_index):
        feat_loss = F.mse_loss(x_recon, x, reduction="mean")
        structure_loss = F.binary_cross_entropy_with_logits(edge_pred, torch.ones_like(edge_pred))
        return self.feature_weight * feat_loss + self.structure_weight * structure_loss
