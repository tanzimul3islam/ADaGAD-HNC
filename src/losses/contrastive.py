import torch
import torch.nn.functional as F
from torch import nn


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / self.temperature
        sim.fill_diagonal_(0.0)
        labels = torch.arange(z_i.size(0), device=z.device)
        labels = torch.cat([labels + z_i.size(0), labels], dim=0)
        loss = self.criterion(sim, labels)
        return loss


def info_nce_loss(pos_score, neg_score):
    logits = torch.cat([pos_score, neg_score], dim=-1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    labels[: pos_score.size(0)] = 1
    return F.cross_entropy(logits, labels)
