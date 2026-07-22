import logging

import torch.nn.functional as F
from torch import nn

from src.losses.contrastive import NTXentLoss
from src.losses.reconstruction import ReconstructionLoss

LOGGER = logging.getLogger("AdaGAD-HNC")


class AdaGADTotalLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.alpha = config.get("alpha", 1.0)
        self.beta = config.get("beta", 1.0)
        self.gamma = config.get("gamma", 1.0)
        self.contrastive_loss = NTXentLoss(temperature=config.get("temperature", 0.2))
        self.recon_loss = ReconstructionLoss()

    def forward(self, outputs, batch, neg_outputs=None):
        context_score = outputs["context_score"]
        patch_score = outputs["patch_score"]
        recon_score = outputs["recon_score"]
        weights = outputs["weights"]
        entropy = outputs["entropy"]
        min_loss = outputs["min_loss"]

        if hasattr(batch, "train_mask"):
            mask = batch.train_mask.to(context_score.device)
            context_score = context_score[mask]
            patch_score = (
                patch_score[mask] if patch_score.numel() == batch.num_nodes else patch_score
            )
            recon_score = recon_score[mask]
            labels = batch.y[mask].to(context_score.device)
        else:
            labels = batch.y.to(context_score.device)

        loss_context = F.binary_cross_entropy(context_score, labels.float())
        loss_patch = F.binary_cross_entropy(patch_score, labels.float())
        loss_recon = recon_score.mean()

        total = (
            self.alpha * loss_context
            + self.beta * loss_patch
            + self.gamma * loss_recon
            + 0.1 * entropy
            + min_loss
        )
        return total, {
            "loss_context": loss_context.item(),
            "loss_patch": loss_patch.item(),
            "loss_recon": loss_recon.item(),
            "loss_entropy": entropy.item(),
            "loss_min": min_loss.item(),
            "weights_mean": weights.mean(0).detach().cpu().numpy().tolist(),
        }
