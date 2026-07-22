import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data

from src.model.discriminators import Discriminator
from src.model.encoders import build_encoder
from src.model.fusion import AdaptiveFusion
from src.model.readout import build_readout
from src.model.reconstruction import GraphDecoder, MLPDecoder

LOGGER = logging.getLogger("AdaGAD-HNC")


class AdaGADHNC(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.in_dim = config.get("in_dim", 1433)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_heads = config.get("heads", 8)

        enc_cfg = config.get("encoder", {})
        self.encoder = build_encoder(
            type=enc_cfg.get("type", "gat"),
            in_dim=self.in_dim,
            hidden_dim=enc_cfg.get("hidden_dim", self.hidden_dim),
            num_layers=enc_cfg.get("num_layers", 3),
            dropout=enc_cfg.get("dropout", 0.1),
            heads=enc_cfg.get("heads", self.num_heads),
            residual=enc_cfg.get("residual", True),
        )

        readout_cfg = config.get("readout", {})
        self.readout = build_readout(
            type=readout_cfg.get("type", "mean"),
            hidden_dim=enc_cfg.get("hidden_dim", self.hidden_dim),
        )

        disc_cfg = config.get("discriminator", {})
        self.discriminator = Discriminator(
            hidden_dim=disc_cfg.get("hidden_dim", self.hidden_dim),
            num_layers=disc_cfg.get("num_layers", 2),
        )

        recon_cfg = config.get("reconstruction", {})
        self.decoder = MLPDecoder(
            hidden_dim=recon_cfg.get("hidden_dim", self.hidden_dim),
            num_layers=recon_cfg.get("num_layers", 2),
            out_dim=self.in_dim,
        )
        self.edge_decoder = GraphDecoder(
            hidden_dim=recon_cfg.get("hidden_dim", self.hidden_dim),
            num_layers=recon_cfg.get("num_layers", 2),
        )

        fusion_cfg = config.get("fusion", {})
        self.fusion = AdaptiveFusion(
            hidden_dim=fusion_cfg.get("hidden_dim", self.hidden_dim),
            num_layers=fusion_cfg.get("num_layers", 2),
            entropy_reg_weight=fusion_cfg.get("entropy_reg_weight", 0.1),
        )

        self.loss_fn = config.get("loss", "bce")
        self.temperature = config.get("temperature", 0.2)

    def forward(self, data: Data, return_features: bool = False):
        x = data.x
        edge_index = data.edge_index
        h = self.encoder(x, edge_index)

        batch = getattr(data, "batch", None)

        global_h = self.readout(h, batch)

        recon_x = self.decoder(h)
        recon_loss = F.mse_loss(recon_x, x, reduction="none").mean(dim=-1)

        context_score = torch.sigmoid(self.discriminator(h))
        patch_score = torch.sigmoid(
            self.discriminator(global_h[batch] if batch is not None else global_h.expand_as(h))
        )
        recon_score = recon_loss

        final_score, weights, entropy = self.fusion(context_score, patch_score, recon_score)

        if self.training:
            h_norm = F.normalize(h, p=2, dim=-1)
            sim = torch.mm(h_norm, h_norm.t())
            mask = torch.eye(sim.size(0), device=sim.device).bool()
            sim.masked_fill_(mask, 0.0)
            min_loss = sim.max(dim=-1)[0].mean()
            loss = min_loss
        else:
            loss = torch.tensor(0.0, device=h.device)

        out = {
            "score": final_score,
            "context_score": context_score,
            "patch_score": patch_score,
            "recon_score": recon_score,
            "weights": weights,
            "entropy": entropy,
            "min_loss": loss,
            "h": h,
        }
        return out

    def compute_loss(self, scores_pos, scores_neg, entropy, min_loss, weights):
        bce = F.binary_cross_entropy(
            scores_pos, torch.ones_like(scores_pos)
        ) + F.binary_cross_entropy(scores_neg, torch.zeros_like(scores_neg))
        entropy_reg = self.fusion.entropy_reg_weight * entropy
        return bce + entropy_reg + min_loss


def build_adagad_hnc(config: dict) -> AdaGADHNC:
    return AdaGADHNC(config)
