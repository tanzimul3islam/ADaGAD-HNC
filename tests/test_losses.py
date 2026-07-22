import torch
from src.losses.contrastive import NTXentLoss, info_nce_loss
from src.losses.reconstruction import ReconstructionLoss


def test_ntxent():
    z = torch.randn(8, 16)
    loss = NTXentLoss()(z, z)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_info_nce():
    pos = torch.randn(4, 8)
    neg = torch.randn(4, 8)
    loss = info_nce_loss(pos, neg)
    assert torch.isfinite(loss)


def test_reconstruction_loss():
    x = torch.randn(16, 32)
    recon = torch.randn(16, 32)
    edge_pred = torch.randn(10)
    edge_index = torch.randint(0, 16, (2, 10))
    loss = ReconstructionLoss()(recon, x, edge_pred, edge_index)
    assert loss.item() > 0
