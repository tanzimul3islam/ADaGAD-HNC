import logging

import torch
from src.losses.total_loss import AdaGADTotalLoss
from src.model.adagad_hnc import AdaGADHNC
from src.train.engine import Engine
from src.train.scheduler import build_scheduler
from torch_geometric.data import Data

LOGGER = logging.getLogger("AdaGAD-HNC")


def test_smoke_train():
    num_nodes = 128
    x = torch.randn(num_nodes, 32)
    edge_index = torch.randint(0, num_nodes, (2, 400), dtype=torch.long)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[: num_nodes // 10] = 1
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=torch.ones(num_nodes, dtype=torch.bool))

    model = AdaGADHNC(
        {
            "in_dim": 32,
            "hidden_dim": 64,
            "encoder": {"type": "gat", "hidden_dim": 64, "num_layers": 2, "heads": 4},
            "readout": {"type": "mean"},
            "discriminator": {"hidden_dim": 64},
            "reconstruction": {"hidden_dim": 64},
            "fusion": {"hidden_dim": 64},
        }
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = AdaGADTotalLoss({})
    scheduler = build_scheduler(optimizer, {"max_epochs": 2, "lr_scheduler": "cosine"})
    device = torch.device("cpu")
    engine = Engine(model, optimizer, scheduler, device, {"amp": False})

    model.train()
    for epoch in range(2):
        loader = [data.to(device)]
        train_metrics = engine.train_epoch(loader, epoch, loss_fn)
        assert "train_loss" in train_metrics
        assert train_metrics["train_loss"] > 0

    engine.evaluate([data.to(device)])
