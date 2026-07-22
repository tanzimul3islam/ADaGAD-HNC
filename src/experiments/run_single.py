import logging
from pathlib import Path
from typing import Any

import torch

from src.data.dataset import AnomalyDataset
from src.data.loaders import get_runner
from src.data.splits import apply_split, stratified_train_val_test_split
from src.eval.evaluator import Evaluator
from src.losses.total_loss import AdaGADTotalLoss
from src.model.adagad_hnc import AdaGADHNC
from src.train.checkpoint import CheckpointManager
from src.train.engine import Engine
from src.train.hooks import EarlyStoppingHook
from src.train.scheduler import build_scheduler
from src.utils.io import save_json
from src.utils.seed import setup_seed

LOGGER = logging.getLogger("AdaGAD-HNC")


def run_single(config: dict[str, Any], ckpt_dir: str | None = None) -> tuple[float, list]:
    seed = config.get("seed", 42)
    setup_seed(seed)

    dataset_name = config["data"]["dataset"]
    root = config.get("data_dir", "data")
    dataset = AnomalyDataset(root=root, name=dataset_name)
    data = dataset.data

    y = data.y.cpu().numpy()
    val_ratio = config["data"].get("val_ratio", 0.1)
    test_ratio = config["data"].get("test_ratio", 0.1)
    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        y, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    data = apply_split(data, train_idx, val_idx, test_idx)
    dataset.data = data

    loader_cfg = config.get("loader", {"loader": "full", "batch_size": 1})
    loader = get_runner(dataset, loader_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = data.x.size(1)
    model_cfg = config["model"]
    model_cfg["in_dim"] = in_dim
    model = AdaGADHNC(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"].get("lr", 0.001),
        weight_decay=config["train"].get("weight_decay", 5e-5),
    )
    scheduler = build_scheduler(optimizer, config["train"])
    loss_fn = AdaGADTotalLoss(config.get("loss", {}))
    engine = Engine(model, optimizer, scheduler, device, config["train"])
    ckpt_manager = CheckpointManager(ckpt_dir or "outputs/checkpoints")
    early_stop = EarlyStoppingHook(patience=config["train"].get("early_stopping_patience", 50))

    best_auc = -float("inf")
    metrics_history = []

    for epoch in range(1, config["train"].get("max_epochs", 100) + 1):
        train_metrics = engine.train_epoch(loader, epoch, loss_fn)
        if epoch % config["train"].get("log_every", 1) == 0:
            LOGGER.info(f"Epoch {epoch:03d} | train_loss={train_metrics['train_loss']:.4f}")

        if epoch % config["train"].get("val_every", 10) == 0 or epoch == config["train"].get(
            "max_epochs", 100
        ):
            eval_out = engine.evaluate(loader)
            evaluator = Evaluator()
            metrics = evaluator.evaluate(eval_out["labels"], eval_out["scores"])
            metrics.update(train_metrics)
            metrics_history.append({"epoch": epoch, **metrics})
            early_stop.on_epoch_end(epoch, metrics)
            LOGGER.info(
                f"Epoch {epoch:03d} | val_auc={metrics.get('auc', -1):.4f} "
                f"| val_auprc={metrics.get('auprc', -1):.4f}"
            )
            if metrics.get("auc", -float("inf")) > best_auc:
                best_auc = metrics["auc"]
                ckpt_manager.save(model, optimizer, epoch, best_auc, is_best=True)
                early_stop.wait = 0
            if early_stop.on_epoch_end(epoch, metrics).get("should_stop", False):
                LOGGER.info(f"Early stopping at epoch {epoch}")
                break

    ckpt_manager.save(model, optimizer, epoch, best_auc, is_best=False)
    save_json(metrics_history, Path("outputs") / "metrics.json")
    return best_auc, metrics_history
