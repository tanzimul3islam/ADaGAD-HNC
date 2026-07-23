import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from src.data.dataset import AnomalyDataset
from src.data.loaders import get_runner
from src.data.splits import apply_split, stratified_train_val_test_split
from src.eval.evaluator import Evaluator
from src.losses.total_loss import AdaGADTotalLoss
from src.model.adagad_hnc import build_adagad_hnc
from src.train.checkpoint import CheckpointManager
from src.train.engine import Engine
from src.train.hooks import EarlyStoppingHook
from src.train.scheduler import build_scheduler
from src.utils.io import save_json
from src.utils.seed import setup_seed

LOGGER = logging.getLogger("AdaGAD-HNC")

CONFIG_GROUPS = ["data", "model", "train"]


def load_config(path: str) -> dict[str, Any]:
    base = Path(path)
    config: dict[str, Any] = {}
    if base.exists():
        with open(base) as f:
            config = yaml.safe_load(f) or {}

    defaults = config.pop("defaults", [])
    for entry in defaults:
        if not isinstance(entry, dict):
            continue
        for group, name in entry.items():
            if group not in CONFIG_GROUPS:
                continue
            cfg_path = base.parent / group / f"{name}.yaml"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    group_cfg = yaml.safe_load(f) or {}
                config.setdefault(group, {})
                config[group].update(group_cfg)

    return config


def _cast_value(value: str) -> Any:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _merge_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value or key.subkey=value")
        key, value = item.split("=", 1)
        keys = key.split(".")
        cur = config
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        leaf = keys[-1]
        cast = _cast_value(value)
        if len(keys) == 1 and leaf in CONFIG_GROUPS and isinstance(cast, str):
            group_cfg_path = Path("configs") / leaf / f"{cast}.yaml"
            if group_cfg_path.exists():
                with open(group_cfg_path) as f:
                    group_cfg = yaml.safe_load(f) or {}
                cur[leaf] = group_cfg
                continue
        cur[leaf] = cast
    return config


def train(config: dict[str, Any]) -> float:
    seed = config.get("seed", 42)
    setup_seed(seed)

    root = config.get("data_dir", "data")
    dataset_name = config["data"]["dataset"]
    dataset_root = Path(root) / dataset_name
    dataset = AnomalyDataset(root=dataset_root, name=dataset_name)
    data = dataset.data

    y = data.y.cpu().numpy()
    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        y,
        val_ratio=config["data"].get("val_ratio", 0.1),
        test_ratio=config["data"].get("test_ratio", 0.1),
        seed=seed,
    )
    data = apply_split(data, train_idx, val_idx, test_idx)
    dataset.data = data

    loader_cfg = config.get("loader", {"loader": "full", "batch_size": 1})
    loader = get_runner(dataset, loader_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = data.x.size(1)
    model_cfg = config["model"]
    model_cfg["in_dim"] = in_dim
    model = build_adagad_hnc(model_cfg).to(device)

    train_cfg = config["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 0.001)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-5)),
    )
    scheduler = build_scheduler(optimizer, config["train"])
    loss_fn = AdaGADTotalLoss(config.get("loss", {}))
    engine = Engine(model, optimizer, scheduler, device, config["train"])
    ckpt_dir = (
        config.get("checkpoint_dir", "outputs") + f"/{config.get('experiment_name', 'default')}"
    )
    ckpt_manager = CheckpointManager(ckpt_dir)
    early_stop = EarlyStoppingHook(patience=config["train"].get("early_stopping_patience", 50))

    start_epoch = 1
    best_auc = -float("inf")
    resume_from = config.get("resume_from")
    if resume_from:
        try:
            state = ckpt_manager.load(resume_from, map_location=str(device))
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state.get("epoch", 0)) + 1
            best_auc = float(state.get("score", -float("inf")))
            LOGGER.info(f"Resumed from {resume_from} at epoch {start_epoch - 1}")
        except Exception as e:
            LOGGER.warning(f"Failed to resume from {resume_from}: {e}")

    metrics_history = []

    for epoch in range(start_epoch, config["train"].get("max_epochs", 100) + 1):
        train_metrics = engine.train_epoch(loader, epoch, loss_fn)

        if epoch % config["train"].get("val_every", 10) == 0 or epoch == config["train"].get(
            "max_epochs", 100
        ):
            eval_out = engine.evaluate(loader)
            evaluator = Evaluator()
            metrics = evaluator.evaluate(eval_out["labels"], eval_out["scores"])
            metrics.update(train_metrics)
            metrics_history.append({"epoch": epoch, **metrics})
            stop = early_stop.on_epoch_end(epoch, metrics)
            LOGGER.info(
                f"Epoch {epoch:03d} | val_auc={metrics.get('auc', -1):.4f} "
                f"| val_auprc={metrics.get('auprc', -1):.4f}"
            )

            should_stop = stop.get("should_stop", False)
            if metrics.get("auc", -float("inf")) > best_auc:
                best_auc = metrics["auc"]
                ckpt_manager.save(model, optimizer, epoch, best_auc, is_best=True)

            if should_stop:
                LOGGER.info(f"Early stopping at epoch {epoch}")
                break

    ckpt_manager.save(model, optimizer, epoch, best_auc, is_best=False)
    save_json(metrics_history, Path(ckpt_dir) / "metrics.json")
    LOGGER.info(f"Training complete. Best val AUC: {best_auc:.4f}")
    return best_auc


def evaluate(config: dict[str, Any], ckpt_path: str) -> dict[str, Any]:
    from src.train.checkpoint import CheckpointManager

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = config.get("data_dir", "data")
    dataset_name = config["data"]["dataset"]
    dataset_root = Path(root) / dataset_name
    dataset = AnomalyDataset(root=dataset_root, name=dataset_name)
    data = dataset.data
    loader = get_runner(dataset, config.get("loader", {"loader": "full", "batch_size": 1}))

    model_cfg = config["model"]
    model_cfg["in_dim"] = data.x.size(1)
    model = build_adagad_hnc(model_cfg).to(device)
    ckpt_mgr = CheckpointManager("outputs")
    state = ckpt_mgr.load(ckpt_path, map_location=str(device))
    model.load_state_dict(state["model"])
    model.eval()

    engine = Engine(model, None, None, device, config.get("train", {}))
    out = engine.evaluate(loader)
    evaluator = Evaluator()
    metrics = evaluator.evaluate(out["labels"], out["scores"])
    save_json(metrics, Path("outputs") / "eval_metrics.json")
    LOGGER.info(f"Eval metrics: {metrics}")
    return metrics


def main() -> None:
    from src.utils.seed import get_logger
    get_logger("AdaGAD-HNC")
    
    parser = argparse.ArgumentParser(description="AdaGAD-HNC CLI")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train")
    p_train.add_argument("--config", default="configs/default.yaml")
    p_train.add_argument("overrides", nargs="*", help="key=value or key.subkey=value")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--config", default="configs/default.yaml")
    p_eval.add_argument("overrides", nargs="*", help="key=value or key.subkey=value")

    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
        config = _merge_overrides(config, args.overrides)
        train(config)
    elif args.command == "eval":
        config = load_config(args.config)
        config = _merge_overrides(config, args.overrides)
        evaluate(config, args.ckpt)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
