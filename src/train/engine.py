import logging
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast

LOGGER = logging.getLogger("AdaGAD-HNC")


class Engine:
    def __init__(self, model, optimizer, scheduler, device, config: dict):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.scaler = GradScaler("cpu", enabled=config.get("amp", False))
        self.grad_clip = config.get("gradient_clip_norm", 1.0)

    def train_epoch(self, loader, epoch: int, loss_fn):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress = _get_progress_bar(loader, epoch)
        for batch in progress:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            with autocast(self.device.type, enabled=self.config.get("amp", False)):
                outputs = self.model(batch)
                loss, _ = loss_fn(outputs, batch)

            if not torch.isfinite(loss):
                LOGGER.warning(f"Non-finite loss at epoch {epoch}, skipping batch.")
                continue

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

            if hasattr(progress, "set_postfix"):
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        scores_all = []
        labels_all = []

        progress = _get_progress_bar(loader, prefix="Eval")
        for batch in progress:
            batch = batch.to(self.device)
            out = self.model(batch, return_features=False)
            scores_all.append(out["score"].cpu())
            if hasattr(batch, "y"):
                labels_all.append(batch.y.cpu())

        if scores_all:
            scores = torch.cat(scores_all).numpy()
            labels = torch.cat(labels_all).numpy() if labels_all else None
            return {"scores": scores, "labels": labels}
        return {"scores": None, "labels": None}


def _get_progress_bar(loader, epoch: int = 0, prefix: str = "Train"):
    try:
        from tqdm import tqdm
        return tqdm(loader, desc=f"{prefix} Epoch {epoch}", leave=False)
    except Exception:
        return loader
