import logging
from abc import ABC, abstractmethod
from typing import Any

LOGGER = logging.getLogger("AdaGAD-HNC")


class Hook(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def on_train_end(self) -> dict[str, Any]:
        pass


class EarlyStoppingHook(Hook):
    def __init__(self, patience: int = 50, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.wait = 0

    def on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
        val_auc = metrics.get("val_auc", -float("inf"))
        if val_auc > self.best + self.min_delta:
            self.best = val_auc
            self.wait = 0
        else:
            self.wait += 1
        metrics["should_stop"] = self.wait >= self.patience
        return metrics

    def on_train_end(self) -> dict[str, Any]:
        return {}


class TensorBoardHook(Hook):
    def __init__(self, writer, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = writer or SummaryWriter(log_dir=log_dir)
        except Exception as e:
            LOGGER.warning(f"TensorBoard not available: {e}")
            self.writer = None

    def on_epoch_end(self, epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
        if self.writer is None:
            return {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, epoch)
        return metrics

    def on_train_end(self) -> dict[str, Any]:
        if self.writer is not None:
            self.writer.close()
        return {}
