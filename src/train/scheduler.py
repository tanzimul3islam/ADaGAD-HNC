import logging

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, ReduceLROnPlateau

LOGGER = logging.getLogger("AdaGAD-HNC")


def build_scheduler(optimizer: Optimizer, config: dict) -> LRScheduler | None:
    name = config.get("lr_scheduler", "cosine").lower()
    max_epochs = config.get("max_epochs", 100)
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    if name is None or name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name}")
