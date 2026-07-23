import logging
import os
import random

import numpy as np
import torch

LOGGER_NAME = "AdaGAD-HNC"


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name or LOGGER_NAME)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_device(preferred_device: str | None = None) -> torch.device:
    if preferred_device and preferred_device.lower() == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        cap = torch.cuda.get_device_capability()
        major = cap[0]
        if major < 7:
            logging.getLogger(LOGGER_NAME).warning(
                f"GPU has CUDA capability {major}.{cap[1]}, "
                "but current PyTorch build only supports >= 7.0. "
                "Falling back to CPU."
            )
            return torch.device("cpu")
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")
