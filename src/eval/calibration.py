import logging

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

LOGGER = logging.getLogger("AdaGAD-HNC")


def compute_calibration(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> dict:
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_score)
    return {
        "brier_score": float(brier),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }


def temperature_scaling(
    logits: np.ndarray, labels: np.ndarray, max_iters: int = 50, lr: float = 0.01
):
    try:
        import torch
        import torch.nn.functional as F

        temperature = torch.nn.Parameter(torch.ones(1))
        opt = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iters)
        log = torch.from_numpy(logits).float()
        lab = torch.from_numpy(labels).float()

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(log / temperature.clamp_min(1e-6), lab)
            loss.backward()
            return loss

        opt.step(closure)
        return temperature.item()
    except Exception as e:
        LOGGER.warning(f"Temperature scaling failed: {e}")
        return 1.0
