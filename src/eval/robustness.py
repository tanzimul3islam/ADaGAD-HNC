import logging

import numpy as np
import torch

LOGGER = logging.getLogger("AdaGAD-HNC")


def perturb_features(x: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    noise = torch.randn_like(x) * noise_std
    return x + noise


def perturb_edges(
    edge_index: torch.Tensor, num_nodes: int, flip_ratio: float = 0.05, seed: int = 42
) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    num_edges = edge_index.size(1)
    num_flip = int(num_edges * flip_ratio)
    flip_idx = rng.choice(num_edges, size=num_flip, replace=False)
    new_edges = edge_index.clone()
    for idx in flip_idx:
        u, v = int(edge_index[0, idx]), int(edge_index[1, idx])
        if rng.rand() < 0.5:
            u = rng.randint(0, num_nodes)
        else:
            v = rng.randint(0, num_nodes)
        new_edges[0, idx] = u
        new_edges[1, idx] = v
    return new_edges


def simulate_missing_features(
    x: torch.Tensor, missing_ratio: float = 0.2, seed: int = 42
) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    mask = torch.from_numpy(rng.rand(*x.shape) > missing_ratio).to(x.device)
    return x * mask


def compute_robustness(model, data, config: dict) -> dict:
    device = next(model.parameters()).device
    data = data.to(device)
    base_out = model(data)
    base_score = base_out["score"].cpu().numpy()
    y = data.y.cpu().numpy()
    results = {}

    x_noisy = perturb_features(data.x, noise_std=config.get("feature_noise_std", 0.1))
    data_noisy = data.clone()
    data_noisy.x = x_noisy
    out_noisy = model(data_noisy)
    score_noisy = out_noisy["score"].cpu().numpy()
    results["feature_noise_auc"] = compute_auc_safe(y, score_noisy)
    results["feature_noise_delta"] = float(np.mean(np.abs(base_score - score_noisy)))

    edge_flipped = perturb_edges(
        data.edge_index, data.num_nodes, flip_ratio=config.get("edge_flip_ratio", 0.05)
    )
    data_edge = data.clone()
    data_edge.edge_index = edge_flipped
    out_edge = model(data_edge)
    score_edge = out_edge["score"].cpu().numpy()
    results["edge_perturb_auc"] = compute_auc_safe(y, score_edge)
    results["edge_perturb_delta"] = float(np.mean(np.abs(base_score - score_edge)))

    x_missing = simulate_missing_features(
        data.x, missing_ratio=config.get("missing_feature_ratio", 0.2)
    )
    data_missing = data.clone()
    data_missing.x = x_missing
    out_missing = model(data_missing)
    score_missing = out_missing["score"].cpu().numpy()
    results["missing_feature_auc"] = compute_auc_safe(y, score_missing)
    results["missing_feature_delta"] = float(np.mean(np.abs(base_score - score_missing)))

    return results


def compute_auc_safe(y_true, y_score):
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.0
