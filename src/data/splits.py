import logging

import numpy as np
import torch
from torch_geometric.data import Data

LOGGER = logging.getLogger("AdaGAD-HNC")


def stratified_train_val_test_split(
    y: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    train_list, val_list, test_list = [], [], []

    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(idx.size * test_ratio))
        n_val = max(1, int(idx.size * val_ratio))
        test_list.append(idx[:n_test])
        val_list.append(idx[n_test : n_test + n_val])
        train_list.append(idx[n_test + n_val :])

    train_idx = np.concatenate(train_list).astype(np.int64)
    val_idx = np.concatenate(val_list).astype(np.int64)
    test_idx = np.concatenate(test_list).astype(np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    LOGGER.info(f"Split sizes: train={train_idx.size}, val={val_idx.size}, test={test_idx.size}")
    return train_idx, val_idx, test_idx


def apply_split(data: Data, train_idx, val_idx, test_idx):
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data
