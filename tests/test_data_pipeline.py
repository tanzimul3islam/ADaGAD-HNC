import numpy as np
import torch
from src.data.splits import stratified_train_val_test_split
from src.data.validation import validate_graph_data, validate_positive_ratio
from torch_geometric.data import Data


def test_stratified_split():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
    tr, va, te = stratified_train_val_test_split(y, val_ratio=0.2, test_ratio=0.2, seed=42)
    assert len(tr) + len(va) + len(te) == 10
    assert len(set(tr) & set(va)) == 0


def test_validate_graph_data():
    x = torch.randn(10, 5)
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)
    y = torch.zeros(10, dtype=torch.long)
    y[3] = 1
    data = Data(x=x, edge_index=edge_index, y=y)
    validate_graph_data(data)
    validate_positive_ratio(y.numpy())


def test_validate_rejects_nan():
    x = torch.randn(10, 5)
    x[0, 0] = float("nan")
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    y = torch.zeros(10, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    try:
        validate_graph_data(data)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
