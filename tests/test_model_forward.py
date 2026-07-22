import torch
from src.model.adagad_hnc import AdaGADHNC
from torch_geometric.data import Data


def test_forward_shape():
    num_nodes = 64
    x = torch.randn(num_nodes, 32)
    edge_index = torch.randint(0, num_nodes, (2, 200), dtype=torch.long)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    y = torch.randint(0, 2, (num_nodes,), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

    model = AdaGADHNC(
        {
            "in_dim": 32,
            "hidden_dim": 64,
            "encoder": {"type": "gat", "hidden_dim": 64, "num_layers": 2, "heads": 4},
            "readout": {"type": "mean"},
            "discriminator": {"hidden_dim": 64},
            "reconstruction": {"hidden_dim": 64},
            "fusion": {"hidden_dim": 64},
        }
    )
    model.eval()
    out = model(data)
    assert out["score"].shape == (num_nodes,)
    assert out["context_score"].shape == (num_nodes,)
    assert "weights" in out
    assert "entropy" in out
