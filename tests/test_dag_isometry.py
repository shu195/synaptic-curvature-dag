import torch
import pytest
from synaptic_dag.models import MultiHopDAGNet

def test_dag_forward_shape():
    model = MultiHopDAGNet(in_features=6, hidden_dim=32, out_features=1, num_hidden_nodes=4)
    x = torch.randn(16, 6)
    out = model(x)
    assert out.shape == (16, 1)

def test_dag_activation_collection():
    model = MultiHopDAGNet(in_features=6, hidden_dim=32, out_features=1, num_hidden_nodes=4)
    x = torch.randn(16, 6)
    out, hiddens = model(x, collect_activations=True)
    assert out.shape == (16, 1)
    assert len(hiddens) == 4
    for h in hiddens:
        assert h.shape == (16, 32)

def test_dag_locked_normalization():
    model = MultiHopDAGNet(in_features=6, hidden_dim=32, out_features=1, num_hidden_nodes=4)
    x = torch.randn(16, 6)
    locked = {5: 0.5}  # Node 5 locked to 0.5
    out1 = model(x, locked_normalization=locked)
    out2 = model(x, locked_normalization=locked)
    assert torch.allclose(out1, out2)
