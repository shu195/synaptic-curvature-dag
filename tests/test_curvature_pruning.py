import torch
import torch.nn as nn
from synaptic_dag.models import MultiHopDAGNet
from synaptic_dag.pruning import (
    compute_hutchinson_diagonal_hessian,
    prune_second_order_curvature,
    prune_magnitude
)

def test_hutchinson_diagonal_hessian():
    torch.manual_seed(42)
    model = MultiHopDAGNet(in_features=4, hidden_dim=8, out_features=1, num_hidden_nodes=2)
    x = torch.randn(20, 4)
    y = torch.randn(20, 1)
    criterion = nn.MSELoss()
    
    diag_H = compute_hutchinson_diagonal_hessian(model, criterion, x, y, hutchinson_samples=3)
    assert len(diag_H) == len(model.edge_keys)
    for k in model.edge_keys:
        assert diag_H[k].shape == model.edges[k].weight.shape

def test_second_order_pruning_execution():
    torch.manual_seed(42)
    model = MultiHopDAGNet(in_features=4, hidden_dim=8, out_features=1, num_hidden_nodes=2)
    x = torch.randn(20, 4)
    y = torch.randn(20, 1)
    criterion = nn.MSELoss()
    
    active_before, total = model.total_active_weights()
    pruned = prune_second_order_curvature(model, prune_fraction=0.2, loss_fn=criterion, x_data=x, y_data=y, hutchinson_samples=2)
    active_after, _ = model.total_active_weights()
    
    assert pruned > 0
    assert active_after == active_before - pruned
