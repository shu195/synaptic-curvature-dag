from typing import Dict, Optional
import torch
import torch.nn as nn
from ..models.dag_net import MultiHopDAGNet

def compute_hutchinson_diagonal_hessian(
    model: MultiHopDAGNet,
    loss_fn: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    hutchinson_samples: int = 5
) -> Dict[str, torch.Tensor]:
    """
    Computes unbiased diagonal Hessian estimates in O(P) memory via
    Pearlmutter Hessian-Vector Products and Rademacher stochastic probes.
    """
    model.eval()
    pred = model(x_data)
    loss = loss_fn(pred, y_data)
    
    weight_params = [model.edges[k].weight for k in model.edge_keys]
    grads = torch.autograd.grad(loss, weight_params, create_graph=True)
    
    diag_H_dict = {k: torch.zeros_like(p.data) for k, p in zip(model.edge_keys, weight_params)}
    
    for _ in range(hutchinson_samples):
        # Sample Rademacher probe v in {-1, +1}
        vs = [(torch.randint_like(p.data, 0, 2) * 2 - 1).float() for p in weight_params]
        # Directional projection g^T v
        g_dot_v = sum((g * v).sum() for g, v in zip(grads, vs))
        # Pearlmutter HVP: H v = grad(g^T v)
        hvps = torch.autograd.grad(g_dot_v, weight_params, retain_graph=True)
        for k, v, hvp in zip(model.edge_keys, vs, hvps):
            diag_H_dict[k] += (v * hvp.detach()) / hutchinson_samples
            
    return diag_H_dict

def prune_second_order_curvature(
    model: MultiHopDAGNet,
    prune_fraction: float,
    loss_fn: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    hutchinson_samples: int = 5,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> int:
    """
    Prunes weights by exact second-order Taylor loss perturbation:
        S_i = | - g_i * w_i + 0.5 * H_ii * w_i^2 |
    where H_ii is estimated via matrix-free Pearlmutter HVPs.
    """
    diag_H = compute_hutchinson_diagonal_hessian(model, loss_fn, x_data, y_data, hutchinson_samples)
    
    # Calculate base gradients
    model.zero_grad()
    pred = model(x_data)
    loss = loss_fn(pred, y_data)
    loss.backward()
    
    all_scores = []
    edge_entries = []
    
    for key in model.edge_keys:
        edge = model.edges[key]
        mask = getattr(model, f"mask_{key}")
        if mask.sum() == 0:
            continue
            
        w = edge.weight.data
        g = edge.weight.grad if edge.weight.grad is not None else torch.zeros_like(w)
        h_ii = torch.clamp(diag_H[key], min=0.0)  # Positive curvature resistance
        
        # Second-order Taylor perturbation
        saliency = (-g * w + 0.5 * h_ii * (w ** 2)).abs() * mask
        
        active_indices = (mask == 1).nonzero(as_tuple=False)
        for idx in active_indices:
            r, c = idx[0].item(), idx[1].item()
            all_scores.append(saliency[r, c].item())
            edge_entries.append((key, r, c))
            
    if not edge_entries:
        return 0
        
    k_to_prune = max(1, int(len(edge_entries) * prune_fraction))
    thresh = torch.topk(torch.tensor(all_scores), k_to_prune, largest=False).values.max().item()
    
    pruned_count = 0
    for i, (key, r, c) in enumerate(edge_entries):
        mask = getattr(model, f"mask_{key}")
        edge = model.edges[key]
        if all_scores[i] <= thresh and mask[r, c] == 1 and pruned_count < k_to_prune:
            mask[r, c] = 0.0
            edge.weight.data[r, c] = 0.0
            pruned_count += 1
            
    # Optimizer buffer synchronization: Zero out momentum for pruned parameters
    if optimizer is not None:
        for key in model.edge_keys:
            edge = model.edges[key]
            mask = getattr(model, f"mask_{key}")
            state = optimizer.state.get(edge.weight, None)
            if state is not None:
                if 'exp_avg' in state:
                    state['exp_avg'] *= mask
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'] *= mask
                    
    return pruned_count
