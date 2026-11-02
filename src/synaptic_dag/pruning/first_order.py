import torch
from ..models.dag_net import MultiHopDAGNet

def prune_first_order_taylor(model: MultiHopDAGNet, prune_fraction: float) -> int:
    """
    First-order Taylor loss sensitivity pruning: S_i = |W_i * (dL / dW_i)|.
    Requires prior backward pass to populate .grad on edge parameters.
    """
    all_scores = []
    edge_entries = []
    
    for key in model.edge_keys:
        edge = model.edges[key]
        mask = getattr(model, f"mask_{key}")
        if mask.sum() == 0:
            continue
        active_indices = (mask == 1).nonzero(as_tuple=False)
        grad = edge.weight.grad.abs() if edge.weight.grad is not None else torch.zeros_like(edge.weight.data)
        saliency = (edge.weight.data * grad).abs()
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
            
    return pruned_count
