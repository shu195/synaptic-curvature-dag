from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHopDAGNet(nn.Module):
    """
    Feedforward Directed Acyclic Graph (DAG) Network.
    
    Each node j receives signals from all upstream predecessors i < j:
        z_j = c_j * sum_{i < j} ((W_{i,j} * M_{i,j}) a_i) + b_j
        a_j = sigma(z_j)
        
    where c_j = 1 / sqrt(active_in_degree) enforces dynamical variance preservation.
    """
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, num_hidden_nodes: int = 4):
        super().__init__()
        self.num_nodes = num_hidden_nodes + 2
        self.node_dims = [in_features] + [hidden_dim] * num_hidden_nodes + [out_features]
        
        self.edges = nn.ModuleDict()
        self.edge_keys: List[str] = []
        
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                key = f"e_{i}_{j}"
                self.edge_keys.append(key)
                linear = nn.Linear(self.node_dims[i], self.node_dims[j], bias=False)
                self.edges[key] = linear
                mask = torch.ones(self.node_dims[j], self.node_dims[i], dtype=torch.float32)
                self.register_buffer(f"mask_{key}", mask)
                
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.node_dims[i])) for i in range(1, self.num_nodes)
        ])

    def forward(
        self,
        x: torch.Tensor,
        collect_activations: bool = False,
        locked_normalization: Optional[Dict[int, float]] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional activation caching and frozen normalization scaling.
        
        Args:
            x: Input tensor [batch_size, in_features]
            collect_activations: If True, returns hidden state activations for SVD/rank analysis.
            locked_normalization: Optional dict mapping node index j to a frozen c_j constant
                                  (essential for lesion testing to avoid in-degree surge confounders).
        """
        node_states = [x]
        for j in range(1, self.num_nodes):
            agg = torch.zeros(x.shape[0], self.node_dims[j], device=x.device)
            active_edges = 0
            for i in range(j):
                key = f"e_{i}_{j}"
                mask = getattr(self, f"mask_{key}")
                if mask.sum() > 0:
                    edge = self.edges[key]
                    effective_w = edge.weight * mask
                    agg = agg + F.linear(node_states[i], effective_w)
                    active_edges += 1
                    
            if locked_normalization is not None and j in locked_normalization:
                agg = agg * locked_normalization[j]
            elif active_edges > 1:
                agg = agg / (active_edges ** 0.5)
                
            agg = agg + self.biases[j - 1]
            if j < self.num_nodes - 1:
                node_states.append(F.silu(agg))
            else:
                node_states.append(agg)
                
        if collect_activations:
            return node_states[-1], node_states[1:-1]
        return node_states[-1]

    def total_active_weights(self) -> Tuple[int, int]:
        """Returns (active_parameters, total_parameters)."""
        active = sum(getattr(self, f"mask_{k}").sum().item() for k in self.edge_keys)
        total = sum(getattr(self, f"mask_{k}").numel() for k in self.edge_keys)
        return int(active), int(total)

    def get_hop_distribution(self) -> Dict[int, Tuple[int, int]]:
        """Computes weight retention grouped by hop distance (j - i)."""
        hop_stats: Dict[int, List[int]] = {}
        for key in self.edge_keys:
            i, j = int(key.split("_")[1]), int(key.split("_")[2])
            hop_len = j - i
            mask = getattr(self, f"mask_{key}")
            if hop_len not in hop_stats:
                hop_stats[hop_len] = [0, 0]
            hop_stats[hop_len][0] += int(mask.sum().item())
            hop_stats[hop_len][1] += int(mask.numel())
        return {h: (v[0], v[1]) for h, v in hop_stats.items()}
