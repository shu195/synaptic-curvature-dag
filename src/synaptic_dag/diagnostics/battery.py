from typing import Dict, Any, Tuple
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..models.dag_net import MultiHopDAGNet

def evaluate_ood_extrapolation(
    model: MultiHopDAGNet,
    data_generator,
    ood_scale: float = 2.0,
    n_samples: int = 1000
) -> float:
    """Evaluates Mean Squared Error on out-of-distribution dilated support."""
    model.eval()
    with torch.no_grad():
        x_ood, y_ood = data_generator(n_samples, ood_scale=ood_scale)
        pred = model(x_ood)
        return float(F.mse_loss(pred, y_ood).item())

def evaluate_lipschitz_sensitivity(
    model: MultiHopDAGNet,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    noise_std: float = 0.3
) -> Tuple[float, float]:
    """Measures input perturbation sensitivity L_emp = ||Delta y|| / ||Delta x||."""
    model.eval()
    with torch.no_grad():
        noise = torch.randn_like(x_test) * noise_std
        clean_pred = model(x_test)
        noisy_pred = model(x_test + noise)
        delta_ratio = (torch.norm(noisy_pred - clean_pred) / (torch.norm(noise) + 1e-8)).item()
        noisy_mse = float(F.mse_loss(noisy_pred, y_test).item())
        return delta_ratio, noisy_mse

def evaluate_hessian_spectral_radius(
    model: MultiHopDAGNet,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    max_iter: int = 20,
    tol: float = 1e-4
) -> float:
    """Computes lambda_max via power iteration with dynamic stopping criteria."""
    model.eval()
    criterion = nn.MSELoss()
    pred = model(x_train)
    loss = criterion(pred, y_train)
    
    params = [model.edges[k].weight for k in model.edge_keys if getattr(model, f"mask_{k}").sum() > 0]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    
    # Initialize random Rademacher probe
    vs = [(torch.randint_like(p.data, 0, 2) * 2 - 1).float() for p in params]
    norm = torch.sqrt(sum((v ** 2).sum() for v in vs))
    vs = [v / norm for v in vs]
    
    prev_lambda = 0.0
    for _ in range(max_iter):
        g_dot_v = sum((g * v).sum() for g, v in zip(grads, vs))
        hvps = torch.autograd.grad(g_dot_v, params, retain_graph=True)
        norm = torch.sqrt(sum((h.detach() ** 2).sum() for h in hvps))
        if norm.item() == 0:
            break
        vs = [h.detach() / norm for h in hvps]
        curr_lambda = norm.item()
        if abs(curr_lambda - prev_lambda) / (curr_lambda + 1e-8) < tol:
            break
        prev_lambda = curr_lambda
        
    g_dot_v = sum((g * v).sum() for g, v in zip(grads, vs))
    hvps = torch.autograd.grad(g_dot_v, params, retain_graph=False)
    lambda_max = sum((v * h.detach()).sum() for v, h in zip(vs, hvps)).item()
    return abs(float(lambda_max))

def evaluate_activation_stable_rank(model: MultiHopDAGNet, x_test: torch.Tensor) -> float:
    """
    Computes mean-centered activation stable rank:
        srank(A_tilde) = ||A_tilde||_F^2 / ||A_tilde||_2^2
    where A_tilde = A - (1/B) * 1 * 1^T * A.
    """
    model.eval()
    with torch.no_grad():
        _, hiddens = model(x_test, collect_activations=True)
        sranks = []
        for h in hiddens:
            # Mean-centering to eliminate artificial rank collapse
            h_centered = h - h.mean(dim=0, keepdim=True)
            frob_sq = torch.norm(h_centered, p='fro') ** 2
            s = torch.linalg.svdvals(h_centered)
            top_s_sq = s[0] ** 2 if len(s) > 0 and s[0] > 0 else 1e-8
            sranks.append((frob_sq / top_s_sq).item())
        return float(np.mean(sranks))

def evaluate_causal_lesion(
    model: MultiHopDAGNet,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    edge_key: str = "e_0_5"
) -> float:
    """
    Evaluates causal damage of severing an edge while locking in-degree
    normalization scalar to avoid activation surge confounders.
    """
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        baseline_mse = float(criterion(model(x_test), y_test).item())
        
        # Determine pre-lesion normalization constant c_j
        j_target = int(edge_key.split("_")[2])
        active_in = sum(1 for i in range(j_target) if getattr(model, f"mask_e_{i}_{j_target}").sum() > 0)
        c_j_locked = 1.0 / (active_in ** 0.5) if active_in > 1 else 1.0
        locked_dict = {j_target: c_j_locked}
        
        # Sever the targeted edge
        mask = getattr(model, f"mask_{edge_key}")
        orig_mask = mask.clone()
        mask.zero_()
        
        damaged_pred = model(x_test, locked_normalization=locked_dict)
        damaged_mse = float(criterion(damaged_pred, y_test).item())
        
        # Restore mask
        mask.copy_(orig_mask)
        return damaged_mse - baseline_mse

def evaluate_hardware_latency(
    model: MultiHopDAGNet,
    x_sample: torch.Tensor,
    warmup: int = 50,
    iterations: int = 500
) -> float:
    """Measures inference latency in microseconds per pass."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x_sample)
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = model(x_sample)
        t1 = time.perf_counter()
        return ((t1 - t0) / iterations) * 1e6

def run_diagnostic_battery(
    model: MultiHopDAGNet,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    data_generator
) -> Dict[str, float]:
    """Runs all 6 diagnostic tests and returns a consolidated metrics dictionary."""
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        in_dist_mse = float(criterion(model(x_test), y_test).item())
        
    ood_mse = evaluate_ood_extrapolation(model, data_generator, ood_scale=2.0)
    lipschitz, noisy_mse = evaluate_lipschitz_sensitivity(model, x_test, y_test, noise_std=0.3)
    lambda_max = evaluate_hessian_spectral_radius(model, x_train, y_train, max_iter=15)
    srank = evaluate_activation_stable_rank(model, x_test)
    lesion_dmg = evaluate_causal_lesion(model, x_test, y_test, edge_key="e_0_5")
    latency = evaluate_hardware_latency(model, x_test[:64], warmup=50, iterations=500)
    
    return {
        "in_distribution_mse": in_dist_mse,
        "ood_extrapolation_mse": ood_mse,
        "lipschitz_sensitivity": lipschitz,
        "hessian_lambda_max": lambda_max,
        "activation_stable_rank": srank,
        "hop5_lesion_damage": lesion_dmg,
        "inference_latency_us": latency
    }
