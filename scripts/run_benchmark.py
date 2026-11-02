import argparse
import random
import torch
import torch.nn as nn
import numpy as np

from synaptic_dag.models import MultiHopDAGNet
from synaptic_dag.pruning import prune_magnitude, prune_first_order_taylor, prune_second_order_curvature
from synaptic_dag.diagnostics import run_diagnostic_battery

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def generate_complex_manifold(n_samples=3000, ood_scale=1.0):
    X = torch.randn(n_samples, 6) * ood_scale
    y = (
        torch.sin(3.0 * X[:, 0]) * torch.cos(2.0 * X[:, 1])
        + 0.5 * (X[:, 2] ** 2)
        - 0.8 * X[:, 3] * X[:, 4]
        + torch.sin(X[:, 5] * 2.0)
    ).unsqueeze(1)
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Synaptic Curvature DAG Benchmark Runner")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    print("=== SYNAPTIC CURVATURE DAG: STANDARDIZED BENCHMARK ===")
    
    X, y = generate_complex_manifold(3000)
    X_train, y_train = X[:2000], y[:2000]
    X_test, y_test = X[2000:], y[2000:]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True
    )
    
    methods = ["dense", "magnitude", "second_order"]
    results = {}
    
    for method in methods:
        set_seed(args.seed)
        model = MultiHopDAGNet(in_features=6, hidden_dim=32, out_features=1, num_hidden_nodes=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        print(f"[*] Training {method.upper()}...")
        for epoch in range(1, args.epochs + 1):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                
            if method != "dense" and epoch % 10 == 0 and epoch <= 70:
                if method == "magnitude":
                    prune_magnitude(model, prune_fraction=0.15)
                elif method == "second_order":
                    prune_second_order_curvature(
                        model,
                        prune_fraction=0.15,
                        loss_fn=criterion,
                        x_data=X_train,
                        y_data=y_train,
                        hutchinson_samples=3,
                        optimizer=optimizer
                    )
                optimizer.zero_grad()
                
        metrics = run_diagnostic_battery(
            model, X_train, y_train, X_test, y_test, generate_complex_manifold
        )
        active, total = model.total_active_weights()
        metrics["sparsity_pct"] = (1.0 - active / total) * 100
        results[method] = metrics
        
    print("\n" + "="*80)
    print(f"{'Metric / Diagnostic Test':<30} | {'Dense Baseline':<15} | {'Magnitude':<15} | {'2nd-Order Curvature'}")
    print("="*80)
    keys = [
        "in_distribution_mse", "ood_extrapolation_mse", "lipschitz_sensitivity",
        "hessian_lambda_max", "activation_stable_rank", "hop5_lesion_damage",
        "inference_latency_us", "sparsity_pct"
    ]
    for k in keys:
        v_d = results["dense"][k]
        v_m = results["magnitude"][k]
        v_c = results["second_order"][k]
        print(f"{k:<30} | {v_d:<15.4f} | {v_m:<15.4f} | {v_c:<15.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
