from typing import Callable, Tuple
import torch

class GradientParticleSwarm:
    """
    Gradient-Guided Particle Swarm Optimization (GPSO).
    
    Coordinates a multi-agent swarm in parameter space using:
        v_{t+1}^i = w * v_t^i - lr * grad_i + c1 * r1 * (p_best^i - x_t^i) + c2 * r2 * (g_best - x_t^i)
        x_{t+1}^i = x_t^i + v_{t+1}^i
    """
    def __init__(
        self,
        num_agents: int,
        dim: int,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        bounds: Tuple[float, float] = (-5.0, 5.0),
        w_inertia: float = 0.7,
        c_pbest: float = 0.5,
        c_gbest: float = 1.0,
        lr: float = 0.02
    ):
        self.num_agents = num_agents
        self.dim = dim
        self.loss_fn = loss_fn
        self.bounds = bounds
        self.w_inertia = w_inertia
        self.c_pbest = c_pbest
        self.c_gbest = c_gbest
        self.lr = lr
        
        # Initialize positions and velocities
        self.x = torch.empty(num_agents, dim).uniform_(bounds[0], bounds[1])
        self.v = torch.zeros_like(self.x)
        
        # Track personal and global bests
        with torch.no_grad():
            init_losses = self.loss_fn(self.x)
            self.p_best = self.x.clone()
            self.p_best_val = init_losses.clone()
            min_idx = init_losses.argmin().item()
            self.g_best = self.x[min_idx].clone()
            self.g_best_val = init_losses[min_idx].item()

    def step(self) -> Tuple[float, torch.Tensor]:
        """Executes a single step of gradient particle swarm optimization."""
        self.x.requires_grad_(True)
        losses = self.loss_fn(self.x)
        grads = torch.autograd.grad(losses.sum(), self.x)[0]
        
        with torch.no_grad():
            # Update personal bests
            better_mask = losses < self.p_best_val
            self.p_best[better_mask] = self.x[better_mask]
            self.p_best_val[better_mask] = losses[better_mask]
            
            # Update global best
            min_val, min_idx = losses.min(dim=0)
            if min_val.item() < self.g_best_val:
                self.g_best_val = min_val.item()
                self.g_best = self.x[min_idx].clone()
                
            # GPSO velocity and position updates
            r1 = torch.rand_like(self.x)
            r2 = torch.rand_like(self.x)
            self.v = (
                self.w_inertia * self.v
                - self.lr * grads
                + self.c_pbest * r1 * (self.p_best - self.x)
                + self.c_gbest * r2 * (self.g_best.unsqueeze(0) - self.x)
            )
            self.x = self.x + self.v
            self.x = torch.clamp(self.x, self.bounds[0], self.bounds[1])
            
        return self.g_best_val, self.g_best.clone()

    def optimize(self, steps: int = 400) -> Tuple[float, torch.Tensor]:
        """Runs the swarm for a fixed step budget."""
        for _ in range(steps):
            self.step()
        return self.g_best_val, self.g_best.clone()
