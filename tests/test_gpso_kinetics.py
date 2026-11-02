import torch
from synaptic_dag.swarm import GradientParticleSwarm

def test_gpso_swarm_monotonic_global_best():
    torch.manual_seed(42)
    # Sphere function with minimum at 0
    def sphere(x):
        return torch.sum(x ** 2, dim=-1)
        
    swarm = GradientParticleSwarm(num_agents=5, dim=3, loss_fn=sphere, bounds=(-3.0, 3.0), lr=0.05)
    best_0 = swarm.g_best_val
    for _ in range(15):
        curr_best, _ = swarm.step()
        assert curr_best <= best_0 + 1e-6
        best_0 = curr_best
    assert best_0 < 1.0  # Should converge rapidly toward 0
