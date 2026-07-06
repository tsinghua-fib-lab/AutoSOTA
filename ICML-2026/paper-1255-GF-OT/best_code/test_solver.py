import torch
import numpy as np
import ot
from src.loss_funcs import quota_loss
from src.datagen import get_gaussian_mixture
from src.solvers import penalized_ot_solver
import time

print("Testing solver...")
(X, Y), (S_X, S_Y) = get_gaussian_mixture(d=2, n_x=250, n_y=25, scale=0.2, p_x0=0.5, p_y0=0.5, rng=42)
eps = 1.0
F_target = torch.tensor([[0.2, 0.3], [0.28, 0.22]])

C = ot.dist(X.numpy(), Y.numpy(), metric="sqeuclidean")
a = np.ones(250) / 250
b = np.ones(25) / 25

# First, let us check what the vanilla OT cost is
T_true = ot.sinkhorn(a, b, C, eps)
true_cost = np.sum(T_true * C)
print(f"Vanilla OT cost: {true_cost:.6f}")

print("Calling penalized_ot_solver with pen=100...")
start = time.time()
try:
    T_fair, log = penalized_ot_solver(C, a, b, 
        lambda plan: quota_loss(plan, S_X, S_Y, F_target),
        reg_constraints=100.0, eps=1.0, log=True)
    elapsed = time.time() - start
    print(f"Solved in {elapsed:.1f}s, niter={log[niter]}")
    fair_cost = np.sum(T_fair * C)
    print(f"Fair OT cost: {fair_cost:.6f}")
    print(f"Cost diff: {abs(fair_cost - true_cost):.6f}")
    fair_loss = quota_loss(torch.from_numpy(T_fair).float(), S_X, S_Y, F_target)
    print(f"Fairness loss: {fair_loss.item():.6f}")
except Exception as e:
    elapsed = time.time() - start
    print(f"Failed after {elapsed:.1f}s: {e}")
    import traceback
    traceback.print_exc()
