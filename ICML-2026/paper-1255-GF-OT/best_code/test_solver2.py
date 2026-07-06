import torch
import numpy as np
import ot
from src.loss_funcs import quota_loss
from src.datagen import get_gaussian_mixture
from src.solvers import penalized_ot_solver
import time

print("Testing solver with full pipeline...")
(X, Y), (S_X, S_Y) = get_gaussian_mixture(d=2, n_x=250, n_y=25, scale=0.2, p_x0=0.5, p_y0=0.5, rng=42)
eps = 1.0
F_target = torch.tensor([[0.2, 0.3], [0.28, 0.22]])

C = ot.dist(X.numpy(), Y.numpy(), metric="sqeuclidean")
C = torch.from_numpy(C).float()
a = torch.ones(250) / 250
b = torch.ones(25) / 25

# Run on CPU (consistent with our fix)
T_true = ot.sinkhorn(a.numpy(), b.numpy(), C.numpy(), eps)
true_cost = np.sum(T_true * C.numpy())
print(f"Vanilla OT cost: {true_cost:.6f}")

start = time.time()
T_fair, log = penalized_ot_solver(
    C, a, b,
    lambda plan: quota_loss(plan, S_X, S_Y, F_target),
    reg_constraints=100.0, eps=1.0, log=True
)
elapsed = time.time() - start
print(f"Solved in {elapsed:.1f}s, niter={log['niter']}")

# Convert T_fair to numpy if needed
if isinstance(T_fair, torch.Tensor):
    T_fair_np = T_fair.detach().cpu().numpy()
else:
    T_fair_np = T_fair

fair_cost = np.sum(T_fair_np * C.numpy())
print(f"Fair OT cost: {fair_cost:.6f}")
print(f"Cost diff: {abs(fair_cost - true_cost):.6f}")

fair_loss = quota_loss(
    torch.from_numpy(T_fair_np).float(), S_X, S_Y, F_target
)
print(f"Fairness loss: {fair_loss.item():.6f}")
print("Success!")
