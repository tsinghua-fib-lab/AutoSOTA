import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.dXPP import dXPPLayer
from src.sparse_helper import initialize_torch_from_npz

# initialize the penalty-based layer and parameters (assumes Gurobi installed)
penalty_layer = dXPPLayer(
    beta=1e-6,
    penalty_coeff=10.0,
    eps_abs=1e-8,
    eps_rel=0.0,
    solve_type="sparse",
)
data_path = os.path.join(PROJECT_ROOT, "examples", "diagnostic", "data", "cross.npz")
P, q, C, d, A, b = initialize_torch_from_npz(data_path)
q = q.reshape(-1)
d = d.reshape(-1)
if b is not None:
    b = b.reshape(-1)
d.retain_grad()

# == solve QP ==
z_star, mu_star, nu_star = penalty_layer(P, q, C, d, A, b)

# == form a scalar loss and differentiate ==
z_star.sum().backward()

print(z_star) # optimal point $$z^*$$
print(d.grad) # gradient (w.r.t. d)