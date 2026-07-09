import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.dXPP import dXPPLayer

n, m = 2, 3
A_data = torch.tensor([[1.0, 2.0],
                       [-1.0, 1.0],
                       [0.5, -0.5]], dtype=torch.float64, requires_grad=True)
b_data = torch.tensor([1.0, -0.5, 0.2], dtype=torch.float64, requires_grad=True)

Q = A_data.T @ A_data + 1e-6 * torch.eye(n, dtype=torch.float64)
q = -(A_data.T @ b_data)
G = -torch.eye(n, dtype=torch.float64)
h = torch.zeros(n, dtype=torch.float64)

layer = dXPPLayer(beta=1e-6, penalty_coeff=10.0, eps_abs=1e-8, solve_type="dense")

x_star, mu_star, nu_star = layer(Q, q, G, h)
x_star.sum().backward()

print("x* =", x_star)
print("d(loss)/d(A_data) =", A_data.grad)
print("d(loss)/d(b_data) =", b_data.grad)
