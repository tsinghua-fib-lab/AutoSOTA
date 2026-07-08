import cvxpy as cp
import torch
import numpy as np
from cvxpylayers.torch import CvxpyLayer
from src.ffolayer import FFOLayer

n = 3
p = 3
torch.manual_seed(1)
A = []
b = []
for i in range(p):
    A.append(torch.randn(n, n).numpy())
    b.append(torch.randn(1).item())

X = cp.Variable((n, n), symmetric=True)
C = cp.Parameter((n, n))

constraints = [X >> 0]
constraints += [cp.trace(A[i] @ X) == b[i] for i in range(p)]
prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)

torch.manual_seed(42)
C_val = torch.randn(n, n)
C.value = C_val.numpy()
prob.solve()

print("The optimal value is", prob.value)
print("A solution X is")
print(X.value)

layer = CvxpyLayer(prob, parameters=[C], variables=[X])
C_th = C_val.clone().requires_grad_(True)
X_th = layer(C_th)[0]
loss = X_th.sum()
loss.backward()
grad_auto = C_th.grad.clone().numpy()

# Finite-difference gradient as ground truth
# Perturb each C_{ij} by eps, re-solve, and estimate dL/dC_{ij}.
def solve_loss(C_np):
    """Solve the SDP for a given C and return the loss sum(X*)."""
    C.value = C_np
    prob.solve(solver=cp.CLARABEL)
    return float(X.value.sum())

eps = 1e-5
C_np = C_val.numpy()
loss0 = solve_loss(C_np)
grad_fd = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        C_pert = C_np.copy()
        C_pert[i, j] += eps
        grad_fd[i, j] = (solve_loss(C_pert) - loss0) / eps

print("\nCvxpyLayer (autodiff) gradient dL/dC:")
print(np.round(grad_auto, 6))
print("\nFinite-difference gradient dL/dC (ground truth):")
print(np.round(grad_fd, 6))
print("\nMax absolute error:", np.abs(grad_auto - grad_fd).max())

C_th_ffo = C_val.double().requires_grad_(True)
ffo = FFOLayer(prob, parameters=[C], variables=[X], backward_eps=1e-9)
X_th_ffo, = ffo(C_th_ffo)
loss_ffo = X_th_ffo.sum()
loss_ffo.backward()
grad_ffo = C_th_ffo.grad.clone().numpy()

print("\nFFOLayer gradient dL/dC:")
print(np.round(grad_ffo, 6))
print("\nMax absolute error:", np.abs(grad_auto - grad_ffo).max())