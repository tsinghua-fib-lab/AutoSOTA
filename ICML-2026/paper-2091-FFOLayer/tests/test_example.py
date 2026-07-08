import torch
import cvxpy as cp
import numpy as np
from ffolayer import FFOLayer


def test_example():
    # ============================================================
    # Problem:
    #   minimize 0.5 * ||Q_sqrt x||^2 + q^T x
    #   subject to A x == b
    #
    # ============================================================
    n = 5
    m = 2

    # CVXPY parameters
    Q_sqrt_cp = cp.Parameter((n, n))
    q_cp = cp.Parameter(n)
    A_cp = cp.Parameter((m, n))
    b_cp = cp.Parameter(m)

    x = cp.Variable(n)

    objective = cp.Minimize(0.5 * cp.sum_squares(Q_sqrt_cp @ x) + q_cp.T @ x)
    constraints = [A_cp @ x == b_cp]

    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp(), "Problem must be DPP-compliant"

    # ============================================================
    # Torch data
    # ============================================================
    torch.manual_seed(1)
    np.random.seed(1)
    torch.set_default_dtype(torch.float64)

    M = torch.randn(n, n, dtype=torch.float64)
    eps = 0.2
    Q_sqrt_tch = (M + eps * torch.eye(n, dtype=torch.float64)).detach().clone().requires_grad_(True)

    q_tch = torch.randn(n, dtype=torch.float64).detach().clone().requires_grad_(True)
    A_tch = torch.randn(m, n, dtype=torch.float64).detach().clone().requires_grad_(True)

    x0 = torch.randn(n, dtype=torch.float64)
    b_tch = (A_tch.detach() @ x0).detach().clone().requires_grad_(True)

    # ============================================================
    # Solve using CVXPY directly (ground-truth x*)
    # ============================================================
    Q_sqrt_cp.value = Q_sqrt_tch.detach().cpu().numpy()
    q_cp.value = q_tch.detach().cpu().numpy()
    A_cp.value = A_tch.detach().cpu().numpy()
    b_cp.value = b_tch.detach().cpu().numpy()

    problem.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10, verbose=False)

    print(f"CVXPY solve status: {problem.status}")
    print(f"Optimal value: {problem.value:.6f}")
    print(f"Optimal x: {x.value}")

    # ============================================================
    # Compute true gradient wrt x (compare dloss/dx at solution)
    # ============================================================
    x_tch_true = torch.tensor(x.value, dtype=torch.float64, requires_grad=True)

    loss_true = 0.5 * torch.sum((Q_sqrt_tch.detach() @ x_tch_true) ** 2) + (q_tch.detach() @ x_tch_true)
    loss_true.backward()
    grad_true = x_tch_true.grad.detach().clone()

    # ============================================================
    # Solve using FFOLayer
    # ============================================================
    ffo = FFOLayer(problem, parameters=[Q_sqrt_cp, q_cp, A_cp, b_cp], variables=[x])

    x_tch_ffo, = ffo(Q_sqrt_tch, q_tch, A_tch, b_tch)
    x_tch_ffo = x_tch_ffo.reshape(-1)
    x_tch_ffo.retain_grad()

    loss_ffo = 0.5 * torch.sum((Q_sqrt_tch @ x_tch_ffo) ** 2) + (q_tch @ x_tch_ffo)
    loss_ffo.backward()
    grad_ffo = x_tch_ffo.grad.detach().clone()

    # ============================================================
    # Compare
    # ============================================================
    print("\n--- Compare x ---")
    print("x_tch_true:", x_tch_true.detach().cpu().numpy())
    print("x_tch_ffo :", x_tch_ffo.detach().cpu().numpy())
    print("||x_true - x_ffo||_2:", torch.norm(x_tch_true.detach() - x_tch_ffo.detach(), p=2).item())

    print("\n--- Compare d(loss)/d(x) ---")
    cos = torch.nn.functional.cosine_similarity(grad_true, grad_ffo, dim=0).item()
    diff = torch.norm(grad_true - grad_ffo, p=2).item()
    print("cosine similarity:", cos)
    print("L2 gradient difference:", diff)

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_example()
