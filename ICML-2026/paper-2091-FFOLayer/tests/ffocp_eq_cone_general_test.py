import os
import time
import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from src.ffolayer.ffocp_eq import FFOLayer
# from ffocp_eq_cone_general_not_dpp_cvxtorch import FFOLayer
# from ffocp_eq_cone_general_not_dpp import FFOLayer

torch.set_default_dtype(torch.double)


# ============================================================
# 1) Build CVXPY problem with cone constraints
# ============================================================
def build_conic_problem(
    n: int,
    m: int,
    k: int,
    p_eq: int,
    p_ineq: int,
    cone_type: str = "soc",
):
    x = cp.Variable(n)

    q = cp.Parameter(n)
    # strongly convex objective
    obj = cp.Minimize(0.5 * cp.sum_squares(x) + q.T @ x)

    constraints = []
    params = [q]

    cone_type = cone_type.lower()

    if cone_type == "soc":
        # ||A_i x + b_i||_2 <= c_i^T x + d_i
        A = [cp.Parameter((k, n)) for _ in range(m)]
        b = [cp.Parameter(k) for _ in range(m)]
        c = [cp.Parameter(n) for _ in range(m)]
        d = [cp.Parameter() for _ in range(m)]
        for i in range(m):
            constraints.append(cp.SOC(c[i] @ x + d[i], A[i] @ x + b[i]))
        params += A + b + c + d

    elif cone_type == "nonneg":
        # A_i x + b_i >= 0  (elementwise)
        A = [cp.Parameter((k, n)) for _ in range(m)]
        b = [cp.Parameter(k) for _ in range(m)]
        for i in range(m):
            constraints.append(A[i] @ x + b[i] >= 0)
        params += A + b

    elif cone_type == "exp":
        # ExpCone(u, v, t) elementwise
        A = [cp.Parameter((k, n)) for _ in range(m)]
        b = [cp.Parameter(k) for _ in range(m)]
        v = [cp.Parameter(k) for _ in range(m)]
        t = [cp.Parameter(k) for _ in range(m)]
        for i in range(m):
            u = A[i] @ x + b[i]
            constraints.append(cp.ExpCone(u, v[i], t[i]))
        params += A + b + v + t

    elif cone_type == "psd":
        psd_dim = 4
        r = psd_dim
        S0 = cp.Parameter((r, r))
        Sj = [cp.Parameter((r, r)) for _ in range(n)]

        S = S0
        for j in range(n):
            S = S + x[j] * Sj[j]

        constraints.append(S >> 0)
        params += [S0] + Sj

    else:
        raise ValueError(f"Unsupported cone_type={cone_type}. Use 'soc'|'nonneg'|'exp' for now.")

    if p_eq > 0:
        F = cp.Parameter((p_eq, n))
        g = cp.Parameter(p_eq)
        constraints.append(F @ x == g)
        params += [F, g]

    if p_ineq > 0:
        H = cp.Parameter((p_ineq, n))
        h = cp.Parameter(p_ineq)
        constraints.append(H @ x <= h)
        params += [H, h]

    prob = cp.Problem(obj, constraints)
    return prob, x, params


# ============================================================
# 2) Random-feasible parameter generator
# ============================================================
def random_params_for_cone(
    n, m, k, p_eq, p_ineq, cone_type,
    scale=0.1, margin=1.0,
    active_nonneg: int | None = None,   # for nonneg: number of active among m*k
    active_pineq: int | None = None,    # for Hx<=h: number of active among p_ineq
    active_cones: int | None = None,    # for soc/exp: number of active cones among m
    q_noise: float = 0.0,               # tiny noise if you want
):
    cone_type = cone_type.lower()
    x_star = torch.randn(n, dtype=torch.double)

    # make x_star the unconstrained minimizer (so it stays optimal if feasible)
    q_param = (-x_star + q_noise * torch.randn(n, dtype=torch.double)).detach().clone()
    q_param.requires_grad_(True)
    params_torch = [q_param]

    if cone_type == "nonneg":
        total = m * k
        a = total if active_nonneg is None else int(active_nonneg)
        a = max(0, min(a, total))
        active_idx = set(np.random.choice(total, size=a, replace=False).tolist())

        A_list, b_list = [], []
        for i in range(m):
            A = scale * torch.randn(k, n, dtype=torch.double)
            slack = margin + torch.rand(k, dtype=torch.double).abs()
            # set chosen components slack=0 -> tight at x_star
            for r in range(k):
                if (i * k + r) in active_idx:
                    slack[r] = 0.0
            b = -A @ x_star + slack
            A_list.append(A); b_list.append(b)

        params_torch += A_list + b_list

    elif cone_type == "soc":
        a = m if active_cones is None else int(active_cones)
        a = max(0, min(a, m))
        active_cone = set(np.random.choice(m, size=a, replace=False).tolist())

        A_list, b_list, c_list, d_list = [], [], [], []
        for i in range(m):
            A = scale * torch.randn(k, n, dtype=torch.double)
            b = scale * torch.randn(k, dtype=torch.double)
            c = scale * torch.randn(n, dtype=torch.double)

            left = torch.linalg.norm(A @ x_star + b)
            right_base = torch.dot(c, x_star)
            slack = 0.0 if i in active_cone else (margin + torch.rand((), dtype=torch.double).abs()).item()
            d = torch.tensor((left - right_base + slack).item(), dtype=torch.double)

            A_list.append(A); b_list.append(b); c_list.append(c); d_list.append(d)

        params_torch += A_list + b_list + c_list + d_list

    elif cone_type == "exp":
        a = m if active_cones is None else int(active_cones)
        a = max(0, min(a, m))
        active_cone = set(np.random.choice(m, size=a, replace=False).tolist())

        A_list, b_list, v_list, t_list = [], [], [], []
        for i in range(m):
            A = scale * torch.randn(k, n, dtype=torch.double)
            b = scale * torch.randn(k, dtype=torch.double)

            u_star = A @ x_star + b
            v = torch.ones(k, dtype=torch.double)
            slack = (margin + torch.rand(k, dtype=torch.double).abs())
            if i in active_cone:
                slack[:] = 0.0  # make whole cone tight at x_star (elementwise)
            t = torch.exp(u_star) + slack

            A_list.append(A); b_list.append(b); v_list.append(v); t_list.append(t)

        params_torch += A_list + b_list + v_list + t_list

    elif cone_type == "psd":
        psd_dim = 4
        r = psd_dim
        def sym(M): return 0.5 * (M + M.T)
        Sj_list = [sym(scale * torch.randn(r, r, dtype=torch.double)) for _ in range(n)]
        S0 = margin * torch.eye(r, dtype=torch.double)
        for j in range(n):
            S0 = S0 - x_star[j] * Sj_list[j]
        params_torch += [sym(S0)] + Sj_list

    else:
        raise ValueError(f"Unsupported cone_type={cone_type} in generator.")

    # Equalities: F x = g  (feasible at x_star)
    if p_eq > 0:
        F = scale * torch.randn(p_eq, n, dtype=torch.double)
        g = F @ x_star
        params_torch += [F, g]

    # Inequalities: H x <= h (control how many tight at x_star)
    if p_ineq > 0:
        H = scale * torch.randn(p_ineq, n, dtype=torch.double)
        total = p_ineq
        a = total if active_pineq is None else int(active_pineq)
        a = max(0, min(a, total))
        active_rows = set(np.random.choice(total, size=a, replace=False).tolist())

        slack = margin + torch.rand(p_ineq, dtype=torch.double).abs()
        for r in active_rows:
            slack[r] = 0.0
        h = H @ x_star + slack
        params_torch += [H, h]

    return params_torch, q_param



# ============================================================
# 3) Test: FFOLayer vs CvxpyLayer
# ============================================================
def test_blolayer_vs_cvxpy(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = 50
    m = 20
    k = 3
    p_eq = 0
    p_ineq = 10

    cone_type = "nonneg"

    problem, x_cp, params_cp = build_conic_problem(
        n=n, m=m, k=k, p_eq=p_eq, p_ineq=p_ineq, cone_type=cone_type
    )
    assert problem.is_dpp()

    cvx_layer = CvxpyLayer(problem, parameters=params_cp, variables=[x_cp])
    ffolayer = FFOLayer(problem, parameters=params_cp, variables=[x_cp], alpha=100.0, backward_eps=1e-12)

    repeat_times = 2

    tolerance = 1e-12
    solver_args = {"solver": cp.SCS, "max_iters": 2500, "eps": tolerance, "ignore_dpp": False}
    cvxpy_solver_args = {"eps": tolerance}

    blo_fw, blo_bw = [], []
    cvx_fw, cvx_bw = [], []
    cos_sims, l2_diffs = [], []

    for _ in range(repeat_times):
        params_torch, q_param = random_params_for_cone(
            n=n, m=m, k=k, p_eq=p_eq, p_ineq=p_ineq, cone_type=cone_type, scale=0.01, margin=5.0, active_pineq=0, active_nonneg=0, active_cones=10
        )

        optimizer = torch.optim.SGD([q_param], lr=0.1)

        # -------- FFOLayer --------
        t0 = time.time()
        sol_blo, = ffolayer(*params_torch, solver_args=solver_args)
        t1 = time.time()

        loss_blo = sol_blo.sum()
        optimizer.zero_grad(set_to_none=True)
        t2 = time.time()
        loss_blo.backward()
        t3 = time.time()

        grad_blo = q_param.grad.detach().clone()
        blo_fw.append(t1 - t0)
        blo_bw.append(t3 - t2)

        # -------- CvxpyLayer --------
        t0 = time.time()
        sol_cvx, = cvx_layer(*params_torch, solver_args=cvxpy_solver_args)
        t1 = time.time()

        loss_cvx = sol_cvx.sum()
        optimizer.zero_grad(set_to_none=True)
        t2 = time.time()
        loss_cvx.backward()
        t3 = time.time()

        grad_cvx = q_param.grad.detach().clone()
        cvx_fw.append(t1 - t0)
        cvx_bw.append(t3 - t2)

        # -------- Compare gradients --------
        est = grad_blo.reshape(-1)
        gt = grad_cvx.reshape(-1)
        denom = (est.norm() * gt.norm()).clamp_min(1e-12)
        cos_sims.append((est @ gt) / denom)
        l2_diffs.append((est - gt).norm())

    print(f"cone_type = {cone_type}")
    print(f"CvxpyLayer forward times: {cvx_fw}, mean={np.mean(cvx_fw):.4f}")
    print(f"CvxpyLayer backward times: {cvx_bw}, mean={np.mean(cvx_bw):.4f}")
    print(f"FFOLayer  forward times: {blo_fw}, mean={np.mean(blo_fw):.4f}")
    print(f"FFOLayer  backward times: {blo_bw}, mean={np.mean(blo_bw):.4f}")
    print(f"cosine similarity: {cos_sims}")
    print(f"l2 diffs: {l2_diffs}")


if __name__ == "__main__":
    test_blolayer_vs_cvxpy(seed=0)
