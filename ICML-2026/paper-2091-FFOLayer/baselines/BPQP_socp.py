import numpy as np
import torch
from torch.autograd import Function
import scipy.sparse as sp
import osqp
import cvxpy as cp

device = "cuda" if torch.cuda.is_available() else "cpu"

def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def _sym(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)

def osqp_solve(P_csc, q_np, A_csc, l_np, u_np, eps=1e-6):
    prob = osqp.OSQP()
    prob.setup(
        P_csc, q_np, A_csc, l_np, u_np,
        verbose=False,
        eps_abs=eps, eps_rel=eps,
        eps_prim_inf=eps, eps_dual_inf=eps
    )
    res = prob.solve()
    if res.x is None:
        raise RuntimeError(res.info.status)
    return res.x.astype(np.float64), res.y.astype(np.float64)

def _soc_dual_scalar_u(dv):
    # CVXPY SOC dual: (u, v)
    if isinstance(dv, (tuple, list)) and len(dv) == 2:
        u = dv[0]
        return float(np.asarray(u, dtype=np.float64).reshape(-1)[0])
    arr = np.asarray(dv, dtype=np.float64).reshape(-1)
    return float(arr[0])


def cvxpy_solve_qp_lin_soc(P, q, G, h, A, b, soc_a, soc_b, sign=1, eps=1e-12, max_iters=2500):
    """
    Solve:
      minimize 0.5 x^T P x + sign*q^T x
      s.t.     Gx <= h
               Ax = b
               a_i^T x + ||x||_2 <= b_i   for i=1..m_soc
    Returns (x, nu_eq, lam_ineq, lam_soc_u)
    """
    Pn = _sym(_np(P)).astype(np.float64)
    qn = (sign * _np(q).reshape(-1)).astype(np.float64)

    Gn = _np(G).astype(np.float64)
    hn = _np(h).reshape(-1).astype(np.float64)

    An = _np(A).astype(np.float64)
    bn = _np(b).reshape(-1).astype(np.float64)

    soc_an = _np(soc_a).astype(np.float64)
    soc_bn = _np(soc_b).reshape(-1).astype(np.float64)

    n = Pn.shape[0]
    m = Gn.shape[0]
    p = An.shape[0]
    msoc = soc_an.shape[0]

    x = cp.Variable(n)

    P_psd = cp.psd_wrap(Pn + 1e-12 * np.eye(n))

    obj = cp.Minimize(0.5 * cp.quad_form(x, P_psd) + qn @ x)

    cons_eq = None
    cons_ineq = None
    cons_soc = []

    constraints = []

    if p > 0:
        cons_eq = (An @ x == bn)
        constraints.append(cons_eq)

    if m > 0:
        cons_ineq = (Gn @ x <= hn)
        constraints.append(cons_ineq)

    for i in range(msoc):
        t = soc_bn[i] - soc_an[i] @ x
        cons_soc_i = cp.SOC(t, x)
        cons_soc.append(cons_soc_i)
        constraints.append(cons_soc_i)

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, eps=eps, max_iters=max_iters, verbose=False)

    if x.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"CVXPY/SCS failed with status={prob.status}")

    x_np = np.asarray(x.value, dtype=np.float64).reshape(-1)

    # duals
    nu = np.zeros((p,), dtype=np.float64)
    lam = np.zeros((m,), dtype=np.float64)
    lam_soc_u = np.zeros((msoc,), dtype=np.float64)

    if cons_eq is not None and cons_eq.dual_value is not None:
        nu = np.asarray(cons_eq.dual_value, dtype=np.float64).reshape(-1)

    if cons_ineq is not None and cons_ineq.dual_value is not None:
        lam = np.asarray(cons_ineq.dual_value, dtype=np.float64).reshape(-1)

    for i, c in enumerate(cons_soc):
        dv = c.dual_value
        if dv is None:
            raise RuntimeError("Missing SOC dual_value from CVXPY; cannot backprop.")
        lam_soc_u[i] = _soc_dual_scalar_u(dv)

    return x_np, nu, lam, lam_soc_u


def bpqp_backward_qp_lin_eq_soc(
    x, grad, P, G, h, A,
    soc_a, soc_b,
    lam_lin, lam_soc_u,
    act_tol=1e-6,
    reg=1e-8,
    backward_eps=1e-10
):
    """
    Backward BPQP for QP + lin ineq + lin eq + SOC (a_i^T x + ||x|| <= b_i).
    """
    x = x.astype(np.float64).reshape(-1)
    grad = grad.astype(np.float64).reshape(-1)
    P = _sym(P.astype(np.float64))
    G = G.astype(np.float64)
    h = h.astype(np.float64).reshape(-1)
    A = A.astype(np.float64)
    soc_a = soc_a.astype(np.float64)
    soc_b = soc_b.astype(np.float64).reshape(-1)
    lam_lin = lam_lin.astype(np.float64).reshape(-1)
    lam_soc_u = lam_soc_u.astype(np.float64).reshape(-1)

    n = x.size
    m = G.shape[0]
    p = A.shape[0]
    msoc = soc_a.shape[0]

    # active linear inequalities
    if m > 0:
        resid_lin = (G @ x) - h
        active_lin = np.where((resid_lin > -act_tol) | (lam_lin > act_tol))[0].astype(np.int64)
    else:
        active_lin = np.zeros((0,), dtype=np.int64)

    # active SOC constraints
    if msoc > 0:
        nx = float(np.linalg.norm(x))
        nx_safe = max(nx, 1e-12)
        resid_soc = (soc_a @ x) + nx - soc_b
        active_soc = np.where((resid_soc > -act_tol) | (lam_soc_u > act_tol))[0].astype(np.int64)

        t1 = float(np.clip(lam_soc_u[active_soc], 0.0, np.inf).sum()) if active_soc.size > 0 else 0.0

        Hnorm = (1.0 / nx_safe) * np.eye(n) - (1.0 / (nx_safe**3)) * np.outer(x, x)
        Pp = P + t1 * Hnorm
    else:
        active_soc = np.zeros((0,), dtype=np.int64)
        Pp = P

    rows = []
    if active_lin.size > 0:
        rows.append(G[active_lin, :])
    if p > 0:
        rows.append(A)
    if active_soc.size > 0:
        nx = float(np.linalg.norm(x))
        nx_safe = max(nx, 1e-12)
        g_soc = soc_a[active_soc, :] + (x[None, :] / nx_safe)  # (a_i + x/||x||)^T z = 0
        rows.append(g_soc)

    if len(rows) == 0:
        try:
            z = -np.linalg.solve(Pp, grad)
        except np.linalg.LinAlgError:
            z = -np.linalg.solve(Pp + reg*np.eye(n), grad)
        return z, np.zeros((0,), dtype=np.float64), active_lin, active_soc

    Ab = sp.csc_matrix(np.vstack(rows).astype(np.float64))
    k = Ab.shape[0]
    P_csc = sp.csc_matrix(Pp)
    z, yb = osqp_solve(P_csc, grad, Ab, np.zeros(k), np.zeros(k), eps=backward_eps)
    return z, yb, active_lin, active_soc


def BPQPLayer_socp(sign=1, act_tol=1e-6, forward_eps=1e-12, backward_eps=1e-10):
    class _Layer(Function):
        @staticmethod
        def forward(ctx, P, q, G, h, A, b, soc_a, soc_b):
            batched = (P.dim() == 3)
            B = P.shape[0] if batched else 1

            xs, nus, lams, lams_soc = [], [], [], []

            for i in range(B):
                Pi = P[i] if batched else P
                qi = q[i] if q.dim() == 2 else q
                Gi = G[i] if G.dim() == 3 else G
                hi = h[i] if h.dim() == 2 else h
                Ai = A[i] if A.dim() == 3 else A
                bi = b[i] if b.dim() == 2 else b
                sai = soc_a[i] if soc_a.dim() == 3 else soc_a
                sbi = soc_b[i] if soc_b.dim() == 2 else soc_b

                x_np, nu_np, lam_np, lam_soc_np = cvxpy_solve_qp_lin_soc(
                    Pi, qi, Gi, hi, Ai, bi, sai, sbi,
                    sign=sign, eps=forward_eps, max_iters=2500
                )

                xs.append(torch.from_numpy(x_np).to(device=Pi.device, dtype=Pi.dtype))
                nus.append(torch.from_numpy(nu_np).to(device=Pi.device, dtype=Pi.dtype))
                lams.append(torch.from_numpy(lam_np).to(device=Pi.device, dtype=Pi.dtype))
                lams_soc.append(torch.from_numpy(lam_soc_np).to(device=Pi.device, dtype=Pi.dtype))

            x = torch.stack(xs, 0) if batched else xs[0]
            nu = torch.stack(nus, 0) if batched else nus[0]
            lam = torch.stack(lams, 0) if batched else lams[0]
            lam_soc = torch.stack(lams_soc, 0) if batched else lams_soc[0]

            ctx.save_for_backward(P, q, G, h, A, b, soc_a, soc_b, x, nu, lam, lam_soc)
            ctx.meta = (batched, B, sign, act_tol, forward_eps, backward_eps)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            P, q, G, h, A, b, soc_a, soc_b, x, nu, lam, lam_soc = ctx.saved_tensors
            batched, B, sign, act_tol, forward_eps, backward_eps = ctx.meta

            gP = torch.zeros_like(P)
            gq = torch.zeros_like(q)
            gG = torch.zeros_like(G)
            gh = torch.zeros_like(h)
            gA = torch.zeros_like(A)
            gb = torch.zeros_like(b)
            gsa = torch.zeros_like(soc_a)
            gsb = torch.zeros_like(soc_b)

            for i in range(B):
                Pi = P[i] if batched else P
                qi = q[i] if q.dim() == 2 else q
                Gi = G[i] if G.dim() == 3 else G
                hi = h[i] if h.dim() == 2 else h
                Ai = A[i] if A.dim() == 3 else A
                bi = b[i] if b.dim() == 2 else b
                sai = soc_a[i] if soc_a.dim() == 3 else soc_a
                sbi = soc_b[i] if soc_b.dim() == 2 else soc_b

                xi = x[i] if batched else x
                nui = nu[i] if batched else nu
                lami = lam[i] if batched else lam
                lam_soci = lam_soc[i] if batched else lam_soc

                gi = grad_output[i] if batched else grad_output

                z, yb, active_lin, active_soc = bpqp_backward_qp_lin_eq_soc(
                    x=_np(xi),
                    grad=_np(gi),
                    P=_np(Pi),
                    G=_np(Gi),
                    h=_np(hi),
                    A=_np(Ai),
                    soc_a=_np(sai),
                    soc_b=_np(sbi),
                    lam_lin=_np(lami),
                    lam_soc_u=_np(lam_soci),
                    act_tol=act_tol,
                    reg=backward_eps,
                    backward_eps=backward_eps
                )

                zt = torch.from_numpy(z).to(device=Pi.device, dtype=Pi.dtype)

                # objective grads
                gq_i = sign * zt
                gP_i = 0.5 * (torch.outer(zt, xi) + torch.outer(xi, zt))

                # unpack yb
                k_lin = int(active_lin.size)
                p = int(Ai.shape[0])
                k_soc = int(active_soc.size)

                mu_lin = torch.from_numpy(yb[:k_lin]).to(device=Pi.device, dtype=Pi.dtype) if k_lin > 0 else torch.empty((0,), device=Pi.device, dtype=Pi.dtype)
                eta_eq = torch.from_numpy(yb[k_lin:k_lin+p]).to(device=Pi.device, dtype=Pi.dtype) if p > 0 else torch.empty((0,), device=Pi.device, dtype=Pi.dtype)
                mu_soc = torch.from_numpy(yb[k_lin+p:k_lin+p+k_soc]).to(device=Pi.device, dtype=Pi.dtype) if k_soc > 0 else torch.empty((0,), device=Pi.device, dtype=Pi.dtype)

                # linear ineq grads
                gG_i = torch.zeros_like(Gi)
                gh_i = torch.zeros_like(hi)
                if Gi.shape[0] > 0 and k_lin > 0:
                    at = torch.tensor(active_lin, device=Pi.device, dtype=torch.long)
                    lam_act = lami.index_select(0, at)
                    block = mu_lin[:, None] * xi[None, :] + lam_act[:, None] * zt[None, :]
                    gG_i.index_copy_(0, at, block)
                    gh_i.index_copy_(0, at, -mu_lin)

                # equality grads
                if p > 0:
                    gb_i = -eta_eq
                    gA_i = eta_eq[:, None] * xi[None, :] + nui[:, None] * zt[None, :]
                else:
                    gb_i = torch.zeros_like(bi)
                    gA_i = torch.zeros_like(Ai)

                # SOC parameter grads (a_i, b_i)
                gsa_i = torch.zeros_like(sai)
                gsb_i = torch.zeros_like(sbi)
                if sai.shape[0] > 0 and k_soc > 0:
                    ats = torch.tensor(active_soc, device=Pi.device, dtype=torch.long)
                    lam_soc_act = lam_soci.index_select(0, ats).clamp_min(0.0)

                    gsa_block = lam_soc_act[:, None] * zt[None, :] + (lam_soc_act * mu_soc)[:, None] * xi[None, :]
                    gsa_i.index_copy_(0, ats, gsa_block)
                    gsb_i.index_copy_(0, ats, mu_soc)

                if batched:
                    gP[i] = gP_i
                    if q.dim() == 2: gq[i] = gq_i
                    else: gq = gq + gq_i
                    if G.dim() == 3: gG[i] = gG_i
                    else: gG = gG + gG_i
                    if h.dim() == 2: gh[i] = gh_i
                    else: gh = gh + gh_i
                    if A.dim() == 3: gA[i] = gA_i
                    else: gA = gA + gA_i
                    if b.dim() == 2: gb[i] = gb_i
                    else: gb = gb + gb_i
                    if soc_a.dim() == 3: gsa[i] = gsa_i
                    else: gsa = gsa + gsa_i
                    if soc_b.dim() == 2: gsb[i] = gsb_i
                    else: gsb = gsb + gsb_i
                else:
                    gP = gP + gP_i
                    gq = gq + gq_i
                    gG = gG + gG_i
                    gh = gh + gh_i
                    gA = gA + gA_i
                    gb = gb + gb_i
                    gsa = gsa + gsa_i
                    gsb = gsb + gsb_i

            return gP, gq, gG, gh, gA, gb, gsa, gsb

    def layer(P, q, G, h, A=None, b=None, soc_a=None, soc_b=None):
        n = P.shape[-1]
        dev = P.device
        dt = P.dtype

        if A is None or (isinstance(A, (list, tuple)) and len(A) == 0):
            A = torch.zeros((0, n), device=dev, dtype=dt)
        if b is None or (isinstance(b, (list, tuple)) and len(b) == 0):
            b = torch.zeros((0,), device=dev, dtype=dt)
        if soc_a is None or (isinstance(soc_a, (list, tuple)) and len(soc_a) == 0):
            soc_a = torch.zeros((0, n), device=dev, dtype=dt)
        if soc_b is None or (isinstance(soc_b, (list, tuple)) and len(soc_b) == 0):
            soc_b = torch.zeros((0,), device=dev, dtype=dt)

        return _Layer.apply(P, q, G, h, A, b, soc_a, soc_b)

    return layer
