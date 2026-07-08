import numpy as np
import torch
from torch.autograd import Function
import cvxpy as cp

def _np(x): return x.detach().cpu().numpy()
def _sym(P): return 0.5 * (P + P.T)

CACHE = True
_QP_CACHE = {}
_EQ_CACHE = {}

def _qp_cvx_osqp(Pn, qn, Gn, hn, An, bn, eps_abs=1e-5, eps_rel=1e-5, max_iter=10000):
    Pn = _sym(Pn).astype(np.float64)
    qn = qn.reshape(-1).astype(np.float64)
    Gn = Gn.astype(np.float64); hn = hn.reshape(-1).astype(np.float64)
    An = An.astype(np.float64); bn = bn.reshape(-1).astype(np.float64)

    n = Pn.shape[0]
    m = Gn.shape[0]
    p = An.shape[0]
    key = (n, m, p)

    if (not CACHE) or (key not in _QP_CACHE):
        x = cp.Variable(n)
        Pp = cp.Parameter((n, n), symmetric=True)
        qp = cp.Parameter(n)
        cons = []
        ineq = eq = None
        Gp = hp = Ap = bp = None
        if m > 0:
            Gp = cp.Parameter((m, n))
            hp = cp.Parameter(m)
            ineq = (Gp @ x <= hp)
            cons.append(ineq)
        if p > 0:
            Ap = cp.Parameter((p, n))
            bp = cp.Parameter(p)
            eq = (Ap @ x == bp)
            cons.append(eq)
        obj = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(Pp)) + qp @ x)
        prob = cp.Problem(obj, cons)
        bundle = {"x": x, "Pp": Pp, "qp": qp, "Gp": Gp, "hp": hp, "Ap": Ap, "bp": bp, "ineq": ineq, "eq": eq, "prob": prob}
        if CACHE:
            _QP_CACHE[key] = bundle
    else:
        bundle = _QP_CACHE[key]

    bundle["Pp"].value = Pn
    bundle["qp"].value = qn
    if m > 0:
        bundle["Gp"].value = Gn
        bundle["hp"].value = hn
    if p > 0:
        bundle["Ap"].value = An
        bundle["bp"].value = bn

    bundle["prob"].solve(
        solver=cp.OSQP,
        warm_start=True,
        verbose=False,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=max_iter,
        polish=True,
    )
    st = bundle["prob"].status
    if st not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"CVXPY/OSQP status: {st}")

    xval = np.asarray(bundle["x"].value, dtype=np.float64).reshape(-1)
    lam = np.asarray(bundle["ineq"].dual_value, dtype=np.float64).reshape(-1) if m > 0 else np.zeros((0,), dtype=np.float64)
    nu = np.asarray(bundle["eq"].dual_value, dtype=np.float64).reshape(-1) if p > 0 else np.zeros((0,), dtype=np.float64)
    y = np.concatenate([lam, nu], axis=0)
    return xval, y

def _eq_qp_cvx_osqp(Pn, gnp, Ab, eps_abs=1e-5, eps_rel=1e-5, max_iter=10000):
    Pn = _sym(Pn).astype(np.float64)
    gnp = gnp.reshape(-1).astype(np.float64)
    Ab = Ab.astype(np.float64)

    k, n = Ab.shape
    key = (n, k)

    if (not CACHE) or (key not in _EQ_CACHE):
        z = cp.Variable(n)
        Pp = cp.Parameter((n, n), symmetric=True)
        gp = cp.Parameter(n)
        Ap = cp.Parameter((k, n))
        con = (Ap @ z == 0)
        obj = cp.Minimize(0.5 * cp.quad_form(z, cp.psd_wrap(Pp)) + gp @ z)
        prob = cp.Problem(obj, [con])
        bundle = {"z": z, "Pp": Pp, "gp": gp, "Ap": Ap, "con": con, "prob": prob}
        if CACHE:
            _EQ_CACHE[key] = bundle
    else:
        bundle = _EQ_CACHE[key]

    bundle["Pp"].value = Pn
    bundle["gp"].value = gnp
    bundle["Ap"].value = Ab

    bundle["prob"].solve(
        solver=cp.OSQP,
        warm_start=True,
        verbose=False,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=max_iter,
        polish=True,
    )
    st = bundle["prob"].status
    if st not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"CVXPY/OSQP status: {st}")

    zval = np.asarray(bundle["z"].value, dtype=np.float64).reshape(-1)
    yb = np.asarray(bundle["con"].dual_value, dtype=np.float64).reshape(-1)
    return zval, yb

def bpqp_backward(xn, yn, Pn, Gn, hn, An, gnp, m, p, act_tol):
    lam = yn[:m] if m > 0 else np.zeros((0,), dtype=np.float64)
    if m > 0:
        resid = Gn @ xn - hn
        active = np.where((resid > -act_tol) | (lam > act_tol))[0].astype(np.int64)
    else:
        active = np.zeros((0,), dtype=np.int64)

    rows = []
    if active.size > 0:
        rows.append(Gn[active, :])
    if p > 0:
        rows.append(An)

    n = xn.size
    if len(rows) == 0:
        try:
            z = -np.linalg.solve(Pn, gnp)
        except np.linalg.LinAlgError:
            z = -np.linalg.solve(Pn + 1e-8 * np.eye(n), gnp)
        return z, np.zeros((0,), dtype=np.float64), active

    Ab = np.vstack(rows)
    z, yb = _eq_qp_cvx_osqp(Pn, gnp.astype(np.float64), Ab)
    return z, yb, active

def BPQPLayer(sign=1, act_tol=1e-6):
    class _Layer(Function):
        @staticmethod
        def forward(ctx, P, q, G, h, A, b):
            batched = (P.dim() == 3)
            B = P.shape[0] if batched else 1
            xs, ys, ms, ps = [], [], [], []

            for i in range(B):
                Pi = P[i] if batched else P
                qi = q[i] if q.dim() == 2 else q
                Gi = G[i] if G.dim() == 3 else G
                hi = h[i] if h.dim() == 2 else h
                Ai = A[i] if A.dim() == 3 else A
                bi = b[i] if b.dim() == 2 else b

                Pn = _sym(_np(Pi)).astype(np.float64)
                qn = (sign * _np(qi)).reshape(-1).astype(np.float64)
                Gn = _np(Gi).astype(np.float64)
                hn = _np(hi).reshape(-1).astype(np.float64)
                An = _np(Ai).astype(np.float64)
                bn = _np(bi).reshape(-1).astype(np.float64)

                m, p = Gn.shape[0], An.shape[0]
                x_np, y_np = _qp_cvx_osqp(Pn, qn, Gn, hn, An, bn)

                xs.append(torch.from_numpy(x_np).to(device=Pi.device, dtype=Pi.dtype))
                ys.append(torch.from_numpy(y_np).to(device=Pi.device, dtype=Pi.dtype))
                ms.append(m); ps.append(p)

            x_out = torch.stack(xs, 0) if batched else xs[0]
            y_out = torch.stack(ys, 0) if batched else ys[0]

            ctx.save_for_backward(P, q, G, h, A, b, x_out, y_out)
            ctx.meta = (batched, B, sign, act_tol, ms, ps)
            return x_out

        @staticmethod
        def backward(ctx, grad_output):
            P, q, G, h, A, b, x, y = ctx.saved_tensors
            batched, B, sign, act_tol, ms, ps = ctx.meta

            gP = torch.zeros_like(P); gq = torch.zeros_like(q); gG = torch.zeros_like(G)
            gh = torch.zeros_like(h); gA = torch.zeros_like(A); gb = torch.zeros_like(b)

            for i in range(B):
                Pi = P[i] if batched else P
                qi = q[i] if q.dim() == 2 else q
                hi = h[i] if h.dim() == 2 else h
                Gi = G[i] if G.dim() == 3 else G
                Ai = A[i] if A.dim() == 3 else A
                bi = b[i] if b.dim() == 2 else b

                xi = x[i] if batched else x
                yi = y[i] if batched else y
                gi = grad_output[i] if batched else grad_output

                Pn = _sym(_np(Pi)).astype(np.float64)
                Gn = _np(Gi).astype(np.float64)
                hn = _np(hi).reshape(-1).astype(np.float64)
                An = _np(Ai).astype(np.float64)
                m, p = Gn.shape[0], An.shape[0]

                z, yb, active = bpqp_backward(
                    _np(xi), _np(yi), Pn, Gn, hn, An, _np(gi), m, p, act_tol
                )

                zt = torch.from_numpy(z).to(device=Pi.device, dtype=Pi.dtype)
                gq_i = sign * zt
                gP_i = 0.5 * (torch.outer(zt, xi) + torch.outer(xi, zt))

                lam = yi[:m] if m > 0 else torch.empty((0,), device=Pi.device, dtype=Pi.dtype)
                nu = yi[m:m+p] if p > 0 else torch.empty((0,), device=Pi.device, dtype=Pi.dtype)

                k = int(active.size)
                mu = torch.from_numpy(yb[:k]).to(device=Pi.device, dtype=Pi.dtype) if k > 0 else torch.empty((0,), device=Pi.device, dtype=Pi.dtype)
                eta = torch.from_numpy(yb[k:k+p]).to(device=Pi.device, dtype=Pi.dtype) if p > 0 else torch.empty((0,), device=Pi.device, dtype=Pi.dtype)

                gG_i = torch.zeros_like(Gi); gh_i = torch.zeros_like(hi)
                if m > 0 and k > 0:
                    at = torch.tensor(active, device=Pi.device, dtype=torch.long)
                    lam_act = lam.index_select(0, at)
                    block = mu[:, None] * xi[None, :] + lam_act[:, None] * zt[None, :]
                    gG_i.index_copy_(0, at, block)
                    gh_i.index_copy_(0, at, -mu)

                if p > 0:
                    gb_i = -eta
                    gA_i = eta[:, None] * xi[None, :] + nu[:, None] * zt[None, :]
                else:
                    gb_i = torch.zeros_like(bi)
                    gA_i = torch.zeros_like(Ai)

                if batched: gP[i] = gP_i
                else: gP = gP + gP_i
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

            return gP, gq, gG, gh, gA, gb

    return _Layer.apply
