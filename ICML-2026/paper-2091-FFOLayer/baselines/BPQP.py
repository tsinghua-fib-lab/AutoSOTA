import numpy as np
import torch
from torch.autograd import Function
import scipy.sparse as sp
import osqp

device = "cuda" if torch.cuda.is_available() else "cpu"

def _np(x): return x.detach().cpu().numpy()
def _sym(P): return 0.5 * (P + P.T)

def osqp_solve(P_csc, q_np, A_csc, l_np, u_np, eps=1e-6):
    prob = osqp.OSQP()
    prob.setup(P_csc, q_np, A_csc, l_np, u_np, verbose=False,
               eps_abs=eps, eps_rel=eps, eps_prim_inf=eps, eps_dual_inf=eps)
    res = prob.solve()
    if res.x is None:
        raise RuntimeError(res.info.status)
    return res.x.astype(np.float64), res.y.astype(np.float64)

def pack_osqp(P, q, G, h, A, b):
    Pn, qn, Gn, hn, An, bn = [_np(x) for x in [P, q, G, h, A, b]]
    Pn = _sym(Pn).astype(np.float64)
    qn = qn.reshape(-1).astype(np.float64)
    Gn = Gn.astype(np.float64); hn = hn.reshape(-1).astype(np.float64)
    An = An.astype(np.float64); bn = bn.reshape(-1).astype(np.float64)
    m, p = Gn.shape[0], An.shape[0]
    if p > 0:
        Aos = sp.csc_matrix(np.vstack([Gn, An]))
        l = np.hstack([-np.inf*np.ones(m), bn])
        u = np.hstack([hn, bn])
    else:
        Aos = sp.csc_matrix(Gn)
        l = -np.inf*np.ones(m)
        u = hn
    return sp.csc_matrix(Pn), qn, Aos, l.astype(np.float64), u.astype(np.float64), m, p, Gn, hn, An

def bpqp_backward(xn, yn, P_csc, Gn, hn, An, gnp, m, p, act_tol, backward_eps):
    lam = yn[:m] if m > 0 else np.zeros((0,), dtype=np.float64)
    if m > 0:
        resid = (Gn @ xn) - hn
        active = np.where((resid > -act_tol) | (lam > act_tol))[0].astype(np.int64)
    else:
        active = np.zeros((0,), dtype=np.int64)
    rows = []
    if active.size > 0: rows.append(Gn[active, :])
    if p > 0: rows.append(An)
    n = xn.size
    if len(rows) == 0:
        Pd = P_csc.toarray()
        try: z = -np.linalg.solve(Pd, gnp)
        except np.linalg.LinAlgError: z = -np.linalg.solve(Pd + 1e-8*np.eye(n), gnp)
        yb = np.zeros((0,), dtype=np.float64)
        return z, yb, active
    Ab = sp.csc_matrix(np.vstack(rows))
    k = Ab.shape[0]
    z, yb = osqp_solve(P_csc, gnp.astype(np.float64), Ab, np.zeros(k), np.zeros(k), eps=backward_eps)
    return z, yb, active

def BPQPLayer(sign=1, act_tol=1e-6, forward_eps=1e-6, backward_eps=1e-10):
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
                P_csc, qn, Aos, l, u, m, p, *_ = pack_osqp(Pi, sign * qi, Gi, hi, Ai, bi)
                x, y = osqp_solve(P_csc, qn, Aos, l, u, eps=forward_eps)
                xs.append(torch.from_numpy(x).to(device=Pi.device, dtype=Pi.dtype))
                ys.append(torch.from_numpy(y).to(device=Pi.device, dtype=Pi.dtype))
                ms.append(m); ps.append(p)
            x = torch.stack(xs, 0) if batched else xs[0]
            y = torch.stack(ys, 0) if batched else ys[0]
            ctx.save_for_backward(P, q, G, h, A, b, x, y)
            ctx.meta = (batched, B, sign, act_tol, forward_eps, backward_eps, ms, ps)

            return x

        @staticmethod
        def backward(ctx, grad_output):
            P, q, G, h, A, b, x, y = ctx.saved_tensors
            batched, B, sign, act_tol, forward_eps, backward_eps, ms, ps = ctx.meta
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
                P_csc, _, _, _, _, m, p, Gn, hn, An = pack_osqp(Pi, sign * qi, Gi, hi, Ai, bi)
                z, yb, active = bpqp_backward(_np(xi), _np(yi), P_csc, Gn, hn, An, _np(gi), m, p, act_tol, backward_eps)
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
