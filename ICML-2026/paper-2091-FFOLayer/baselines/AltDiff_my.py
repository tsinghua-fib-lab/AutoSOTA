import torch
from torch.autograd import Function


@torch.no_grad()
def alt_qp_solve_batched(
    P, q, A, b, G, h,
    rho=1.0, eps=1e-6, max_iter=5000, reg=1e-7
):
    device, dtype = q.device, q.dtype
    B, n = q.shape
    rho_t = torch.as_tensor(rho, device=device, dtype=dtype)

    m = A.shape[1]  # A: (B,m,n)
    p = G.shape[1]  # G: (B,p,n)

    x = torch.zeros((B, n), device=device, dtype=dtype)
    s = torch.zeros((B, p), device=device, dtype=dtype)
    lam = torch.zeros((B, m), device=device, dtype=dtype)
    nu = torch.zeros((B, p), device=device, dtype=dtype)

    I = torch.eye(n, device=device, dtype=dtype).expand(B, n, n)

    if m > 0:
        AtA = A.transpose(-1, -2) @ A
        Atb = (A.transpose(-1, -2) @ b.unsqueeze(-1)).squeeze(-1)
    else:
        AtA = torch.zeros((B, n, n), device=device, dtype=dtype)
        Atb = torch.zeros((B, n), device=device, dtype=dtype)

    if p > 0:
        GtG = G.transpose(-1, -2) @ G
    else:
        GtG = torch.zeros((B, n, n), device=device, dtype=dtype)

    H = P + rho_t * (AtA + GtG) + reg * I
    L = torch.linalg.cholesky(H)

    def solve_H(rhs_2d):  # (B,n)
        return torch.cholesky_solve(rhs_2d.unsqueeze(-1), L).squeeze(-1)

    alive = torch.ones((B,), device=device, dtype=torch.bool)

    for _ in range(max_iter):
        if not alive.any():
            break

        x0 = x
        s0 = s

        rhs = q.clone()

        if m > 0:
            rhs = rhs + (A.transpose(-1, -2) @ lam.unsqueeze(-1)).squeeze(-1) - rho_t * Atb

        if p > 0:
            rhs = rhs + (G.transpose(-1, -2) @ nu.unsqueeze(-1)).squeeze(-1)
            rhs = rhs + rho_t * (G.transpose(-1, -2) @ (s - h).unsqueeze(-1)).squeeze(-1)

        x = solve_H(-rhs)

        if m > 0:
            ax = (A @ x.unsqueeze(-1)).squeeze(-1)  # (B,m)
            lam = lam + rho_t * (ax - b)
        else:
            ax = None

        if p > 0:
            gx = (G @ x.unsqueeze(-1)).squeeze(-1)  # (B,p)
            s = torch.relu(-(nu / rho_t) - (gx - h))
            nu = nu + rho_t * (gx + s - h)
        else:
            gx = None

        dx = torch.linalg.norm(x - x0, dim=-1)
        tol_x = eps * (1.0 + torch.linalg.norm(x0, dim=-1))
        conv_x = dx <= tol_x

        conv_pri = torch.ones((B,), device=device, dtype=torch.bool)

        if m > 0:
            r_eq = torch.linalg.norm(ax - b, dim=-1)
            axn = torch.linalg.norm(ax, dim=-1)
            bn = torch.linalg.norm(b, dim=-1)
            tol_eq = eps * (1.0 + torch.maximum(axn, bn))
            conv_pri = conv_pri & (r_eq <= tol_eq)

        if p > 0:
            r_in = torch.linalg.norm(gx + s - h, dim=-1)
            gsn = torch.linalg.norm(gx + s, dim=-1)
            hn = torch.linalg.norm(h, dim=-1)
            tol_in = eps * (1.0 + torch.maximum(gsn, hn))
            conv_pri = conv_pri & (r_in <= tol_in)

            ds = torch.linalg.norm(s - s0, dim=-1)
            tol_s = eps * (1.0 + torch.linalg.norm(s0, dim=-1))
            conv_s = ds <= tol_s
        else:
            conv_s = torch.ones((B,), device=device, dtype=torch.bool)

        conv = conv_x & conv_pri & conv_s
        alive = alive & (~conv)

    return x, lam, nu, s


def _adjoint_and_param_grads_single(
    P, q, A, b, G, h, x, lam, nu, active_mask, g, reg=1e-9, sym_P=True
):
    device, dtype = g.device, g.dtype
    n = P.shape[0]
    m = A.shape[0]
    p = G.shape[0]

    if p == 0 or active_mask.numel() == 0 or active_mask.sum() == 0:
        Ga = G.new_zeros((0, n))
        nu_a = nu.new_zeros((0,))
        ha = h.new_zeros((0,))
    else:
        Ga = G[active_mask]
        nu_a = nu[active_mask]
        ha = h[active_mask]

    ka = Ga.shape[0]

    K = torch.zeros((n + m + ka, n + m + ka), device=device, dtype=dtype)
    K[:n, :n] = P + reg * torch.eye(n, device=device, dtype=dtype)

    if m > 0:
        K[:n, n:n+m] = A.T
        K[n:n+m, :n] = A

    if ka > 0:
        K[:n, n+m:] = Ga.T
        K[n+m:, :n] = Ga

    rhs = torch.zeros((n + m + ka,), device=device, dtype=dtype)
    rhs[:n] = g

    sol = torch.linalg.solve(K, rhs)
    u = sol[:n]
    v = sol[n:n+m] if m > 0 else sol.new_zeros((0,))
    w = sol[n+m:] if ka > 0 else sol.new_zeros((0,))

    grad_q = -u

    grad_P = -torch.outer(u, x)
    if sym_P:
        grad_P = 0.5 * (grad_P + grad_P.T)

    if m > 0:
        grad_b = v
        grad_A = -(torch.outer(v, x) + torch.outer(lam, u))
    else:
        grad_b = None
        grad_A = None

    if p > 0:
        grad_h = h.new_zeros((p,))
        grad_G = G.new_zeros((p, n))
        if ka > 0:
            grad_h[active_mask] = w
            grad_Ga = -(torch.outer(w, x) + torch.outer(nu_a, u))
            grad_G[active_mask] = grad_Ga
    else:
        grad_h = None
        grad_G = None

    return grad_P, grad_q, grad_G, grad_h, grad_A, grad_b


def AltDiffLayer(
    eps=1e-5,
    max_iter=2500,
    rho=1.0,
    reg_fwd=1e-5,
    active_tol=1e-6,
    reg_bwd=1e-5,
    sym_P=True,
):
    class L(Function):
        @staticmethod
        def forward(ctx, P, q, G=None, h=None, A=None, b=None):
            unbatched = (q.dim() == 1)

            P_in, q_in, A_in, b_in, G_in, h_in = P, q, A, b, G, h

            def ensure_batch(x, want_dim):
                if x is None:
                    return None
                return x.unsqueeze(0) if x.dim() == want_dim else x

            P = ensure_batch(P, 2)
            q = ensure_batch(q, 1)
            B, n = q.shape

            A = ensure_batch(A, 2)
            b = ensure_batch(b, 1)
            G = ensure_batch(G, 2)
            h = ensure_batch(h, 1)

            if A is None or A.numel() == 0:
                A = torch.empty((1, 0, n), device=q.device, dtype=q.dtype)
            if b is None or b.numel() == 0:
                b = torch.empty((1, 0), device=q.device, dtype=q.dtype)
            if G is None or G.numel() == 0:
                G = torch.empty((1, 0, n), device=q.device, dtype=q.dtype)
            if h is None or h.numel() == 0:
                h = torch.empty((1, 0), device=q.device, dtype=q.dtype)

            P_shared = (P.shape[0] == 1 and B > 1)
            A_shared = (A.shape[0] == 1 and B > 1)
            b_shared = (b.shape[0] == 1 and B > 1)
            G_shared = (G.shape[0] == 1 and B > 1)
            h_shared = (h.shape[0] == 1 and B > 1)

            if P_shared: P = P.expand(B, -1, -1)
            if A_shared: A = A.expand(B, -1, -1)
            if b_shared: b = b.expand(B, -1)
            if G_shared: G = G.expand(B, -1, -1)
            if h_shared: h = h.expand(B, -1)

            P_d, q_d, A_d, b_d, G_d, h_d = P.detach(), q.detach(), A.detach(), b.detach(), G.detach(), h.detach()

            x, lam, nu, s = alt_qp_solve_batched(
                P_d, q_d, A_d, b_d, G_d, h_d,
                rho=rho, eps=eps, max_iter=max_iter, reg=reg_fwd
            )

            if s.numel() == 0:
                active = torch.zeros((B, 0), device=q.device, dtype=torch.bool)
            else:
                active = (s <= active_tol)

            ctx.save_for_backward(P_d, q_d, A_d, b_d, G_d, h_d, x, lam, nu, active)
            ctx.unbatched = unbatched
            ctx.shared_flags = (P_shared, A_shared, b_shared, G_shared, h_shared)
            ctx.orig_none = (P_in is None, q_in is None, G_in is None, h_in is None, A_in is None, b_in is None)
            ctx.orig_shapes = (
                getattr(P_in, "shape", None),
                getattr(q_in, "shape", None),
                getattr(G_in, "shape", None),
                getattr(h_in, "shape", None),
                getattr(A_in, "shape", None),
                getattr(b_in, "shape", None),
            )
            ctx.sym_P = sym_P

            return x[0] if unbatched else x

        @staticmethod
        def backward(ctx, grad_output):
            P, q, A, b, G, h, x, lam, nu, active = ctx.saved_tensors
            P_shared, A_shared, b_shared, G_shared, h_shared = ctx.shared_flags
            unbatched = ctx.unbatched
            sym_P = ctx.sym_P

            g = grad_output.unsqueeze(0) if grad_output.dim() == 1 else grad_output
            B, n = g.shape

            grad_P = torch.zeros_like(P)
            grad_q = torch.zeros_like(q)
            grad_A = torch.zeros_like(A) if A.numel() > 0 else None
            grad_b = torch.zeros_like(b) if b.numel() > 0 else None
            grad_G = torch.zeros_like(G) if G.numel() > 0 else None
            grad_h = torch.zeros_like(h) if h.numel() > 0 else None

            for i in range(B):
                act_i = active[i] if active.numel() > 0 else active.new_zeros((0,), dtype=torch.bool)

                Pi = P[i]
                qi = q[i]
                Ai = A[i] if A.numel() > 0 else A.new_zeros((0, n))
                bi = b[i] if b.numel() > 0 else b.new_zeros((0,))
                Gi = G[i] if G.numel() > 0 else G.new_zeros((0, n))
                hi = h[i] if h.numel() > 0 else h.new_zeros((0,))
                xi = x[i]
                lami = lam[i] if lam.numel() > 0 else lam.new_zeros((0,))
                nui = nu[i] if nu.numel() > 0 else nu.new_zeros((0,))

                dP_i, dq_i, dG_i, dh_i, dA_i, db_i = _adjoint_and_param_grads_single(
                    Pi, qi, Ai, bi, Gi, hi, xi, lami, nui, act_i, g[i],
                    reg=reg_bwd, sym_P=sym_P
                )

                grad_P[i] = dP_i
                grad_q[i] = dq_i
                if grad_A is not None and dA_i is not None:
                    grad_A[i] = dA_i
                if grad_b is not None and db_i is not None:
                    grad_b[i] = db_i
                if grad_G is not None and dG_i is not None:
                    grad_G[i] = dG_i
                if grad_h is not None and dh_i is not None:
                    grad_h[i] = dh_i

            if P_shared:
                grad_P = grad_P.sum(dim=0, keepdim=True)
            if A_shared and grad_A is not None:
                grad_A = grad_A.sum(dim=0, keepdim=True)
            if b_shared and grad_b is not None:
                grad_b = grad_b.sum(dim=0, keepdim=True)
            if G_shared and grad_G is not None:
                grad_G = grad_G.sum(dim=0, keepdim=True)
            if h_shared and grad_h is not None:
                grad_h = grad_h.sum(dim=0, keepdim=True)

            if unbatched:
                grad_P = grad_P[0]
                grad_q = grad_q[0]
                if grad_A is not None: grad_A = grad_A[0]
                if grad_b is not None: grad_b = grad_b[0]
                if grad_G is not None: grad_G = grad_G[0]
                if grad_h is not None: grad_h = grad_h[0]

            return grad_P, grad_q, grad_G, grad_h, grad_A, grad_b

    return L.apply
