import torch
from torch.autograd import Function

def _safe_norm(x, delta=1e-12):
    return torch.sqrt(torch.dot(x, x) + x.new_tensor(delta))

def _newton_x(P, q, A, b, G, h, s, lam, nu, t, mu, rho, newton_max=30, newton_tol=1e-10, reg=1e-9):
    n = P.shape[0]
    I = torch.eye(n, device=P.device, dtype=P.dtype)

    Hlin = P + rho * (A.T @ A) + rho * (G.T @ G)
    rhs_lin = q + A.T @ lam - rho * (A.T @ b) + G.T @ nu + rho * (G.T @ (s - h))

    x = torch.zeros(n, device=P.device, dtype=P.dtype)

    for _ in range(newton_max):
        r = _safe_norm(x)
        u = x / r
        c = r - 1.0 + t
        alpha = mu + rho * c

        F = Hlin @ x + rhs_lin + alpha * u

        if torch.linalg.norm(F) <= newton_tol * (1.0 + torch.linalg.norm(rhs_lin)):
            break

        M = Hlin + rho * (u[:, None] @ u[None, :]) + alpha * ((1.0 / r) * I - (1.0 / (r**3)) * (x[:, None] @ x[None, :]))
        M = M + reg * I

        dx = torch.linalg.solve(M, -F)

        step = 1.0
        x_new = x + step * dx
        if not torch.isfinite(x_new).all():
            step = 0.5
            x_new = x + step * dx
        x = x_new

    r = _safe_norm(x)
    u = x / r
    c = r - 1.0 + t
    alpha = mu + rho * c
    M = Hlin + rho * (u[:, None] @ u[None, :]) + alpha * ((1.0 / r) * I - (1.0 / (r**3)) * (x[:, None] @ x[None, :]))
    M = M + reg * I
    return x, M, u

def alt_diff_qp_ball(P, q, A, b, G, h, rho=50.0, eps=1e-8, max_iter=2000):
    if rho <= 0:
        raise ValueError("rho must be > 0")

    rho = torch.tensor(float(rho), device=q.device, dtype=q.dtype)
    n = P.shape[0]
    m = b.shape[0]
    p = h.shape[0]

    x = torch.zeros(n, device=q.device, dtype=q.dtype)
    s = torch.zeros(p, device=q.device, dtype=q.dtype)
    t = torch.zeros((), device=q.device, dtype=q.dtype)      # scalar slack
    lam = torch.zeros(m, device=q.device, dtype=q.dtype)
    nu  = torch.zeros(p, device=q.device, dtype=q.dtype)
    mu  = torch.zeros((), device=q.device, dtype=q.dtype)    # scalar multiplier

    Jx   = torch.zeros((n, n), device=q.device, dtype=q.dtype)
    Js   = torch.zeros((p, n), device=q.device, dtype=q.dtype)
    Jt   = torch.zeros((1, n), device=q.device, dtype=q.dtype)
    Jlam = torch.zeros((m, n), device=q.device, dtype=q.dtype)
    Jnu  = torch.zeros((p, n), device=q.device, dtype=q.dtype)
    Jmu  = torch.zeros((1, n), device=q.device, dtype=q.dtype)

    I = torch.eye(n, device=q.device, dtype=q.dtype)

    for _ in range(max_iter):
        x, M, u = _newton_x(P, q, A, b, G, h, s, lam, nu, t, mu, rho)

        term = I
        if m:
            term = term + A.T @ Jlam
        if p:
            term = term + G.T @ Jnu + rho * (G.T @ Js)
        term = term + (u[:, None] @ Jmu) + rho * (u[:, None] @ Jt)
        Jx = torch.linalg.solve(M, -term)

        # s-step
        if p:
            pre = -(nu / rho) - (G @ x - h)
            s_new = torch.relu(pre)
            gate_s = (s_new > 0).to(q.dtype)
            Js = gate_s[:, None] * (-(1.0 / rho) * Jnu - (G @ Jx))
            s = s_new

        # t-step (scalar)
        r = _safe_norm(x)
        g = 1.0 - r - (mu / rho)
        t_new = torch.relu(g)
        gate_t = (t_new > 0).to(q.dtype)  # scalar 0/1

        Jt = gate_t * (-(u[None, :] @ Jx) - (1.0 / rho) * Jmu)
        t = t_new

        # dual updates
        if m:
            lam = lam + rho * (A @ x - b)
            Jlam = Jlam + rho * (A @ Jx)

        if p:
            nu = nu + rho * (G @ x + s - h)
            Jnu = Jnu + rho * ((G @ Jx) + Js)

        mu = mu + rho * (r - 1.0 + t)
        Jmu = Jmu + rho * ((u[None, :] @ Jx) + Jt)

        # stopping (primal residuals)
        r_eq = torch.linalg.norm(A @ x - b) if m else x.new_tensor(0.0)
        r_in = torch.linalg.norm(G @ x + s - h) if p else x.new_tensor(0.0)
        r_ball = torch.abs(r - 1.0 + t)
        if (r_eq + r_in + r_ball) <= eps:
            break

    return x, Jx


def AltDiffLayer(eps=1e-8, max_iter=2000, rho=50.0):
    class L(Function):
        @staticmethod
        def forward(ctx, P, q, G=None, h=None, A=None, b=None):
            def bat(x, d):
                if x is None: return None
                return x.unsqueeze(0) if x.dim() == d else x

            P = bat(P, 2); q = bat(q, 1)
            B, n = q.shape

            A = bat(A, 2) if A is not None else None
            b = bat(b, 1) if b is not None else None
            G = bat(G, 2) if G is not None else None
            h = bat(h, 1) if h is not None else None

            if A is None or A.numel() == 0: A = torch.empty((1, 0, n), device=q.device, dtype=q.dtype)
            if b is None or b.numel() == 0: b = torch.empty((1, 0), device=q.device, dtype=q.dtype)
            if G is None or G.numel() == 0: G = torch.empty((1, 0, n), device=q.device, dtype=q.dtype)
            if h is None or h.numel() == 0: h = torch.empty((1, 0), device=q.device, dtype=q.dtype)

            if A.shape[0] == 1 and B > 1: A = A.expand(B, -1, -1)
            if b.shape[0] == 1 and B > 1: b = b.expand(B, -1)
            if G.shape[0] == 1 and B > 1: G = G.expand(B, -1, -1)
            if h.shape[0] == 1 and B > 1: h = h.expand(B, -1)

            P_det = P.detach(); q_det = q.detach()
            A_det = A.detach(); b_det = b.detach(); G_det = G.detach(); h_det = h.detach()

            xs, Js = [], []
            for i in range(B):
                xi, Ji = alt_diff_qp_ball(P_det[i], q_det[i], A_det[i], b_det[i], G_det[i], h_det[i],
                                          rho=rho, eps=eps, max_iter=max_iter)
                xs.append(xi); Js.append(Ji)

            X = torch.stack(xs, 0)
            J = torch.stack(Js, 0)
            ctx.save_for_backward(J)
            ctx.unbatched = (q.dim() == 1 and P.dim() == 2)
            return X[0] if ctx.unbatched else X

        @staticmethod
        def backward(ctx, grad_output):
            (J,) = ctx.saved_tensors
            g = grad_output.unsqueeze(0) if grad_output.dim() == 1 else grad_output
            grad_q = torch.zeros_like(g)
            for i in range(g.shape[0]):
                grad_q[i] = J[i].T.mv(g[i])
            if grad_output.dim() == 1:
                grad_q = grad_q[0]
            return None, grad_q, None, None, None, None

    return L.apply
