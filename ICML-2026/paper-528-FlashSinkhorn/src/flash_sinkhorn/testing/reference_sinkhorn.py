import torch


def sqeuclid_cost(x, y):
    x = x.float()
    y = y.float()
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).transpose(0, 1)
    return x2 + y2 - 2.0 * x @ y.transpose(0, 1)


def apply_lse_kernel_ref(x, y, f, g, eps, axis, vec=None):
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")
    cost = sqeuclid_cost(x, y)
    f = f.float()
    g = g.float()
    eps = float(eps)
    logits = (f[:, None] + g[None, :] - cost) / eps

    if vec is None:
        if axis == 1:
            lse = torch.logsumexp(logits, dim=1)
            sgn = torch.ones_like(lse)
            remove = f
        else:
            lse = torch.logsumexp(logits, dim=0)
            sgn = torch.ones_like(lse)
            remove = g
    else:
        vec = vec.float()
        vec_abs = vec.abs()
        log_vec = torch.where(
            vec_abs > 0,
            vec_abs.log(),
            torch.full_like(vec_abs, -float("inf")),
        )
        vec_sign = torch.where(vec > 0, 1.0, torch.where(vec < 0, -1.0, 0.0))
        if axis == 1:
            vals = logits + log_vec[None, :]
            maxv = vals.max(dim=1).values
            s = (vec_sign[None, :] * torch.exp(vals - maxv[:, None])).sum(dim=1)
            sgn = torch.where(s > 0, 1.0, torch.where(s < 0, -1.0, 0.0))
            s_abs = s.abs()
            lse = torch.where(
                s_abs > 0, maxv + s_abs.log(), torch.full_like(maxv, -float("inf"))
            )
            remove = f
        else:
            vals = logits + log_vec[:, None]
            maxv = vals.max(dim=0).values
            s = (vec_sign[:, None] * torch.exp(vals - maxv[None, :])).sum(dim=0)
            sgn = torch.where(s > 0, 1.0, torch.where(s < 0, -1.0, 0.0))
            s_abs = s.abs()
            lse = torch.where(
                s_abs > 0, maxv + s_abs.log(), torch.full_like(maxv, -float("inf"))
            )
            remove = g

    safe_remove = torch.where(torch.isfinite(remove), remove, torch.zeros_like(remove))
    out = eps * lse - safe_remove
    return out, sgn


def update_potential_ref(x, y, f, g, log_marginal, eps, axis, vec=None):
    lse, _ = apply_lse_kernel_ref(x, y, f, g, eps, axis, vec=vec)
    safe_lse = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    return eps * log_marginal - safe_lse


def sinkhorn_potentials_ref(x, y, loga, logb, eps, n_iters):
    n = x.shape[0]
    m = y.shape[0]
    f = torch.zeros((n,), device=x.device, dtype=torch.float32)
    g = torch.zeros((m,), device=y.device, dtype=torch.float32)
    loga = loga.float()
    logb = logb.float()

    for _ in range(n_iters):
        g = update_potential_ref(x, y, f, g, logb, eps, axis=0)
        f = update_potential_ref(x, y, f, g, loga, eps, axis=1)
    return f, g


def apply_transport_from_potentials_ref(x, y, f, g, vec, eps, axis):
    lse_res, lse_sgn = apply_lse_kernel_ref(x, y, f, g, eps, axis, vec=vec)
    remove = f if axis == 1 else g
    lse_res = lse_res + remove
    return lse_sgn * torch.exp(lse_res / eps)


def log_weights_ref(w):
    w = w.float()
    out = w.log()
    out = torch.where(w > 0, out, torch.full_like(out, -100000.0))
    return out


def max_diameter_ref(x, y):
    x_f = x.float()
    y_f = y.float()
    mins = torch.stack((x_f.min(dim=0).values, y_f.min(dim=0).values)).min(dim=0).values
    maxs = torch.stack((x_f.max(dim=0).values, y_f.max(dim=0).values)).max(dim=0).values
    return (maxs - mins).norm().item()


def epsilon_schedule_ref(diameter, blur, scaling, p=2.0):
    import numpy as np

    eps_list = [diameter**p]
    eps_list += [
        float(np.exp(e))
        for e in np.arange(p * np.log(diameter), p * np.log(blur), p * np.log(scaling))
    ]
    eps_list += [blur**p]
    return eps_list


def sinkhorn_geomloss_potentials_ref(
    x,
    y,
    a,
    b,
    *,
    blur=0.05,
    scaling=0.5,
    use_epsilon_scaling=True,
    last_extrapolation=True,
    eps=None,
    n_iters=None,
    diameter=None,
    eps_list=None,
):
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors.")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1D tensors.")
    n = x.shape[0]
    m = y.shape[0]
    if a.shape[0] != n or b.shape[0] != m:
        raise ValueError("a and b shapes must match x and y.")

    if eps_list is None:
        if use_epsilon_scaling:
            if diameter is None:
                diameter = max_diameter_ref(x, y)
            eps_list = epsilon_schedule_ref(diameter, blur, scaling, p=2.0)
        else:
            if eps is None or n_iters is None:
                raise ValueError("Provide eps and n_iters when use_epsilon_scaling=False.")
            eps_list = [float(eps)] * int(n_iters)

    if len(eps_list) == 0:
        raise ValueError("eps_list must be non-empty.")
    if n_iters is not None:
        eps_list = list(eps_list)[: int(n_iters)]
        if len(eps_list) == 0:
            raise ValueError("n_iters is 0 after slicing eps_list.")

    loga = log_weights_ref(a)
    logb = log_weights_ref(b)
    cost = sqeuclid_cost(x, y)

    f = torch.zeros((n,), device=x.device, dtype=torch.float32)
    g = torch.zeros((m,), device=y.device, dtype=torch.float32)

    def softmin_x_from_y(step_eps, g_vec):
        vals = (g_vec[None, :] - cost) / float(step_eps) + logb[None, :]
        return -float(step_eps) * torch.logsumexp(vals, dim=1)

    def softmin_y_from_x(step_eps, f_vec):
        vals = (f_vec[:, None] - cost) / float(step_eps) + loga[:, None]
        return -float(step_eps) * torch.logsumexp(vals, dim=0)

    # Init at eps_list[0].
    eps0 = eps_list[0]
    f = softmin_x_from_y(eps0, torch.zeros_like(g))
    g = softmin_y_from_x(eps0, torch.zeros_like(f))

    for step_eps in eps_list:
        ft = softmin_x_from_y(step_eps, g)
        gt = softmin_y_from_x(step_eps, f)
        f = 0.5 * (f + ft)
        g = 0.5 * (g + gt)

    if last_extrapolation:
        step_eps = float(eps_list[-1])
        f_new = softmin_x_from_y(step_eps, g)
        g_new = softmin_y_from_x(step_eps, f)
        f, g = f_new, g_new

    return f, g


def sinkhorn_geomloss_plan_ref(x, y, a, b, f, g, eps: float):
    cost = sqeuclid_cost(x, y)
    logits = (f.float()[:, None] + g.float()[None, :] - cost) / float(eps)
    return (a.float()[:, None] * b.float()[None, :]) * torch.exp(logits)


def sinkhorn_geomloss_grads_ref(x, y, a, b, f, g, eps: float):
    p = sinkhorn_geomloss_plan_ref(x, y, a, b, f, g, eps)
    grad_x = 2.0 * (a.float()[:, None] * x.float() - p @ y.float())
    grad_y = 2.0 * (b.float()[:, None] * y.float() - p.transpose(0, 1) @ x.float())
    return grad_x, grad_y


def sinkhorn_geomloss_barycentric_grads_ref(x, y, a, b, f, g, eps: float):
    cost = sqeuclid_cost(x, y)
    eps = float(eps)

    a_f = a.float()
    b_f = b.float()
    x_f = x.float()
    y_f = y.float()
    f_f = f.float()
    g_f = g.float()

    # Conditional barycenters computed from softmin derivatives.
    logb = torch.where(b_f > 0, b_f.log(), torch.full_like(b_f, -float("inf")))
    loga = torch.where(a_f > 0, a_f.log(), torch.full_like(a_f, -float("inf")))

    logw_xy = (g_f[None, :] - cost) / eps + logb[None, :]
    w_xy = torch.softmax(logw_xy, dim=1)
    y_bar = w_xy @ y_f
    grad_x = 2.0 * a_f[:, None] * (x_f - y_bar)

    logw_yx = (f_f[:, None] - cost) / eps + loga[:, None]
    w_yx = torch.softmax(logw_yx, dim=0)
    x_bar = w_yx.transpose(0, 1) @ x_f
    grad_y = 2.0 * b_f[:, None] * (y_f - x_bar)

    return grad_x, grad_y
