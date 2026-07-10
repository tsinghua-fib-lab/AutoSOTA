# Calculates the translation invariant KL-regularized unbalanced OT cost (not the map!)
# using code from:
# https://github.com/thibsej/fast_uot/tree/main
import torch

import torch._dynamo

torch._dynamo.config.suppress_errors = True


def sinkx(C, f, a, eps):
    return -eps * (a.log()[:, None] + (f[:, None] - C) / eps).logsumexp(dim=0)


def sinky(C, g, b, eps):
    return -eps * (b.log()[None, :] + (g[None, :] - C) / eps).logsumexp(dim=1)


def softmin(a, f, rho):
    return -rho * (a.log() - f / rho).logsumexp(dim=0)


def aprox(f, eps, rho):
    return (1.0 / (1.0 + (eps / rho))) * f


def kl_entropy(x):
    return x * torch.log(x + 1e-16) - x + 1


def primal_cost(pi, a, b, eps, rho):
    pi1, pi2 = torch.sum(pi, dim=1), torch.sum(pi, dim=0)
    cost = rho * torch.sum(a * kl_entropy(pi1 / a))
    cost = cost + rho * torch.sum(b * kl_entropy(pi2 / b))
    cost = cost - eps * torch.sum(pi) + eps * torch.sum(a) * torch.sum(b)
    return cost


def f_sinkhorn_loop(f, a, b, C, eps, rho, rho2):
    # Update on G
    g = sinkx(C, f, a, eps)
    g = aprox(g, eps, rho2)

    # Update on F
    f = sinky(C, g, b, eps)
    f = aprox(f, eps, rho)
    return f, g


def h_sinkhorn_loop(f, a, b, C, eps, rho, rho2, k1, k2, xi1, xi2):
    g = aprox(sinkx(C, f, a, eps), eps, rho) - k2 * softmin(a, f, rho)
    g = g + xi2 * softmin(b, g, rho2)
    f = aprox(sinky(C, g, b, eps), eps, rho) - k1 * softmin(b, g, rho2)
    f = f + xi1 * softmin(a, f, rho)
    return f, g


@torch.compile(mode="max-autotune")
def fast_uot_sinkhorn(a, b, C, eps, rho, rho2, n_iter, thresh, verbose=False):
    k1 = 1.0 / ((1.0 + (rho / eps)) * (1.0 + (rho2 / rho)))
    k2 = 1.0 / ((1.0 + (rho2 / eps)) * (1.0 + (rho / rho2)))
    xi1 = rho2 / (rho * (1.0 + (rho / eps) + (rho2 / eps)))
    xi2 = rho / (rho2 * (1.0 + (rho / eps) + (rho2 / eps)))
    f_h = torch.zeros_like(a)
    f_h_old = torch.zeros_like(a)
    for i in range(n_iter):
        f_h_old = f_h
        # f_h, g_h = f_sinkhorn_loop(f_h, a, b, C, eps, rho, rho2)
        f_h, g_h = h_sinkhorn_loop(f_h, a, b, C, eps, rho, rho2, k1, k2, xi1, xi2)
        if verbose:
            print(f"Error norm at iter {i}: {torch.linalg.norm(f_h - f_h_old)}")
        if torch.linalg.norm(f_h - f_h_old) < thresh:
            break
    return torch.exp((f_h[:, None] + g_h[None, :] - C) / eps) * a[:, None] * b[None, :]


def mm_unbalanced(
    a, b, C, rho, rho2, numItermax=1000, G0=None, autograd_at_convergence=False
):
    """MM-based unbalanced OT solver using POT's optimized CPU implementation
    for bulk iterations, with an optional final GPU iteration for autograd.

    Uses the POT library's mm_unbalanced solver (Chapel et al., 2021) which
    matches the paper's description: "It is implemented as part of the POT
    library (Flamary et al., 2021)". For gradient computation, the OT plan
    is detached; if autograd_at_convergence is True, one final GPU iteration
    is performed with autograd enabled for correct gradient flow.
    """
    import ot
    device = a.device
    # Convert to numpy for POT (CPU-based, much faster than custom GPU impl)
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    C_np = C.detach().cpu().numpy()

    # Use POT's efficient implementation
    G_np = ot.unbalanced.mm_unbalanced(
        a_np, b_np, C_np,
        reg_m=(float(rho), float(rho2)),
        numItermax=numItermax,
        stopThr=1e-6,
    )
    G = torch.from_numpy(G_np).to(device=device, dtype=C.dtype)

    if autograd_at_convergence:
        # One final GPU iteration with autograd enabled for gradient flow
        sum_r = rho + rho2
        r1 = rho / sum_r
        r2 = rho2 / sum_r
        K = (a[:, None] ** r1) * (b[None, :] ** r2) * torch.exp(-C / sum_r)
        Gd = (torch.sum(G, 1, keepdims=True) ** r1) * (
            torch.sum(G, 0, keepdims=True) ** r2
        ) + 1e-16
        G = K * G ** (r1 + r2) / Gd
    return G
