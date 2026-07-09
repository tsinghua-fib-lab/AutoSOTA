import torch
import logging

logger = logging.getLogger(__name__)


def get_lr_initializer(init_method: str):
    init_method = init_method.lower()
    if init_method == 'srr':     return srr_init
    elif init_method == 'zero':   return zero_init
    else:
        raise ValueError(f"Unknown init method: {init_method}")


def zero_init(weight: torch.Tensor, scale: torch.Tensor, rank: int, layer_qera_config: dict):
    # LR=0
    device = weight.device
    dtype = weight.dtype
    out_dim, in_dim = weight.shape
    L = torch.zeros((in_dim, rank), dtype=dtype, device=device)
    R = torch.zeros((rank, out_dim), dtype=dtype, device=device)
    return L, R


def srr_init(weight: torch.Tensor, scale: torch.Tensor, rank: int, layer_qera_config: dict, return_ada = False):
    device, dtype = weight.device, weight.dtype
    W = weight.to(device=device, dtype=dtype)
    scale = scale.to(device=device, dtype=dtype)
    use_rand_svd = layer_qera_config.get("apply_rand_svd", False)

    if scale.ndim == 1:
        scaled_W = scale.view(-1, 1) * W.T
        inv_scale_vec = 1.0 / scale
    elif scale.ndim == 2:
        scaled_W = scale @ W.T
        inv_scale = torch.linalg.inv(scale)
    else:
        raise ValueError("Scale must be either 1D or 2D")

    if use_rand_svd:
        total_energy = torch.norm(scaled_W) ** 2
        U, S_vals, _ = torch.svd_lowrank(scaled_W, q=rank * 2, niter=4)
    else:
        U, S_vals, _ = torch.linalg.svd(scaled_W, full_matrices=False)
        total_energy = None

    num_mc_samples = layer_qera_config.get("num_mc_samples", 1)

    k_star, _ = find_optimal_k(
        M=scaled_W.shape[1],
        N=scaled_W.shape[0],
        sigma_SW=S_vals,
        scale=scale,
        rank_budget=rank,
        total_sigma_sq=total_energy,
        use_rand_svd=use_rand_svd,
        num_mc_samples=num_mc_samples,
    )

    U_top = U[:, :k_star]
    print(f"Adaptive Rank selected: {k_star}")

    if scale.ndim == 1:
        Ph_sorted = U_top @ U_top.T
        Ph_scaled = (
            Ph_sorted
            * inv_scale_vec.view(-1, 1)
            * scale.view(1, -1)
        )
    elif scale.ndim == 2:
        if use_rand_svd:
            left_mat = inv_scale @ U_top
            right_mat_T = scale.T @ U_top
            Ph_scaled = left_mat @ right_mat_T.T
        else:
            Ph_sorted = U_top @ U_top.T
            Ph_scaled = inv_scale @ (Ph_sorted @ scale)

    L = Ph_scaled
    R = W.T

    if return_ada:
        return L, R, k_star
    else:
        return L, R


@torch.no_grad()
def find_optimal_k(
    M: int,
    N: int,
    sigma_SW: torch.Tensor,
    scale: torch.Tensor,
    rank_budget: int,
    total_sigma_sq: torch.Tensor = None,
    use_rand_svd: bool = False,
    num_mc_samples: int = 1,
):
    """
    MC-based uniform-iid noise proxy for estimating rho(q).

    When num_mc_samples > 1, rho is averaged over multiple independent
    noise realizations to reduce variance in k_star selection.
    This adds negligible runtime (~1s per layer at 5 MC samples)
    while producing more consistent rank split decisions.

    Objective:
      obj(k) = alpha_k^2 * rho(r-k)
      alpha_k^2 = sum_{i>k} sigma_i(SW)^2
      rho(q) estimated from sample Y = S E,
        rho(q) = 1 - ||Y_q||_F^2 / ||Y||_F^2
    """
    device = sigma_SW.device
    dtype = sigma_SW.dtype
    r = int(rank_budget)
    eps = torch.finfo(dtype).eps

    m = int(scale.shape[0]) if scale.dim() == 2 else int(scale.numel())

    ks = torch.arange(r + 1, device=device)
    qs = r - ks

    s = sigma_SW.to(device=device, dtype=dtype)

    if total_sigma_sq is not None:
        total = total_sigma_sq
    else:
        total = (s.square()).sum()

    if use_rand_svd:
        top = s[:r]
    else:
        top = torch.topk(s, k=r, largest=True, sorted=True).values

    ctop = torch.cumsum(top.square(), dim=0)

    prefix = torch.zeros(r + 1, device=device, dtype=dtype)
    prefix[1:] = ctop
    alpha2 = (total - prefix).clamp_min(0.0)

    S = scale.to(device=device, dtype=dtype)
    def apply_S(X: torch.Tensor) -> torch.Tensor:
        return (S.view(-1, 1) * X) if S.dim() == 1 else (S @ X)

    a = (12.0 / max(int(N), 1)) ** 0.5

    # Accumulate rho over multiple MC samples to reduce variance
    rho_sum = torch.zeros(r + 1, device=device, dtype=dtype)
    for mc_idx in range(num_mc_samples):
        E = (torch.rand(m, int(N), device=device, dtype=dtype) - 0.5) * a
        Y = apply_S(E)

        norm2 = (Y.square()).sum().clamp_min(eps)

        rho_vec = torch.ones(r + 1, device=device, dtype=dtype)
        if use_rand_svd:
            _, svals_Y, _ = torch.svd_lowrank(Y, q=r * 2, niter=4)
            svals_Y, _ = torch.sort(svals_Y, descending=True)
            svals = svals_Y[:r]
        else:
            svals = torch.linalg.svdvals(Y)[:r]
        cs2 = torch.cumsum(svals.square(), dim=0)
        rho_vec[1:] = (1.0 - cs2 / norm2).clamp(0.0, 1.0)
        rho_sum += rho_vec

    rho_avg = rho_sum / num_mc_samples
    rho = rho_avg[qs.to(torch.long)]
    obj = alpha2 * rho
    best_k = int(torch.argmin(obj).item())
    return best_k, obj.tolist()



def _compute_scale_inv_dot_U(scale: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    # scale^-1 @ U

    if scale.ndim == 1:
        scale = torch.where(
            scale <= 0, torch.ones_like(scale) * torch.finfo(scale.dtype).eps, scale
        )
        return torch.linalg.solve(torch.diag(scale), U)
    elif scale.ndim == 2:
        try:
            return torch.linalg.solve(scale, U)
        except RuntimeError as e:
            logger.warning(
                f"Matrix inversion failed: {e} Adding turbulence to scale"
            )
            U_scale, S_scale, V_T_scale = torch.linalg.svd(scale)
            S_scale = torch.where(
                S_scale <= 0,
                torch.ones_like(S_scale) * torch.finfo(S_scale.dtype).eps,
                S_scale,
            )
            scale = U_scale @ torch.diag(S_scale) @ V_T_scale
            return torch.linalg.solve(scale, U)
    else:
        raise ValueError("Scale must be either a vector (diagonal) or a matrix")