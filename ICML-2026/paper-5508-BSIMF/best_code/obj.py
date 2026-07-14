import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F

from utils import LOG_VAR_MIN, LOG_VAR_MAX, _log_normal_diag
from model import ContentUncertaintyDAG


def _reparameterize_gaussian(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """
    Reparameterization trick for diagonal Gaussians.

    Returns samples with shape (num_samples, *mu.shape).
    """
    log_var = log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    std = torch.exp(0.5 * log_var)
    expanded_mu = mu.unsqueeze(0)  # (1, B, D)
    expanded_std = std.unsqueeze(0)
    eps = torch.randn(
        (num_samples,) + mu.shape,
        device=mu.device,
        dtype=mu.dtype,
    )
    return expanded_mu + eps * expanded_std  # (num_samples, B, D)


def _log_normal_lowrank_plus_diag(
    x: torch.Tensor,
    mean: torch.Tensor,
    diag_var: torch.Tensor,
    cov_factor: torch.Tensor,
    jitter: float = 1e-5,
) -> torch.Tensor:
    """
    Log-density of a multivariate Gaussian with covariance

        Sigma = diag(diag_var) + cov_factor cov_factor^T

    Args:
        x:          (B, D)
        mean:       (B, D)
        diag_var:   (B, D)
        cov_factor: (B, D, r)

    Returns:
        log_prob: (B,)
    """
    B, D = x.shape
    r = cov_factor.shape[2]
    if r == 0:
        log_var = torch.log(diag_var.clamp(min=1e-8))
        return _log_normal_diag(x, mean, log_var).sum(dim=-1)

    # Clamp diagonal variance for stability
    diag_var = diag_var.clamp(min=1e-6)
    D_inv = 1.0 / diag_var  # (B, D)

    diff = x - mean  # (B, D)

    # A = D^{-1} U
    A = cov_factor * D_inv.unsqueeze(-1)  # (B, D, r)

    # K = I + U^T D^{-1} U  (B, r, r)
    I = torch.eye(r, device=x.device, dtype=x.dtype).unsqueeze(0)  # (1, r, r)
    K = I + torch.einsum("bdr,bdq->brq", cov_factor, A)  # (B, r, r)
    if jitter > 0:
        K = K + jitter * I

    # Cholesky for logdet and solves
    L = torch.linalg.cholesky(K)  # (B, r, r)

    # logdet(Sigma) = sum log(diag_var) + logdet(K)
    logdet_D = torch.log(diag_var).sum(dim=-1)  # (B,)
    logdet_K = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + 1e-12).sum(dim=-1)  # (B,)
    logdet = logdet_D + logdet_K  # (B,)

    # quadratic form using Woodbury:
    # diff^T Sigma^{-1} diff = diff^T D^{-1} diff - v^T K^{-1} v,
    # where v = U^T D^{-1} diff = A^T diff
    quad0 = (diff.pow(2) * D_inv).sum(dim=-1)  # (B,)
    v = torch.einsum("bdr,bd->br", A, diff)  # (B, r)
    w = torch.cholesky_solve(v.unsqueeze(-1), L).squeeze(-1)  # (B, r)
    corr = (v * w).sum(dim=-1)  # (B,)
    quad = quad0 - corr

    log_prob = -0.5 * (D * math.log(2.0 * math.pi) + logdet + quad)  # (B,)
    return log_prob



def compute_image_likelihood_term_moment_matched(
    model: ContentUncertaintyDAG,
    y: torch.Tensor,
    encode_out: Dict[str, torch.Tensor],
    num_samples_z: int = 4,
) -> torch.Tensor:
    """Moment-matched surrogate for E_{q_Z,q_M}[log p(y|z,m)].

    Returns:
        L_Y: (B,)
    """
    B = y.size(0)
    y_flat = y.view(B, -1)  # (B, d_Y^2)

    # Base distribution parameters (shared across z)
    mu_base, log_var_base = model.image_base_params(batch_size=B)  # (B, d_Y^2)
    log_var_base = log_var_base.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    var_base = torch.exp(log_var_base)  # (B, d_Y^2)

    # Sample z ~ q(z | x, y)
    mu_z = encode_out["mu_z"]  # (B, d_Z)
    log_var_z = encode_out["log_var_z"]  # (B, d_Z)
    z_samples = _reparameterize_gaussian(mu_z, log_var_z, num_samples_z)  # (L, B, d_Z)

    y_tokens = encode_out["y_tokens"]  # (B, N_patches, D_y_feat)

    per_sample_terms = []
    for l in range(num_samples_z):
        z_l = z_samples[l]  # (B, d_Z)

        # Content mean and covariance factor
        if hasattr(model, "image_content_factor"):
            mu_cont_l, cov_factor_l = model.image_content_factor(z_l)  # (B, d_Y^2), (B, d_Y^2, r)
        else:
            mu_cont_l, cov_factor_l = model.image_decoder.content_factor(z_l)

        # Mask probabilities g = q(m=1|y,z) (background probability)
        g_l = model.mask_probs(y_tokens, z_l).clamp(1e-6, 1.0 - 1e-6)  # (B, d_Y^2)

        # Moment-matched Gaussian parameters
        mu_mm = g_l * mu_base + (1.0 - g_l) * mu_cont_l  # (B, d_Y^2)

        delta_mu = mu_base - mu_cont_l  # (B, d_Y^2)
        var_extra = cov_factor_l.pow(2).sum(dim=-1)  # diag(GG^T), (B, d_Y^2)

        # Diagonal term for moment-matched covariance
        mix_var = g_l * (1.0 - g_l)  # (B, d_Y^2)
        diag_var_mm = var_base + mix_var * (var_extra + delta_mu.pow(2))  # (B, d_Y^2)

        # Low-rank term for moment-matched covariance: ((1-g) ⊙ G)((1-g) ⊙ G)^T
        cov_factor_eff = (1.0 - g_l).unsqueeze(-1) * cov_factor_l  # (B, d_Y^2, r)

        # log N(y; mu_mm, Sigma_mm)
        term_l = _log_normal_lowrank_plus_diag(
            x=y_flat,
            mean=mu_mm,
            diag_var=diag_var_mm,
            cov_factor=cov_factor_eff,
        )  # (B,)
        per_sample_terms.append(term_l)

    L_Y = torch.stack(per_sample_terms, dim=0).mean(dim=0)  # (B,)
    return L_Y


def compute_image_likelihood_term(
    model: ContentUncertaintyDAG,
    y: torch.Tensor,
    encode_out: Dict[str, torch.Tensor],
    num_samples_z: int = 4,
) -> torch.Tensor:
    """Factorized objective-side likelihood term :math:`\mathcal{L}_Y` (paper).

    Implements the expression used in the paper:

    .. math::

        \mathcal{L}_Y
        = \mathbb{E}_{q(Z)}\Big[\sum_i g_i(y,z)\,\ell_{b,i}
          + (1-g_i(y,z))\,\ell_{c,i}(z)\Big],

    where

    * :math:`g_i(y,z) = q(m_i=1\mid y,z)` is the *background* probability,
    * :math:`\ell_{b,i}=\log \mathcal{N}(y_i;\mu_{\text{base},i},\sigma^2_{\text{base},i})`,
    * :math:`\ell_{c,i}(z)=\log \mathcal{N}(y_i;\mu_{\text{cont},i}(z),\Sigma_{\text{cont},ii}(z))`.

    In the implementation we use the *diagonal marginal* variance
    :math:`\Sigma_{\text{cont},ii}(z)` from the low-rank parameterization

    .. math::
        \Sigma_{\text{cont}}(z) = \mathrm{diag}(\sigma^2_{\text{base}}) + G(z)G(z)^\top,

    so that

    .. math::
        \Sigma_{\text{cont},ii}(z) = \sigma^2_{\text{base},i} + \|G_i(z)\|_2^2.

    This matches the factorized likelihood used in the derivation and produces
    the expected gradients on :math:`g` (linear interpolation of log-densities).

    Returns:
        L_Y: (B,) tensor
    """
    B = y.size(0)
    y_flat = y.view(B, -1)  # (B, d_Y)

    # Base distribution parameters
    mu_base, log_var_base = model.image_base_params(batch_size=B)  # (B, d_Y)
    log_var_base = log_var_base.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    ll_base = _log_normal_diag(y_flat, mu_base, log_var_base)  # (B, d_Y)

    # Sample z ~ q(z | x, y)
    mu_z = encode_out["mu_z"]  # (B, d_Z)
    log_var_z = encode_out["log_var_z"]  # (B, d_Z)
    z_samples = _reparameterize_gaussian(mu_z, log_var_z, num_samples_z)  # (L, B, d_Z)

    y_tokens = encode_out["y_tokens"]  # (B, N_patches, D_y_feat)

    eps = 1e-8
    per_sample_terms = []
    for l in range(num_samples_z):
        z_l = z_samples[l]  # (B, d_Z)

        # Content mean and diagonal marginal variance
        mu_cont_l, log_var_cont_l = model.image_content_params(z_l)  # (B, d_Y), (B, d_Y)
        log_var_cont_l = log_var_cont_l.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        ll_cont = _log_normal_diag(y_flat, mu_cont_l, log_var_cont_l)  # (B, d_Y)

        # Mask probabilities g = q(m=1|y,z)
        g_l = model.mask_probs(y_tokens, z_l).clamp(eps, 1.0 - eps)  # (B, d_Y)

        # E_{q(m|y,z)}[log p(y|z,m)]: sum_i g_i * ll_base_i + (1-g_i) * ll_cont_i
        term_l = (g_l * ll_base + (1.0 - g_l) * ll_cont).sum(dim=1)  # (B,)
        per_sample_terms.append(term_l)

    L_Y = torch.stack(per_sample_terms, dim=0).mean(dim=0)  # (B,)
    return L_Y


def compute_label_likelihood_term(
    model: ContentUncertaintyDAG,
    x: torch.Tensor,
    encode_out: Dict[str, torch.Tensor],
    num_samples: int = 4,
    mask_x: Optional[torch.Tensor] = None,
    x_encoder: Optional[torch.Tensor] = None,
    mask_x_encoder: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    L_X: label likelihood term.

    Args:
        model: ContentUncertaintyDAG
        x: (B, d_X)
        encode_out: output dict from model.encode(...)
        num_samples: MC samples for categorical case
        mask_x: (B, d_X) float mask (1=observed, 0=missing)

    Returns:
        L_X: (B,)
    """
    mu_z = encode_out["mu_z"]
    log_var_z = encode_out["log_var_z"]
    mu_u = encode_out["mu_u"]
    log_var_u = encode_out["log_var_u"]

    if model.x_distribution == "continuous":
        if mask_x is None:
            mask_x = torch.isfinite(x).to(dtype=x.dtype, device=x.device)
        else:
            mask_x = mask_x.to(dtype=x.dtype, device=x.device)
        x_in = torch.where(mask_x.bool(), x, torch.zeros_like(x))

        mean_x, log_var_x = model.decode_x_continuous(mu_z, mu_u)  # (B, d_X)
        log_var_x = log_var_x.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        var_x = torch.exp(log_var_x)  # (B, d_X)

        s_z2 = torch.exp(log_var_z.clamp(LOG_VAR_MIN, LOG_VAR_MAX))  # (B, d_Z)
        s_u2 = torch.exp(log_var_u.clamp(LOG_VAR_MIN, LOG_VAR_MAX))  # (B, d_U)

        A_sq = model.x_decoder.A.pow(2)  # (d_X, d_Z)
        var_z_contrib = s_z2.matmul(A_sq.t())  # (B, d_X)  diag(A diag(s_z^2) A^T)
        var_u_contrib = s_u2  # (B, d_X)

        inv_var = 1.0 / (var_x + 1e-8)
        term_data = (x_in - mean_x).pow(2) * inv_var
        term_lat_z = var_z_contrib * inv_var
        term_lat_u = var_u_contrib * inv_var
        term_log = torch.log(2.0 * math.pi * var_x + 1e-8)

        L_X = -0.5 * ((term_data + term_lat_z + term_lat_u + term_log) * mask_x).sum(dim=1)
        return L_X

    elif model.x_distribution == "categorical":
        if mask_x is None:
            mask_x = (x >= 0).to(dtype=torch.float32, device=x.device)
        else:
            mask_x = mask_x.to(dtype=torch.float32, device=x.device)

        x_in = torch.where(mask_x.bool(), x, torch.zeros_like(x))

        # Monte Carlo over q_Z and q_U
        z_samples = _reparameterize_gaussian(mu_z, log_var_z, num_samples)
        u_samples = _reparameterize_gaussian(mu_u, log_var_u, num_samples)

        per_sample_terms = []
        mask_f = mask_x.to(dtype=torch.float32, device=x.device)

        for l in range(num_samples):
            z_l = z_samples[l]
            u_l = u_samples[l]
            logits = model.decode_x_categorical(z_l, u_l)  # (B, d_X, C)
            log_probs = F.log_softmax(logits, dim=-1)      # (B, d_X, C)

            target = x_in.long().unsqueeze(-1)  # (B, d_X, 1)
            ll = log_probs.gather(dim=-1, index=target).squeeze(-1)  # (B, d_X)
            ll = ll * mask_f  # mask out missing label entries

            per_sample_terms.append(ll.sum(dim=1))  # (B,)

        L_X = torch.stack(per_sample_terms, dim=0).mean(dim=0)  # (B,)
        return L_X

def compute_kl_zc_term(
    model: ContentUncertaintyDAG,
    encode_out: Dict[str, torch.Tensor],
    num_samples_z: int = 4,
) -> torch.Tensor:
    """
    Compute
        KL(q(z,c | x,y) || p(z,c))

    Using the expression (with Monte Carlo over q_Z):
        KL = E_Z[ log q(z | x,y) - log p(z) ]
    where
        p(z) = sum_k pi_k N(z; mu_k, Sigma_k)
    with fully vector diagonal covariances Sigma_k.
    """
    mu_z = encode_out["mu_z"]  # (B, d_Z)
    log_var_z = encode_out["log_var_z"]  # (B, d_Z)
    B, d_z = mu_z.shape

    z_samples = _reparameterize_gaussian(mu_z, log_var_z, num_samples_z)  # (L, B, d_Z)
    var_z = torch.exp(log_var_z)  # (B, d_Z)

    # log q(z | x,y) for each sample
    mu_z_exp = mu_z.unsqueeze(0).expand_as(z_samples)  # (L, B, d_Z)
    log_var_z_exp = log_var_z.unsqueeze(0).expand_as(z_samples)
    var_z_exp = var_z.unsqueeze(0).expand_as(z_samples)

    diff_q = z_samples - mu_z_exp  # (L, B, d_Z)
    log_q = -0.5 * (
        d_z * math.log(2.0 * math.pi)
        + log_var_z_exp.sum(dim=-1)
        + (diff_q.pow(2) / (var_z_exp + 1e-8)).sum(dim=-1)
    )  # (L, B)

    # log p(z) under mixture prior with vector variances
    pi_logits = model.pi_logits  # (K,)
    log_pi = F.log_softmax(pi_logits, dim=0)  # (K,)
    mu_components = model.mixture_means        # (K, d_Z)
    log_var_components = model.mixture_log_vars  # (K, d_Z)
    var_components = torch.exp(log_var_components)  # (K, d_Z)

    K = mu_components.size(0)
    z_flat = z_samples.reshape(num_samples_z * B, d_z)  # (L*B, d_Z)

    diff_p = z_flat.unsqueeze(1) - mu_components.unsqueeze(0)  # (L*B, K, d_Z)
    mahal = (diff_p.pow(2) / (var_components.unsqueeze(0) + 1e-8)).sum(dim=-1)  # (L*B, K)

    log_det = log_var_components.sum(dim=-1)  # (K,)
    log_comp = -0.5 * (
        d_z * math.log(2.0 * math.pi)
        + log_det.view(1, K)
        + mahal
    )  # (L*B, K)

    log_p_flat = torch.logsumexp(
        log_pi.view(1, K) + log_comp, dim=1
    )  # (L*B,)

    log_p = log_p_flat.view(num_samples_z, B)  # (L, B)

    kl_per_sample = (log_q - log_p)  # (L, B)
    KL_zc = kl_per_sample.mean(dim=0)  # (B,)
    return KL_zc


def compute_kl_u_term(
    model: ContentUncertaintyDAG,
    y: torch.Tensor,
    encode_out: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute KL(q(u | x,y) || p(u | y)) analytically for diagonal Gaussians (vector).
    """
    mu_u = encode_out["mu_u"]  # (B, d_U)
    log_var_u = encode_out["log_var_u"]  # (B, d_U)
    var_u = torch.exp(log_var_u)

    y_global = encode_out.get("y_global", None)
    if y_global is None:
        _, y_global = model.vit(y)
    mu_psi, log_var_psi = model.u_prior_net(y_global)  # (B, d_U)
    var_psi = torch.exp(log_var_psi)

    # KL between diagonal Gaussians
    kl = 0.5 * torch.sum(
        (var_u / (var_psi + 1e-8))
        + (mu_psi - mu_u) ** 2 / (var_psi + 1e-8)
        - 1.0
        + log_var_psi
        - log_var_u,
        dim=-1,
    )  # (B,)
    return kl


def compute_kl_m_term(
    model: ContentUncertaintyDAG,
    y: torch.Tensor,
    encode_out: Dict[str, torch.Tensor],
    num_samples_z: int = 4,
) -> torch.Tensor:
    """
    E_{q_Z} KL(q(m | y,z) || p(m)).

    KL = sum_i [ g_i log(g_i / rho_i) + (1-g_i) log((1-g_i)/(1-rho_i)) ]
    """

    B = y.size(0)
    y_tokens = encode_out["y_tokens"]  # (B, N_patches, D_y_feat)
    mu_z = encode_out["mu_z"]
    log_var_z = encode_out["log_var_z"]

    z_samples = _reparameterize_gaussian(mu_z, log_var_z, num_samples_z)  # (L, B, d_Z)

    rho = model.mask_prior_probs  # (d_Y^2,)
    rho_expanded = rho.unsqueeze(0).expand(B, -1)  # (B, d_Y^2)

    eps = 1e-8
    per_sample_terms = []
    for l in range(num_samples_z):
        z_l = z_samples[l]  # (B, d_Z)
        # Pixel-level mask probabilities
        g_l = model.mask_probs(y_tokens, z_l)  # (B, d_Y^2)
        g_l = torch.clamp(g_l, eps, 1.0 - eps)

        kl_pixel = g_l * (torch.log(g_l) - torch.log(rho_expanded + eps)) + (
            1.0 - g_l
        ) * (
            torch.log(1.0 - g_l) - torch.log(1.0 - rho_expanded + eps)
        )  # (B, d_Y^2)

        per_sample_terms.append(kl_pixel.sum(dim=1))  # (B,)

    KL_m = torch.stack(per_sample_terms, dim=0).mean(dim=0)  # (B,)
    return KL_m


def compute_sparse_m_term(
    model: ContentUncertaintyDAG,
    y: torch.Tensor,
    encode_out: Dict[str, torch.Tensor],
    num_samples_z: int = 4,
    sparse_on: str = "content",
    target: Optional[float] = None,
) -> torch.Tensor:
    """
    Sparsity regularizer for the mask probabilities.

    Returns:
        penalty: (B,)
    """
    y_tokens = encode_out["y_tokens"]  # (B, N_patches, D_y_feat)
    mu_z = encode_out["mu_z"]
    log_var_z = encode_out["log_var_z"]

    z_samples = _reparameterize_gaussian(mu_z, log_var_z, num_samples_z)  # (L, B, d_Z)

    per_sample_terms = []
    for l in range(num_samples_z):
        z_l = z_samples[l]
        g_l = model.mask_probs(y_tokens, z_l)  # (B, d_Y^2)

        if sparse_on == "content":
            p = 1.0 - g_l  # content probability
        elif sparse_on == "mask" or sparse_on == "background":
            p = g_l        # m=1 probability

        # Use mean over pixels => expected fraction in [0,1]
        frac = p.mean(dim=1)  # (B,)

        if target is None:
            term = frac
        else:
            term = (frac - float(target)) ** 2

        per_sample_terms.append(term)

    return torch.stack(per_sample_terms, dim=0).mean(dim=0)  # (B,)




def compute_mask_tv_term(
    model: ContentUncertaintyDAG,
    encode_out: Dict[str, torch.Tensor],
    num_samples_z: int = 1,
) -> torch.Tensor:
    """TV regularizer on mask probabilities g(y,z).

    TV(g) = mean(|g[i,j]-g[i+1,j]|) + mean(|g[i,j]-g[i,j+1]|)

    Returns:
        penalty: (B,)
    """
    y_tokens = encode_out["y_tokens"]
    mu_z = encode_out["mu_z"]
    log_var_z = encode_out["log_var_z"]

    if num_samples_z == 1:
        z_samples = mu_z.unsqueeze(0)  # (1, B, d_Z)
    else:
        z_samples = _reparameterize_gaussian(mu_z, log_var_z, num_samples_z)  # (L, B, d_Z)

    B = mu_z.size(0)
    H = int(model.image_size)
    W = int(model.image_size)

    per = []
    for l in range(num_samples_z):
        z_l = z_samples[l]
        g = model.mask_probs(y_tokens, z_l).clamp(1e-6, 1.0 - 1e-6)  # (B, H*W)
        g2 = g.view(B, H, W)

        tv_h = (g2[:, 1:, :] - g2[:, :-1, :]).abs().mean(dim=(1, 2))
        tv_w = (g2[:, :, 1:] - g2[:, :, :-1]).abs().mean(dim=(1, 2))
        per.append(tv_h + tv_w)

    return torch.stack(per, dim=0).mean(dim=0)  # (B,)
def compute_elbo(
    model: ContentUncertaintyDAG,
    x: torch.Tensor,
    y: torch.Tensor,
    num_samples_z: int = 4,
    num_samples_u: int = 4,
    average_over_batch: bool = True,
    sparse_m_lambda: float = 0.0,
    sparse_m_target: Optional[float] = None,
    sparse_m_on: str = "content",
    mask_tv_lambda: float = 0.0,
    mask_tv_samples: int = 1,
    mask_x: Optional[torch.Tensor] = None,
    x_encoder: Optional[torch.Tensor] = None,
    mask_x_encoder: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the full ELBO (per data point or averaged over batch):

        L = E_q[ log p(x,y,z,u,m,c) - log q(z,u,m | x,y) ]
          = L_Y + L_X - KL_zc - KL_u - KL_m - sparse_m_lambda * Sparse_M
    """
    if model.x_distribution == "continuous":
        if mask_x is None:
            mask_x_like = torch.isfinite(x).to(dtype=x.dtype, device=x.device)
        else:
            mask_x_like = mask_x.to(dtype=x.dtype, device=x.device)
        x_like_in = torch.where(mask_x_like.bool(), x, torch.zeros_like(x))
    else:
        if mask_x is None:
            mask_x_like = (x >= 0).to(dtype=torch.float32, device=x.device)
        else:
            mask_x_like = mask_x.to(dtype=torch.float32, device=x.device)
        x_like_in = torch.where(mask_x_like.bool(), x, torch.zeros_like(x))

    x_enc = x if x_encoder is None else x_encoder
    if model.x_distribution == "continuous":
        if mask_x_encoder is None:
            mask_x_enc = mask_x_like
        else:
            mask_x_enc = mask_x_encoder.to(dtype=x.dtype, device=x.device)
        x_enc_in = torch.where(mask_x_enc.bool(), x_enc, torch.zeros_like(x_enc))
    else:
        if mask_x_encoder is None:
            mask_x_enc = mask_x_like
        else:
            mask_x_enc = mask_x_encoder.to(dtype=torch.float32, device=x.device)
        x_enc_in = torch.where(mask_x_enc.bool(), x_enc, torch.zeros_like(x_enc))

    encode_out = model.encode(x_enc_in, y, mask_x=mask_x_enc)

    L_Y = compute_image_likelihood_term(
        model=model,
        y=y,
        encode_out=encode_out,
        num_samples_z=num_samples_z,
    )
    L_X = compute_label_likelihood_term(
        model=model,
        x=x_like_in,
        encode_out=encode_out,
        num_samples=num_samples_u,
        mask_x=mask_x_like,
    )
    KL_zc = compute_kl_zc_term(
        model=model,
        encode_out=encode_out,
        num_samples_z=num_samples_z,
    )
    KL_u = compute_kl_u_term(
        model=model,
        y=y,
        encode_out=encode_out,
    )
    KL_m = compute_kl_m_term(
        model=model,
        y=y,
        encode_out=encode_out,
        num_samples_z=num_samples_z,
    )

    # Optional sparsity penalty on the mask probabilities
    if sparse_m_lambda != 0.0:
        Sparse_M = compute_sparse_m_term(
            model=model,
            y=y,
            encode_out=encode_out,
            num_samples_z=num_samples_z,
            sparse_on=sparse_m_on,
            target=sparse_m_target,
        )
    else:
        Sparse_M = torch.zeros_like(KL_m)

    # Optional spatial coherence penalty on the mask probabilities (Total Variation)
    if mask_tv_lambda != 0.0:
        TV_M = compute_mask_tv_term(model=model, encode_out=encode_out, num_samples_z=mask_tv_samples)
    else:
        TV_M = torch.zeros_like(KL_m)

    elbo_per_sample = 1.0/1000 * L_Y + L_X - 1.0/100 *KL_zc - KL_u - 1.0/1000 * KL_m - sparse_m_lambda * Sparse_M - mask_tv_lambda * TV_M

    if average_over_batch:
        elbo = elbo_per_sample.mean()
        terms = {
            "L_Y": L_Y.mean(),
            "L_X": L_X.mean(),
            "KL_zc": KL_zc.mean(),
            "KL_u": KL_u.mean(),
            "KL_m": KL_m.mean(),
            "Sparse_M": Sparse_M.mean(),
            "TV_M": TV_M.mean(),
        }
        return elbo, terms
    else:
        # Return per-sample vectors
        terms = {
            "L_Y": L_Y,
            "L_X": L_X,
            "KL_zc": KL_zc,
            "KL_u": KL_u,
            "KL_m": KL_m,
            "Sparse_M": Sparse_M,
            "TV_M": TV_M,
        }
        return elbo_per_sample, terms
