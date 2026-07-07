"""
Input preprocessing for InfoAtlas inference.

InfoAtlas does not consume raw samples directly. Before a batch of paired
observations is fed to the model it is mapped through a *soft-rank Gaussian
copula* transform and padded with Gaussian noise up to the model's maximum
input dimension. These two steps make the estimator invariant to monotone
marginal transforms and let a single model handle inputs of any dimension
``d <= max_dim``.

This module exposes those transforms as standalone, dependency-light functions
(only ``torch``, ``torchsort`` and ``einops`` are required) so that the
inference API (``infer.py``) and the training-time validation (``evaluation.py``)
can preprocess data without pulling in any training-data-generation code.

Both a CPU/host variant and a device-preserving ``*_gpu`` variant are provided;
they are numerically equivalent.
"""

import torch
import torchsort
from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Scaling
# ============================================================

def scale_data_pytorch(input_tensor, method="linear_scaling"):
    if method == "linear_scaling":
        min_val = torch.min(input_tensor, dim=1, keepdim=True).values
        max_val = torch.max(input_tensor, dim=1, keepdim=True).values
        scaled_tensor = 2 * (input_tensor - min_val) / (max_val - min_val) - 1
    elif method == "standardization":
        mean_val = torch.mean(input_tensor, dim=1, keepdim=True)
        std_val = torch.std(input_tensor, dim=1, keepdim=True)
        scaled_tensor = (input_tensor - mean_val) / std_val
    else:
        raise ValueError(f"Unknown method: {method}")
    return scaled_tensor


def scale_data_pytorch_gpu(input_tensor, method="linear_scaling"):
    """Device-preserving variant of scale_data_pytorch."""
    if method == "linear_scaling":
        min_val = torch.min(input_tensor, dim=1, keepdim=True).values
        max_val = torch.max(input_tensor, dim=1, keepdim=True).values
        scaled_tensor = 2 * (input_tensor - min_val) / (max_val - min_val) - 1
    elif method == "standardization":
        mean_val = torch.mean(input_tensor, dim=1, keepdim=True)
        std_val = torch.std(input_tensor, dim=1, keepdim=True)
        scaled_tensor = (input_tensor - mean_val) / std_val
    else:
        raise ValueError(f"Unknown method: {method}")
    return scaled_tensor


# ============================================================
# Soft-rank Gaussian copula transform
# ============================================================

def softrank_preprocessing_new(input_tensor, regularization_strength=0.1, gauss_copula=True, gauss_range=1.1):
    b, n, d = input_tensor.shape
    scaled_tensor = input_tensor

    softrank = torchsort.soft_rank(rearrange(scaled_tensor, 'b n d -> (d b) n'), regularization_strength=regularization_strength)
    softrank = rearrange(softrank, '(d b) n -> b n d', d=d)

    softrank = torchsort.soft_rank(rearrange(scaled_tensor, 'b n d -> (d b) n'), regularization_strength=regularization_strength)
    softrank = rearrange(softrank, '(d b) n -> b n d', d=d)

    min_val = torch.min(softrank, dim=1, keepdim=True).values
    max_val = torch.max(softrank, dim=1, keepdim=True).values
    scaled_softrank = ((softrank - min_val) / (max_val - min_val)) / gauss_range + (0.5 - 0.5 / gauss_range)

    if gauss_copula:
        normal_dist = torch.distributions.Normal(0, 1)
        scaled_softrank = scaled_softrank.permute(0, 2, 1)
        scaled_softrank = normal_dist.icdf(scaled_softrank)
        scaled_softrank = scaled_softrank.permute(0, 2, 1)

    return scaled_softrank


def softrank_preprocessing_correct(
    input_tensor,
    regularization_strength=0.1,
    gauss_copula=True,
    gauss_range=1.25,
    eps=1e-6,
):
    b, n, d = input_tensor.shape
    scaled_tensor = input_tensor

    # First soft-rank
    softrank = torchsort.soft_rank(
        rearrange(scaled_tensor, "b n d -> (d b) n"),
        regularization_strength=regularization_strength,
    )
    softrank = rearrange(softrank, "(d b) n -> b n d", d=d)

    # Second soft-rank
    softrank = torchsort.soft_rank(
        rearrange(softrank, "b n d -> (d b) n"),
        regularization_strength=regularization_strength,
    )
    softrank = rearrange(softrank, "(d b) n -> b n d", d=d)

    min_val = torch.min(softrank, dim=1, keepdim=True).values
    max_val = torch.max(softrank, dim=1, keepdim=True).values
    denom = (max_val - min_val).clamp_min(eps)

    scaled = (softrank - min_val) / denom
    scaled = scaled / gauss_range + (0.5 - 0.5 / gauss_range)

    if gauss_copula:
        scaled = scaled.clamp(min=eps, max=1.0 - eps)
        normal_dist = torch.distributions.Normal(0.0, 1.0)
        scaled = scaled.permute(0, 2, 1)
        scaled = normal_dist.icdf(scaled)
        scaled = scaled.permute(0, 2, 1)

    return scaled


def softrank_preprocessing_correct_gpu(
    input_tensor,
    regularization_strength=0.1,
    gauss_copula=True,
    gauss_range=1.25,
    eps=1e-6,
):
    """
    Identical semantics to softrank_preprocessing_correct, but enforces that the
    whole pipeline (including torchsort) stays on the input tensor's device.
    Expects ``input_tensor`` already on CUDA; works in fp32 for numerical parity.
    """
    if not input_tensor.is_cuda:
        raise ValueError(
            f"softrank_preprocessing_correct_gpu requires a CUDA tensor, got device {input_tensor.device}"
        )

    x = input_tensor.contiguous().float()
    b, n, d = x.shape

    flat = rearrange(x, "b n d -> (d b) n")
    softrank = torchsort.soft_rank(flat, regularization_strength=regularization_strength)
    softrank = torchsort.soft_rank(softrank, regularization_strength=regularization_strength)
    softrank = rearrange(softrank, "(d b) n -> b n d", d=d)

    min_val = torch.min(softrank, dim=1, keepdim=True).values
    max_val = torch.max(softrank, dim=1, keepdim=True).values
    denom = (max_val - min_val).clamp_min(eps)

    scaled = (softrank - min_val) / denom
    scaled = scaled / gauss_range + (0.5 - 0.5 / gauss_range)

    if gauss_copula:
        scaled = scaled.clamp(min=eps, max=1.0 - eps)
        normal_dist = torch.distributions.Normal(0.0, 1.0)
        scaled = scaled.permute(0, 2, 1)
        scaled = normal_dist.icdf(scaled)
        scaled = scaled.permute(0, 2, 1)

    return scaled


# ============================================================
# Per-side whitening — second preprocessing layer (config: whiten=eig)
# ============================================================

def whiten_blocks(data, max_dim, eps_floor=1e-3):
    """
    Decorrelate the X block and the Y block SEPARATELY, per task (ZCA whitening).

    ``data`` : [b, n, 2*max_dim], X = data[..., :max_dim], Y = data[..., max_dim:2*max_dim],
    already soft-rank Gaussian-copula preprocessed (so marginals are ~N(0,1)).

    Each block Z is replaced by  (Z - mean) @ W,  with  W = V diag(s^{-1/2}) Vᵀ
    built from THAT task's own empirical covariance and eigenvalues floored at
    ``s = max(lambda, eps_floor * lambda_max)``. Consequences:

      * W is symmetric positive-definite -> an INVERTIBLE linear map on each side,
        so I(X;Y) is preserved EXACTLY (MI is invariant under a separate bijection
        applied to X and to Y).
      * cond(W) <= (1/eps_floor)^{1/2}; with eps_floor=1e-3 that is ~32, so the map
        is numerically well-conditioned and reversible even on (near-)rank-deficient
        tasks -- no eigenvalue blow-up, no precision loss.

    Applied identically at training and inference time (see ``train.py`` and
    ``infer.py``); the transform must match on both sides. Done per block:
    whitening the *stacked* [X,Y] jointly would mix the two sides and destroy the
    cross-dependence (MI -> ~0).
    """
    if max_dim <= 0:
        return data
    orig_dtype = data.dtype
    n = data.shape[1]

    def _whiten(Z):                                       # Z: [b, n, m]  (float64)
        m = Z.shape[-1]
        Zc = Z - Z.mean(dim=1, keepdim=True)
        cov = torch.einsum("bni,bnj->bij", Zc, Zc) / max(n - 1, 1)   # [b, m, m]
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        cov = cov + 1e-9 * torch.eye(m, dtype=cov.dtype, device=cov.device)
        evals, evecs = torch.linalg.eigh(cov)                        # ascending
        lam_max = evals[..., -1:].clamp_min(1e-12)
        evals = torch.maximum(evals, eps_floor * lam_max)            # floor cond
        inv_sqrt = evals.rsqrt()
        W = (evecs * inv_sqrt.unsqueeze(-2)) @ evecs.transpose(-1, -2)
        return torch.einsum("bni,bij->bnj", Zc, W)

    work = data.double()
    out = work.clone()
    out[..., :max_dim] = _whiten(work[..., :max_dim])
    out[..., max_dim:2 * max_dim] = _whiten(work[..., max_dim:2 * max_dim])
    return out.to(orig_dtype)


# ============================================================
# Gaussian-noise padding to a fixed input dimension
# ============================================================

def gauss_noise_padding(batch, aim_dim=32, perm=True):
    current_dim = batch.shape[2]
    batchsize = batch.shape[0]
    seq_len = batch.shape[1]

    if current_dim > aim_dim:
        raise ValueError("current dimension is larger than the padding dimension!")
    elif current_dim == aim_dim:
        return batch

    padding_dim = aim_dim - current_dim
    noise = torch.randn(batchsize, seq_len, padding_dim)
    padded_batch = torch.cat((batch, noise), dim=2)

    if perm:
        re_setdim = torch.randperm(aim_dim).to(device)
        padded_batch = torch.index_select(padded_batch, 2, re_setdim)

    padded_batch = scale_data_pytorch(padded_batch)
    return padded_batch


def gauss_noise_padding_gpu(batch, aim_dim=32, perm=True):
    """
    Device-preserving variant of gauss_noise_padding. Creates the noise tensor on
    the same device as ``batch`` so the whole op stays on GPU.
    """
    current_dim = batch.shape[2]
    batchsize = batch.shape[0]
    seq_len = batch.shape[1]
    in_device = batch.device

    if current_dim > aim_dim:
        raise ValueError("current dimension is larger than the padding dimension!")
    elif current_dim == aim_dim:
        return batch

    padding_dim = aim_dim - current_dim
    noise = torch.randn(batchsize, seq_len, padding_dim, device=in_device, dtype=batch.dtype)
    padded_batch = torch.cat((batch, noise), dim=2)

    if perm:
        re_setdim = torch.randperm(aim_dim, device=in_device)
        padded_batch = torch.index_select(padded_batch, 2, re_setdim)

    padded_batch = scale_data_pytorch_gpu(padded_batch)
    return padded_batch
