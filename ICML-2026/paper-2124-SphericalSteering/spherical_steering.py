"""
Spherical Steering: Core Algorithm

This module implements the Spherical Steering intervention method for
steering language model hidden states toward truthful directions using
von Mises-Fisher (vMF) distributions on the unit sphere.

Key Components:
- spherical_geometric_logic: Core steering logic for single vectors
- baukit_hook_fn: Hook function compatible with baukit TraceDict
- get_spherical_intervention: Factory function for creating steering hooks
"""

import torch
import torch.nn.functional as F
from functools import partial


def spherical_geometric_logic(x, mu_T, mu_H, kappa, alpha, beta):
    """
    Core spherical steering logic for a single hidden state vector.

    This function implements the geometric steering operation:
    1. Compute vMF probabilities for truthful (T) and hallucination (H) prototypes
    2. If hallucination probability exceeds threshold, apply steering
    3. Steering rotates the vector toward the truthful prototype on the sphere

    Args:
        x: Hidden state vector [D]
        mu_T: Truthful prototype (unit vector) [D]
        mu_H: Hallucination prototype (unit vector) [D]
        kappa: vMF concentration parameter (higher = sharper decisions)
        alpha: Maximum steering strength (0 to 1)
        beta: Threshold for triggering steering (p_H - p_T > beta)

    Returns:
        x_new: Steered hidden state vector [D]
        triggered: Boolean indicating if steering was applied
    """
    orig_dtype = x.dtype
    x = x.float()
    mu_T = mu_T.float()
    mu_H = mu_H.float()

    # Preserve original norm for rescaling
    orig_norm = x.norm(p=2).clamp_min(1e-12)
    x_hat = x / orig_norm

    # Compute vMF log-likelihoods (proportional to cosine similarity)
    cos_T = torch.dot(x_hat, mu_T).clamp(-1, 1)
    cos_H = torch.dot(x_hat, mu_H).clamp(-1, 1)

    # Softmax to get probabilities
    logits = torch.stack([kappa * cos_T, kappa * cos_H])
    probs = F.softmax(logits, dim=0)
    p_T, p_H = probs[0], probs[1]

    # Check steering condition
    delta = p_H - p_T

    if delta <= beta:
        # No steering needed
        return x.to(orig_dtype), False

    # Compute steering strength (linear interpolation above threshold)
    t = alpha * (delta - beta) / (1.0 - beta)
    t = torch.clamp(t, 0.0, 1.0)

    # Compute angle from truthful prototype
    theta = torch.acos(cos_T)
    # Increased threshold for numerical stability (was 1e-4)
    if theta < 1e-3:
        # Already very close to mu_T
        return x.to(orig_dtype), False

    # Handle theta near pi (x_hat almost opposite to mu_T)
    if cos_T < -0.999:
        # Use a fixed reference direction to construct perpendicular vector
        angle_rot = t * (torch.pi / 2.0)
        ref = torch.zeros_like(mu_T)
        ref[0] = 1.0
        if torch.abs(torch.dot(ref, mu_T)) > 0.999:
            ref[0] = 0.0
            ref[1] = 1.0
        u_perp = ref - torch.dot(ref, mu_T) * mu_T
        u_perp = u_perp / u_perp.norm(p=2).clamp_min(1e-12)
        x_new_hat = torch.cos(angle_rot) * mu_T + torch.sin(angle_rot) * u_perp
    else:
        # Compute new angle (rotate toward mu_T)
        theta_new = (1.0 - t) * theta

        # Spherical interpolation (SLERP-like)
        sin_theta = torch.sin(theta).clamp_min(1e-8)
        u = (x_hat - cos_T * mu_T) / sin_theta  # Orthogonal component

        x_new_hat = torch.cos(theta_new) * mu_T + torch.sin(theta_new) * u

    # Ensure unit norm after interpolation
    x_new_hat = x_new_hat / x_new_hat.norm(p=2).clamp_min(1e-12)
    x_new = x_new_hat * orig_norm  # Restore original norm

    # Safety: replace NaN/Inf in output
    x_new = torch.nan_to_num(x_new, nan=0.0, posinf=0.0, neginf=0.0)

    return x_new.to(orig_dtype), True


def baukit_hook_fn(output, layer_name, mu_T, mu_H, kappa, alpha, beta,
                   stats=None, start_idx=None,
                   norm_mean=None, norm_std=None):
    """
    Hook function for use with baukit TraceDict.

    This function modifies hidden states in-place during forward pass.
    It supports range-based intervention for scoring tasks.

    Args:
        output: Layer output (tuple or tensor) from transformer layer
        layer_name: Name of the layer being hooked (unused, required by baukit)
        mu_T: Truthful prototype [D]
        mu_H: Hallucination prototype [D]
        kappa: vMF concentration parameter
        alpha: Maximum steering strength
        beta: Steering threshold
        stats: Optional dict to track steering statistics
        start_idx: Starting position for intervention (None = last token only)
        norm_mean: Optional per-dimension mean for z-score normalization [D]
        norm_std: Optional per-dimension std for z-score normalization [D]

    Returns:
        Modified output with steered hidden states
    """
    if isinstance(output, tuple):
        h_hidden = output[0]  # [Batch, Seq, Dim]
    else:
        h_hidden = output

    device = h_hidden.device

    # Ensure prototypes are on the correct device
    if not isinstance(mu_T, torch.Tensor):
        mu_T = torch.tensor(mu_T, device=device)
    else:
        mu_T = mu_T.to(device)
    if not isinstance(mu_H, torch.Tensor):
        mu_H = torch.tensor(mu_H, device=device)
    else:
        mu_H = mu_H.to(device)

    # Ensure normalization params are on the correct device
    has_norm = norm_mean is not None and norm_std is not None
    if has_norm:
        if not isinstance(norm_mean, torch.Tensor):
            norm_mean = torch.tensor(norm_mean, device=device)
        else:
            norm_mean = norm_mean.to(device)
        if not isinstance(norm_std, torch.Tensor):
            norm_std = torch.tensor(norm_std, device=device)
        else:
            norm_std = norm_std.to(device)

    batch_size, seq_len, _ = h_hidden.shape

    # Determine intervention range
    if start_idx is None:
        # Default: only last token (for generation)
        range_to_steer = [seq_len - 1]
    else:
        # Range-based intervention (for MC scoring)
        safe_start = max(0, min(start_idx, seq_len - 1))
        range_to_steer = range(safe_start, seq_len)

    # Apply steering to each position in range
    for i in range(batch_size):
        for t in range_to_steer:
            vec = h_hidden[i, t, :].clone()

            # Optional z-score normalization before steering
            if has_norm:
                vec = (vec - norm_mean) / (norm_std + 1e-8)

            modified_vec, triggered = spherical_geometric_logic(
                vec, mu_T, mu_H, kappa, alpha, beta
            )

            # Denormalize if normalization was applied
            if has_norm:
                modified_vec = modified_vec * norm_std + norm_mean

            h_hidden[i, t, :] = modified_vec

            if stats is not None:
                stats['total'] += 1
                if triggered:
                    stats['steered'] += 1

    if isinstance(output, tuple):
        return (h_hidden,) + output[1:]
    else:
        return h_hidden


def get_spherical_intervention(mu_T, mu_H, kappa=20.0, alpha=0.15, beta=0.1,
                                stats=None, norm_mean=None, norm_std=None):
    """
    Factory function to create a spherical steering hook.

    Returns a partially-applied hook function that can be used with baukit.
    The returned function accepts an additional `start_idx` parameter for
    range-based intervention.

    Args:
        mu_T: Truthful prototype [D]
        mu_H: Hallucination prototype [D]
        kappa: vMF concentration parameter (default: 20.0)
        alpha: Maximum steering strength (default: 0.15)
        beta: Steering threshold (default: 0.1)
        stats: Optional dict to track steering statistics
        norm_mean: Optional per-dimension mean for z-score normalization [D]
        norm_std: Optional per-dimension std for z-score normalization [D]

    Returns:
        Hook function compatible with baukit TraceDict

    Example:
        >>> hook_fn = get_spherical_intervention(mu_T, mu_H, kappa=20, alpha=0.6, beta=-0.05)
        >>> with TraceDict(model, [layer_name], edit_output=partial(hook_fn, start_idx=0)):
        ...     outputs = model(input_ids)
    """
    return partial(
        baukit_hook_fn,
        mu_T=mu_T,
        mu_H=mu_H,
        kappa=kappa,
        alpha=alpha,
        beta=beta,
        stats=stats,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
