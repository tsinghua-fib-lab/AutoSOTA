from __future__ import annotations

from collections.abc import Sequence

import torch

from ..utils.geometry import orthonormal_basis


def orth_from_factor(x: torch.Tensor, *, side: str) -> torch.Tensor:
    if side == "B":
        q = orthonormal_basis(x)
    elif side == "A":
        q = orthonormal_basis(x.T)
    else:
        raise ValueError(f"Unknown side: {side}")
    return q.contiguous()


def grassmann_log_map(reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    reference64 = reference.to(dtype=torch.float64)
    target64 = target.to(dtype=torch.float64)

    overlap = reference64.T @ target64
    residual = target64 - reference64 @ overlap

    if torch.count_nonzero(residual).item() == 0:
        return torch.zeros_like(reference64)

    tangent_operator = residual @ torch.linalg.pinv(overlap)
    q, s, vh = torch.linalg.svd(tangent_operator, full_matrices=False)
    theta = torch.arctan(s)
    xi = (q * theta.unsqueeze(0)) @ vh
    xi = xi - reference64 @ (reference64.T @ xi)
    return xi.contiguous()


def grassmann_exp_map(reference: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    reference64 = reference.to(dtype=torch.float64)
    tangent64 = tangent.to(dtype=torch.float64)

    if torch.count_nonzero(tangent64).item() == 0:
        return reference64.contiguous()

    q, s, vh = torch.linalg.svd(tangent64, full_matrices=False)
    v = vh.T
    cos_s = torch.cos(s)
    sin_s = torch.sin(s)

    ref_term = reference64 @ ((v * cos_s.unsqueeze(0)) @ v.T)
    tan_term = (q * sin_s.unsqueeze(0)) @ vh
    merged = ref_term + tan_term
    return torch.linalg.qr(merged, mode="reduced")[0].contiguous()


def geodesic_interpolate(reference: torch.Tensor, target: torch.Tensor, eta: float) -> torch.Tensor:
    if eta <= 0.0:
        return reference.to(dtype=torch.float64).contiguous()
    if eta >= 1.0:
        return target.to(dtype=torch.float64).contiguous()
    tangent = grassmann_log_map(reference, target)
    return grassmann_exp_map(reference, float(eta) * tangent)


def resolve_geodesic_mean_weights(
    *,
    num_bases: int,
    weights: Sequence[float] | None,
    weighting: str,
    context_name: str,
) -> list[float]:
    if weighting == "equal":
        return [1.0] * num_bases
    if weights is None:
        return [1.0] * num_bases

    resolved = [float(weight) for weight in weights]
    if len(resolved) != num_bases:
        raise ValueError(
            f"Resolved merge weights must match the number of tasks for {context_name} "
            f"(got {len(resolved)} weights for {num_bases} tasks)."
        )
    if any(weight < 0.0 for weight in resolved):
        raise ValueError(f"{context_name} with merge-weighted geometry requires non-negative weights.")
    return resolved


def resolve_nonnegative_weights(
    *,
    num_bases: int,
    weights: Sequence[float] | None,
    context_name: str,
) -> list[float]:
    resolved = [1.0] * num_bases if weights is None else [float(weight) for weight in weights]
    if len(resolved) != num_bases:
        raise ValueError(
            f"Resolved merge weights must match the number of tasks for {context_name} "
            f"(got {len(resolved)} weights for {num_bases} tasks)."
        )
    if any(weight < 0.0 for weight in resolved):
        raise ValueError(f"{context_name} requires non-negative weights.")
    if sum(resolved) <= 0.0:
        raise ValueError(f"{context_name} requires a positive total weight.")
    return resolved


def incremental_grassmann_mean(
    bases: list[torch.Tensor],
    *,
    weighting: str,
    weights: Sequence[float] | None,
    context_name: str,
) -> torch.Tensor:
    if not bases:
        raise ValueError("Expected at least one basis.")

    rank = int(bases[0].shape[1])
    for basis in bases[1:]:
        if int(basis.shape[1]) != rank:
            raise ValueError(f"{context_name} requires the same LoRA rank across tasks for each layer.")

    resolved_weights = resolve_geodesic_mean_weights(
        num_bases=len(bases),
        weights=weights,
        weighting=weighting,
        context_name=context_name,
    )

    mean = bases[0].to(dtype=torch.float64).contiguous()
    cumulative = float(resolved_weights[0]) if weighting == "merge_weights" else 1.0
    if weighting == "merge_weights" and cumulative <= 0.0:
        raise ValueError(f"{context_name} with merge-weighted geometry requires a positive first weight.")

    for idx in range(1, len(bases)):
        target = bases[idx]
        if weighting == "merge_weights":
            weight = float(resolved_weights[idx])
            cumulative += weight
            if cumulative <= 0.0:
                raise ValueError(f"{context_name} with merge-weighted geometry requires positive cumulative weight.")
            eta = weight / cumulative
        else:
            eta = 1.0 / float(idx + 1)
        mean = geodesic_interpolate(mean, target, eta)

    return torch.linalg.qr(mean, mode="reduced")[0].contiguous()
