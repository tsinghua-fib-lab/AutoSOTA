from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


def compute_group_means(losses, group_ids, num_groups: int) -> Tuple["object", "object"]:
    """Compute per-group mean losses for a minibatch.

    Returns
    -------
    group_means : (G,) tensor
    group_counts : (G,) tensor

    Notes
    -----
    - Missing groups in the minibatch get mean 0 (and count 0).
    - Callers should be aware: rare groups benefit from group-balanced batching,
      but this function itself stays agnostic.
    """
    try:
        import torch
    except Exception as e:
        raise ImportError("PyTorch is required for compute_group_means.") from e

    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}.")

    l = losses.reshape(-1)
    g = group_ids.reshape(-1).long()
    if l.numel() != g.numel():
        raise ValueError(f"losses and group_ids must have same length, got {l.numel()} vs {g.numel()}.")

    device = l.device
    dtype = l.dtype

    sums = torch.zeros((num_groups,), device=device, dtype=dtype)
    counts = torch.zeros((num_groups,), device=device, dtype=dtype)

    sums.scatter_add_(0, g, l)
    counts.scatter_add_(0, g, torch.ones_like(l, dtype=dtype))

    means = sums / counts.clamp(min=1.0)
    means = torch.where(counts > 0, means, torch.zeros_like(means))
    return means, counts


@dataclass
class GroupDROState:
    """State for GroupDRO (per-group weights q)."""
    num_groups: int
    eta: float = 0.1
    q: "object" = None  # torch.Tensor initialised lazily

    def maybe_init(self, device, dtype):
        try:
            import torch
        except Exception as e:
            raise ImportError("PyTorch is required for GroupDROState.") from e

        if self.q is None:
            self.q = torch.full((self.num_groups,), 1.0 / float(self.num_groups), device=device, dtype=dtype)
        return self


def group_dro_loss(losses, group_ids, state: GroupDROState) -> Tuple["object", GroupDROState]:
    """Compute GroupDRO loss and update q in-place (returned for convenience)."""
    try:
        import torch
    except Exception as e:
        raise ImportError("PyTorch is required for group_dro_loss.") from e

    l = losses.reshape(-1)
    g = group_ids.reshape(-1).long()
    state.maybe_init(device=l.device, dtype=l.dtype)

    group_means, group_counts = compute_group_means(l, g, num_groups=state.num_groups)

    # Exponentiated-gradient update on q using *detached* group losses.
    with torch.no_grad():
        state.q = state.q * torch.exp(float(state.eta) * group_means.detach())
        state.q = state.q / state.q.sum().clamp(min=1e-12)

    robust = torch.sum(state.q * group_means)
    return robust, state
