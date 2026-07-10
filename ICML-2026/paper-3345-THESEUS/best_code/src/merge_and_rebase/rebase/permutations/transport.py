"""
High-level permutation transport for visual state dicts.

Provides apply_visual_permutation_to_state() used by vision_rebase.py
for the permutation sanity check.
"""
from __future__ import annotations

from typing import Any, Dict

import torch

from .spec import CLIP_Visual_PermutationSpecBuilder
from .utils import apply_permutation_to_statedict


def apply_visual_permutation_to_state(
    *,
    state: Dict[str, torch.Tensor],
    perm_indices: Dict[str, torch.Tensor],
    heads_indices: Dict[str, Dict[str, torch.Tensor]] | None,
    prefix: str,
    depth: int,
    num_heads: int,
    split_qkv: bool = True,
    spec_variant: str | None = None,
    reference: Dict[str, torch.Tensor] | None = None,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Apply pre-computed permutations to a visual state dict.

    Args:
        state: State dict to permute (visual-only keys, no "visual." prefix).
        perm_indices: Permutation indices from WeightMatcher.run().
        heads_indices: Per-head permutation indices (or None).
        prefix: Prefix for the permutation spec ("" for visual-only).
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        split_qkv: Whether QKV is split (True for TransFusion-patched models).
        spec_variant: Unused, kept for API compatibility.
        reference: Reference state dict for key filtering (unused, kept for API compatibility).
        device: Device to run permutation on.

    Returns:
        Permuted state dict.
    """
    ps = CLIP_Visual_PermutationSpecBuilder(depth=depth, prefix=prefix).create_permutation_spec()

    perm_device = {k: v.to(device) for k, v in perm_indices.items()}
    heads_device = None
    if heads_indices is not None:
        heads_device = {
            k: {hk: hv.to(device) for hk, hv in v.items()}
            for k, v in heads_indices.items()
        }

    state_cuda = {k: v.to(device) for k, v in state.items()}

    permuted = apply_permutation_to_statedict(
        ps=ps,
        perm_matrices=perm_device,
        model_a_dict=state_cuda,
        heads_permutation=heads_device,
        skip_params=False,
        num_heads=num_heads,
    )

    return {k: v.cpu() for k, v in permuted.items()}
