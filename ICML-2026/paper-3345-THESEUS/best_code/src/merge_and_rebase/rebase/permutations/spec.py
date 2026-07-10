"""
Permutation specification for CLIP visual transformer.

Stripped-down version of TransFusion/permutations/permutation_spec.py containing
only the classes needed for CLIP visual weight matching.
"""
from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple


class PermutationSpec(NamedTuple):
    """Maps permutation names to the layers/axes they permute, and vice versa."""

    perm_to_layers_and_axes: dict
    layer_and_axes_to_perm: dict


class PermutationSpecBuilder:
    """Base class for building permutation specifications."""

    def create_permutation_spec(self) -> PermutationSpec:
        raise NotImplementedError

    def permutation_spec_from_axes_to_perm(
        self, axes_to_perm: dict
    ) -> PermutationSpec:
        """Convert a mapping from axes to permutations into a PermutationSpec."""
        perm_to_axes = defaultdict(list)
        for wk, axis_perms in axes_to_perm.items():
            for axis, perm in enumerate(axis_perms):
                if perm is not None:
                    perm_to_axes[perm].append((wk, axis))
        return PermutationSpec(
            perm_to_layers_and_axes=dict(perm_to_axes),
            layer_and_axes_to_perm=axes_to_perm,
        )


def conv_axes(name: str, in_perm, out_perm, bias: bool = False) -> dict:
    """Specify permutation axes for convolutional layers."""
    axes = {f"{name}.weight": (out_perm, in_perm, None, None)}
    if bias:
        axes[f"{name}.bias"] = (out_perm,)
    return axes


def layernorm_axes(name: str, perm) -> dict:
    """Specify permutation axes for LayerNorm layers."""
    return {f"{name}.weight": (perm,), f"{name}.bias": (perm,)}


def transformer_block_axes_clip(
    depth: int, p_in: str, p_out: str, prefix: str = ""
) -> dict:
    """Specify permutation axes for CLIP transformer blocks."""
    all_axes: dict = {}

    for block_ind in range(depth):
        block_out = p_out if block_ind == depth - 1 else f"P{block_ind}_out"
        block_in = p_in if block_ind == 0 else f"P{block_ind - 1}_out"

        block_axes = {
            f"{prefix}transformer.resblocks.{block_ind}.ln_1.weight": (block_in,),
            f"{prefix}transformer.resblocks.{block_ind}.ln_1.bias": (block_in,),
            f"{prefix}transformer.resblocks.{block_ind}.attn.q.weight": (
                f"P{block_ind}_attn_QK",
                block_in,
            ),
            f"{prefix}transformer.resblocks.{block_ind}.attn.k.weight": (
                f"P{block_ind}_attn_QK",
                block_in,
            ),
            f"{prefix}transformer.resblocks.{block_ind}.attn.v.weight": (
                f"P{block_ind}_attn_QK",
                block_in,
            ),
            f"{prefix}transformer.resblocks.{block_ind}.attn.q.bias": (
                f"P{block_ind}_attn_QK",
            ),
            f"{prefix}transformer.resblocks.{block_ind}.attn.k.bias": (
                f"P{block_ind}_attn_QK",
            ),
            f"{prefix}transformer.resblocks.{block_ind}.attn.v.bias": (
                f"P{block_ind}_attn_QK",
            ),
            f"{prefix}transformer.resblocks.{block_ind}.attn.proj.weight": (
                f"P{block_ind}_out_proj",
                f"P{block_ind}_attn_QK",
            ),
            f"{prefix}transformer.resblocks.{block_ind}.attn.proj.bias": (
                f"P{block_ind}_out_proj",
            ),
            f"{prefix}transformer.resblocks.{block_ind}.attn.shortcut_1.identity": (
                f"P{block_ind}_out_proj",
                block_in,
            ),
            f"{prefix}transformer.resblocks.{block_ind}.ln_2.weight": (
                f"P{block_ind}_out_proj",
            ),
            f"{prefix}transformer.resblocks.{block_ind}.ln_2.bias": (
                f"P{block_ind}_out_proj",
            ),
            f"{prefix}transformer.resblocks.{block_ind}.mlp.fc1.weight": (
                f"P{block_ind}_mlp_out",
                f"P{block_ind}_out_proj",
            ),
            f"{prefix}transformer.resblocks.{block_ind}.mlp.fc1.bias": (
                f"P{block_ind}_mlp_out",
            ),
            f"{prefix}transformer.resblocks.{block_ind}.mlp.fc2.weight": (
                block_out,
                f"P{block_ind}_mlp_out",
            ),
            f"{prefix}transformer.resblocks.{block_ind}.mlp.fc2.bias": (block_out,),
            f"{prefix}transformer.resblocks.{block_ind}.mlp.shortcut_2.identity": (
                block_out,
                f"P{block_ind}_out_proj",
            ),
        }
        all_axes.update(block_axes)

    return all_axes


class CLIP_Visual_PermutationSpecBuilder(PermutationSpecBuilder):
    """Builder for CLIP visual transformer permutation specifications."""

    def __init__(self, depth: int, prefix: str = "") -> None:
        self.depth = depth
        self.prefix = prefix

    def create_permutation_spec(self) -> PermutationSpec:
        prefix = self.prefix
        axes_to_perm = {
            **conv_axes(f"{prefix}conv1", in_perm=None, out_perm="P_conv"),
            f"{prefix}class_embedding": ("P_conv",),
            f"{prefix}positional_embedding": (None, "P_conv"),
            f"{prefix}ln_pre.weight": ("P_conv",),
            f"{prefix}ln_pre.bias": ("P_conv",),
            **transformer_block_axes_clip(
                self.depth, p_in="P_conv", p_out="P_last", prefix=prefix
            ),
            f"{prefix}ln_post.weight": ("P_last",),
            f"{prefix}ln_post.bias": ("P_last",),
            f"{prefix}proj": ("P_last", None),
        }
        return self.permutation_spec_from_axes_to_perm(axes_to_perm)
