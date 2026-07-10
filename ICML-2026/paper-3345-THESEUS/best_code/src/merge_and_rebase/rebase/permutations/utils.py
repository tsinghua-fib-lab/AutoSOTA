"""
Permutation utilities for applying permutations to model state dicts.

Stripped-down version of TransFusion/permutations/utils.py containing only
the functions needed for the transport workflow.
"""
from __future__ import annotations

import copy
from typing import Dict, Union

import torch

from .spec import PermutationSpec

PermutationMatrix = torch.Tensor
PermutationIndices = torch.Tensor


def perm_indices_to_perm_matrix(perm_indices: PermutationIndices) -> PermutationMatrix:
    """Convert permutation indices to a permutation matrix."""
    n = len(perm_indices)
    return torch.eye(n, device=perm_indices.device)[perm_indices.long()]


def perm_matrix_to_perm_indices(perm_matrix: PermutationMatrix) -> PermutationIndices:
    """Convert a permutation matrix to permutation indices."""
    return perm_matrix.nonzero()[:, 1].long()


def perm_rows(x: torch.Tensor, perm: PermutationMatrix) -> torch.Tensor:
    """Permute the first axis (rows) of tensor x by permutation matrix perm."""
    assert x.shape[0] == perm.shape[0]
    assert perm.dim() == 2 and perm.shape[0] == perm.shape[1]
    input_dims = "jklm"[: x.dim()]
    output_dims = "iklm"[: x.dim()]
    return torch.einsum(f"ij,{input_dims}->{output_dims}", perm, x)


def perm_cols(x: torch.Tensor, perm: PermutationMatrix) -> torch.Tensor:
    """Permute the second axis (columns) of tensor x by permutation matrix perm."""
    assert x.shape[1] == perm.shape[0]
    x = x.transpose(1, 0)
    perm = perm.transpose(1, 0)
    return perm_rows(x, perm).transpose(1, 0)


def perm_tensor_by_perm_matrix(
    tens: torch.Tensor, perm: PermutationMatrix, axis: int
) -> torch.Tensor:
    """Permute a tensor along the specified axis using a permutation matrix."""
    assert axis == 0 or axis == 1
    if axis == 0:
        return perm_rows(tens, perm)
    return perm_cols(tens, perm.T)


def get_permuted_param(
    param: torch.Tensor,
    perms_to_apply,
    perm_matrices: Dict[str, torch.Tensor],
    except_axis=None,
    num_heads: int = 12,
    all_heads_indices: Dict[str, Dict[str, torch.Tensor]] | None = None,
) -> torch.Tensor:
    """Apply all relevant permutations to a parameter tensor."""
    for axis, perm_id in enumerate(perms_to_apply):
        if axis == except_axis or perm_id is None:
            continue

        perm = perm_matrices[perm_id].cuda()
        if perm.dim() == 1:
            if "attn" in perm_id:
                original_shape = param.shape
                if all_heads_indices is None:
                    if axis == 0:
                        param = param.reshape(num_heads, -1)
                        param = torch.index_select(param, axis, perm.int())
                    elif axis == 1:
                        param = param.T
                        param = param.reshape(
                            num_heads, param.shape[0] // num_heads, param.shape[1]
                        )
                        param = torch.index_select(param, 0, perm)
                        param = param.reshape(original_shape[1], original_shape[0])
                        param = param.T
                    param = param.reshape(original_shape)
                else:
                    heads_perm = all_heads_indices[perm_id]
                    if axis == 0:
                        if len(param.shape) > 1:
                            param = param.reshape(
                                num_heads, original_shape[0] // num_heads, -1
                            )
                            param = torch.index_select(param, axis, perm.int())
                            param = param.transpose(1, 0)
                            for i in range(num_heads):
                                param[:, i] = torch.index_select(
                                    param[:, i],
                                    axis,
                                    heads_perm[f"P_head_{i}"].cuda(),
                                )
                            param = param.transpose(1, 0)
                        else:
                            param = param.reshape(num_heads, -1)
                            param = torch.index_select(param, axis, perm.int())
                            param = param.transpose(1, 0)
                            for i in range(num_heads):
                                param[:, i] = torch.index_select(
                                    param[:, i],
                                    axis,
                                    heads_perm[f"P_head_{i}"].cuda(),
                                )
                            param = param.transpose(1, 0)
                    elif axis == 1:
                        if len(param.shape) > 1:
                            param = param.T
                            param = param.reshape(
                                num_heads,
                                param.shape[0] // num_heads,
                                param.shape[1],
                            )
                            param = torch.index_select(param, 0, perm)
                            param = param.transpose(1, 0)
                            for i in range(num_heads):
                                param[:, i] = torch.index_select(
                                    param[:, i],
                                    0,
                                    heads_perm[f"P_head_{i}"].cuda(),
                                )
                            param = param.transpose(1, 0)
                            param = param.reshape(original_shape[1], original_shape[0])
                            param = param.T
                        else:
                            param = param.T
                            param = param.reshape(num_heads, -1)
                            param = torch.index_select(param, 0, perm)
                            param = param.transpose(1, 0)
                            for i in range(num_heads):
                                param[:, i] = torch.index_select(
                                    param[:, i],
                                    0,
                                    heads_perm[f"P_head_{i}"].cuda(),
                                )
                            param = param.transpose(1, 0)
                            param = param.reshape(original_shape[1], original_shape[0])
                            param = param.T
                    param = param.reshape(original_shape)
            else:
                param = torch.index_select(param, axis, perm.int())
        else:
            param = perm_tensor_by_perm_matrix(param, perm, axis)

    return param


def apply_permutation_to_statedict(
    ps: PermutationSpec,
    perm_matrices: Dict[str, torch.Tensor],
    model_a_dict: Dict[str, torch.Tensor],
    model_b_dict: Dict[str, torch.Tensor] | None = None,
    heads_permutation: Dict[str, Dict[str, torch.Tensor]] | None = None,
    skip_params: bool = False,
    num_heads: int = 12,
) -> Dict[str, torch.Tensor]:
    """Apply a set of permutations to a model's state_dict according to a PermutationSpec."""
    permuted_params: Dict[str, torch.Tensor] = {}

    for param_name, param in model_a_dict.items():
        param_name_in_perm_dict = param_name

        if skip_params:
            if param_name_in_perm_dict not in ps.layer_and_axes_to_perm:
                permuted_params[param_name] = param
                continue
        else:
            assert (
                param_name_in_perm_dict in ps.layer_and_axes_to_perm
            ), f"param_name {param_name} not found in ps.layer_and_axes_to_perm"

        try:
            param = copy.deepcopy(param)
            perms_to_apply = ps.layer_and_axes_to_perm[param_name_in_perm_dict]
            param = get_permuted_param(
                param,
                perms_to_apply,
                perm_matrices,
                all_heads_indices=heads_permutation,
                num_heads=num_heads,
            )
            permuted_params[param_name] = param
        except Exception:
            print(
                f"Problem during application of permutation {perms_to_apply} on layer {param_name}"
            )

    return permuted_params
