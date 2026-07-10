"""
WeightMatcher for permutation-based model alignment.

Stripped-down version of TransFusion/permutations/weights_matcher.py containing
only WeightMatcher and its direct dependencies.
"""
from __future__ import annotations

import sys
from enum import auto
from typing import Dict, Tuple, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from tqdm import tqdm

from .spec import PermutationSpec

eps = 1e-12


class LayerIterationOrder:
    """Order in which layers are iterated during weight matching."""

    RANDOM = "random"
    FORWARD = "forward"
    BACKWARD = "backward"
    ALTERNATE = "alternate"


def alternate_layers(num_layers: int) -> list[int]:
    """Alternating order: start, end, next, prev, ..."""
    all_layers = list(range(num_layers))
    result = []
    for i in range((num_layers + 1) // 2):
        result.append(all_layers[i])
        if i != num_layers - i - 1:
            result.append(all_layers[-i - 1])
    return result


def get_layer_iteration_order(
    order: str, num_layers: int
) -> Union[torch.Tensor, range, list]:
    if order == LayerIterationOrder.RANDOM:
        return torch.randperm(num_layers)
    elif order == LayerIterationOrder.FORWARD:
        return torch.arange(num_layers)
    elif order == LayerIterationOrder.BACKWARD:
        return range(num_layers)[num_layers:0:-1]
    elif order == LayerIterationOrder.ALTERNATE:
        return alternate_layers(num_layers)
    else:
        raise NotImplementedError(f"Unknown layer iteration order {order}")


def singular_values_norm_multihead(
    a: torch.Tensor, b: torch.Tensor, k: int = 10
) -> torch.Tensor:
    """Norm of difference between top-k singular values of each head."""
    svd_vals_a = torch.linalg.svdvals(a)[:, :k]
    svd_vals_b = torch.linalg.svdvals(b)[:, :k]
    diff_sings = svd_vals_a.unsqueeze(1) - svd_vals_b.unsqueeze(0)
    return torch.norm(diff_sings, dim=-1)


def compute_weights_similarity(
    similarity_matrix: torch.Tensor, perm_indices: torch.Tensor
) -> torch.Tensor:
    """Total similarity given a similarity matrix and permutation indices."""
    n = len(perm_indices)
    return torch.sum(
        similarity_matrix[torch.arange(n), perm_indices.long()]
    )


def solve_linear_assignment_problem(
    sim_matrix: Union[torch.Tensor, np.ndarray],
    return_matrix: bool = False,
    maximize: bool = True,
) -> torch.Tensor:
    """Solve the linear assignment problem (Hungarian algorithm)."""
    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.cpu().detach().numpy()

    ri, ci = linear_sum_assignment(sim_matrix, maximize)
    assert (torch.tensor(ri) == torch.arange(len(ri))).all()

    indices = torch.tensor(ci)
    return indices


class WeightMatcher:
    """
    Aligns model weights by solving assignment problems for each permutation group.

    Args:
        ps: Permutation specification.
        fixed: State dict of the reference (fixed) model.
        permutee: State dict of the model to be permuted.
        max_iter: Maximum number of matching iterations.
        init_perm: Initial permutation indices (default: identity).
        layer_iteration_order: Order to iterate layers.
        p_trim: Quantile threshold for pruning small weights.
        num_heads: Number of attention heads.
        intra_head: Whether to perform intra-head matching.
        normalize_weights: Whether to normalize weights before computing similarity.
    """

    def __init__(
        self,
        ps: PermutationSpec,
        fixed: Dict[str, Tensor],
        permutee: Dict[str, Tensor],
        max_iter: int = 100,
        init_perm: Dict[str, Tensor] | None = None,
        layer_iteration_order: str = LayerIterationOrder.FORWARD,
        p_trim: float | None = None,
        num_heads: int | None = None,
        intra_head: bool = False,
        normalize_weights: bool = False,
    ):
        self.ps = ps
        self.params_a = fixed
        self.params_b = permutee
        self.max_iter = max_iter
        self.init_perm = init_perm
        self.layer_iteration_order = layer_iteration_order
        self.p_trim = p_trim
        self.num_heads = num_heads
        self.intra_head = intra_head
        self.normalize_weights = normalize_weights

        self.perm_sizes = {
            p: self.params_a[ref_tuple[0]].shape[ref_tuple[1]]
            for p, params_and_axes in ps.perm_to_layers_and_axes.items()
            for ref_tuple in [params_and_axes[0]]
        }
        self.all_perm_indices = self._initialize_perm_indices()
        self.all_heads_indices = self._initialize_head_indices()

        self.perm_names = list(self.all_perm_indices.keys())
        self.num_layers = len(self.perm_names)

    def _initialize_perm_indices(self) -> Dict[str, Tensor]:
        if self.init_perm is not None:
            return self.init_perm
        if self.num_heads is not None:
            return {
                p: torch.arange(n)
                if "attn" not in p
                else torch.arange(n // (n // self.num_heads))
                for p, n in self.perm_sizes.items()
            }
        return {p: torch.arange(n) for p, n in self.perm_sizes.items()}

    def _initialize_head_indices(self) -> Dict[str, Dict[str, Tensor]] | None:
        if self.intra_head:
            return {
                key: {
                    f"P_head_{idx}": torch.arange(n // self.num_heads)
                    for idx in range(self.num_heads)
                }
                for key, n in zip(
                    self.all_perm_indices.keys(), self.perm_sizes.values()
                )
                if "attn" in key
            }
        return None

    def _initialize_similarity_matrices(
        self, p: str, intra_head_phase: bool
    ) -> Tuple[int, Union[Tensor, Dict[str, Tensor]]]:
        if "attn" in p and self.num_heads is not None:
            if intra_head_phase:
                num_neurons = self.perm_sizes[p] // self.num_heads
                return num_neurons, {
                    f"sim_matrix_{idx}": torch.zeros(
                        (num_neurons, num_neurons)
                    ).cuda()
                    for idx in range(self.num_heads)
                }
            else:
                num_neurons = self.perm_sizes[p] // (
                    self.perm_sizes[p] // self.num_heads
                )
                return num_neurons, torch.zeros((num_neurons, num_neurons)).cuda()
        else:
            num_neurons = self.perm_sizes[p]
            return num_neurons, torch.zeros((num_neurons, num_neurons)).cuda()

    def _compute_extra_head_similarity(
        self,
        w_a: Tensor,
        w_b: Tensor,
        params_name: str,
        p: str,
        sim_matrix: Tensor,
    ):
        if self.normalize_weights:
            norms_a = torch.norm(w_a, dim=(1, 2))
            mean_norm_heads_a = norms_a.mean()
            norms_b = torch.norm(w_b, dim=(1, 2))
            normalized_heads_b = w_b / norms_b.view(self.num_heads, 1, 1)
            normalized_heads_b = normalized_heads_b * mean_norm_heads_a
            normalized_heads_a = (
                w_a / norms_a.view(self.num_heads, 1, 1)
            ) * mean_norm_heads_a
            w_a = normalized_heads_a
            w_b = normalized_heads_b

        sim_matrix += singular_values_norm_multihead(w_a, w_b, k=w_a.shape[1])

    def _compute_intra_head_similarity(
        self,
        w_a: Tensor,
        w_b: Tensor,
        p: str,
        sim_matrix: Dict[str, Tensor],
    ):
        for i in range(self.num_heads):
            sim_matrix[f"sim_matrix_{i}"] += w_a[i] @ w_b[
                self.all_perm_indices[p][i]
            ].T

    def _process_attention_layer(
        self,
        w_a: Tensor,
        w_b: Tensor,
        params_name: str,
        p: str,
        sim_matrix: Union[Tensor, Dict[str, Tensor]],
        intra_head_phase: bool,
    ):
        if "bias" in params_name:
            return
        head_dim = w_a.shape[1] // self.num_heads
        w_a = w_a.reshape(self.num_heads, head_dim, -1)
        w_b = w_b.reshape(self.num_heads, head_dim, -1)

        if not intra_head_phase:
            self._compute_extra_head_similarity(
                w_a, w_b, params_name, p, sim_matrix
            )
        elif intra_head_phase and self.intra_head:
            self._compute_intra_head_similarity(w_a, w_b, p, sim_matrix)

    def _process_non_attention_layer(
        self, w_a: Tensor, w_b: Tensor, num_neurons: int, sim_matrix: Tensor
    ):
        w_a, w_b = w_a.reshape(num_neurons, -1), w_b.reshape(num_neurons, -1)
        if self.normalize_weights:
            w_a = w_a / torch.norm(w_a, dim=1, keepdim=True)
            w_b = w_b / torch.norm(w_b, dim=1, keepdim=True)
        sim_matrix += w_a @ w_b.T

    def _update_attention_perm_indices(
        self,
        p: str,
        sim_matrix: Union[Tensor, Dict[str, Tensor]],
        intra_head_phase: bool,
    ) -> bool:
        if not intra_head_phase:
            perm_indices = solve_linear_assignment_problem(
                sim_matrix, maximize=False
            )
            old_sim = compute_weights_similarity(
                sim_matrix, self.all_perm_indices[p]
            )
            self.all_perm_indices[p] = perm_indices
            new_sim = compute_weights_similarity(sim_matrix, perm_indices)
            return new_sim < old_sim - eps
        elif intra_head_phase and self.intra_head:
            intra_progress = []
            for i in range(self.num_heads):
                head_sim_matrix = sim_matrix[f"sim_matrix_{i}"]
                perm_indices = solve_linear_assignment_problem(
                    head_sim_matrix, maximize=True
                )
                old_sim = compute_weights_similarity(
                    head_sim_matrix, self.all_heads_indices[p][f"P_head_{i}"]
                )
                self.all_heads_indices[p][f"P_head_{i}"] = perm_indices
                new_sim = compute_weights_similarity(
                    head_sim_matrix, perm_indices
                )
                intra_progress.append(new_sim > (old_sim + eps))
            return sum(intra_progress) > 0

    def _update_non_attention_perm_indices(
        self, p: str, sim_matrix: Tensor
    ) -> bool:
        perm_indices = solve_linear_assignment_problem(sim_matrix, maximize=True)
        old_sim = compute_weights_similarity(
            sim_matrix, self.all_perm_indices[p]
        )
        self.all_perm_indices[p] = perm_indices
        new_sim = compute_weights_similarity(sim_matrix, perm_indices)
        return new_sim > (old_sim + eps)

    def run(
        self,
    ) -> Tuple[
        Dict[str, Tensor], Union[Dict[str, Dict[str, Tensor]], None]
    ]:
        """Run weight matching until convergence or max_iter."""
        intra_head_phase = False
        extra_head_progress = False
        for iteration in tqdm(
            range(self.max_iter), desc="Weight matching", file=sys.stdout
        ):
            sys.stdout.flush()
            progress = False
            perm_order = get_layer_iteration_order(
                self.layer_iteration_order, self.num_layers
            )

            for p_ix in perm_order:
                p = self.perm_names[p_ix]
                num_neurons, sim_matrix = self._initialize_similarity_matrices(
                    p, intra_head_phase
                )
                params_and_axes = self.ps.perm_to_layers_and_axes[p]

                for params_name, axis in params_and_axes:
                    if "identity" in params_name:
                        continue
                    w_a, w_b = (
                        self.params_a[params_name],
                        self.params_b[params_name],
                    )
                    perms_to_apply = self.ps.layer_and_axes_to_perm[
                        params_name
                    ]
                    # Lazy import to avoid circular dependency
                    from .utils import get_permuted_param

                    w_b = get_permuted_param(
                        w_b,
                        perms_to_apply,
                        self.all_perm_indices,
                        axis,
                        self.num_heads,
                        self.all_heads_indices,
                    )
                    w_a, w_b = torch.moveaxis(w_a, axis, 0), torch.moveaxis(
                        w_b, axis, 0
                    )

                    if self.p_trim is not None:
                        threshold_a = torch.quantile(
                            torch.abs(w_a), self.p_trim
                        )
                        threshold_b = torch.quantile(
                            torch.abs(w_b), self.p_trim
                        )
                        mask_a = torch.abs(w_a) >= threshold_a
                        mask_b = torch.abs(w_b) >= threshold_b
                        w_a = torch.where(mask_a, w_a, 0.0)
                        w_b = torch.where(mask_b, w_b, 0.0)

                    if "attn" in p and self.num_heads is not None:
                        self._process_attention_layer(
                            w_a, w_b, params_name, p, sim_matrix, intra_head_phase
                        )
                    else:
                        self._process_non_attention_layer(
                            w_a, w_b, num_neurons, sim_matrix
                        )

                if "attn" in p and self.num_heads is not None:
                    update = self._update_attention_perm_indices(
                        p, sim_matrix, intra_head_phase
                    )
                    if not intra_head_phase:
                        extra_head_progress = update
                else:
                    update = self._update_non_attention_perm_indices(
                        p, sim_matrix
                    )

                progress = progress or update

            if not progress:
                break
            if extra_head_progress:
                intra_head_phase = False
            else:
                intra_head_phase = True

        return self.all_perm_indices, self.all_heads_indices
