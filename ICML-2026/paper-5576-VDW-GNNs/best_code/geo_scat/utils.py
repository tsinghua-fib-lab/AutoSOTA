"""Utility helpers for geometric scattering layers."""
from __future__ import annotations

from typing import Literal, Union

import torch
from torch_sparse import SparseTensor

SparseMatrix = Union[torch.Tensor, SparseTensor]


def ensure_sparse_tensor(
    matrix: torch.Tensor | SparseTensor,
) -> SparseTensor:
    """
    Convert tensors storing sparse matrices into torch_sparse SparseTensor form.
    """
    if isinstance(matrix, SparseTensor):
        return matrix
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor or SparseTensor, got type {type(matrix)}."
        )
    if matrix.layout != torch.sparse_coo:
        matrix = matrix.to_sparse()
    matrix = matrix.coalesce()
    row, col = matrix.indices()
    sparse = SparseTensor(
        row=row,
        col=col,
        value=matrix.values(),
        sparse_sizes=matrix.size(),
    )
    return sparse.coalesce()


def subset_second_order_wavelets(
    W2raw: torch.Tensor,
    *,
    feature_type: Literal['scalar', 'vector'],
) -> torch.Tensor:
    """
    Select strictly upper-triangular wavelet pairs for 2nd-order coefficients (i.e., where second-order wavelet index j' > j,
    the index of the first-order wavelet already applied).

    Args:
        W2raw:
            - Scalar track input: shape (N, C, W_prev, W_next).
            - Vector track input: shape (N * d, W_prev, W_next).
          Here W_prev is the number of first-order wavelet channels and
          W_next is the number of newly generated channels from the second
          scattering pass.
        feature_type: 'scalar' or 'vector' to indicate which layout to return.

    Returns:
        torch.Tensor
            - Scalar track output: (N, C, W_pairs) where W_pairs = W_prev * (W_prev - 1) / 2
              if W_prev == W_next, or the count of strictly upper-triangular pairs otherwise.
            - Vector track output: (N * d, 1, W_pairs).
          Intermediate mask shape: (W_prev, W_next).
    """
    if feature_type not in ('scalar', 'vector'):
        raise ValueError(f"feature_type must be 'scalar' or 'vector', got {feature_type}.")

    n_prev = int(W2raw.shape[-2])
    n_next = int(W2raw.shape[-1])
    if n_prev < 2:
        raise ValueError(
            "subset_second_order_wavelets requires at least two first-order wavelet channels."
        )

    mask = torch.triu(
        torch.ones(
            n_prev,
            n_next,
            dtype=torch.bool,
            device=W2raw.device,
        ),
        diagonal=1,
    )

    if feature_type == 'scalar':
        if W2raw.ndim != 4:
            raise ValueError(
                "Scalar 2nd-order tensors must have shape (N, C, W_prev, W_next)."
            )
        return W2raw[:, :, mask]

    if W2raw.ndim != 3:
        raise ValueError(
            "Vector 2nd-order tensors must have shape (N*d, W_prev, W_next)."
        )
    return W2raw[:, mask].unsqueeze(1)
