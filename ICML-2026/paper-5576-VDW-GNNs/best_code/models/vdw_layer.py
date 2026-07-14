"""
VDWLayer: vector-only diffusion scattering + MLP block (wind setting).

This module implements a lightweight, SO(d)-equivariant message-passing layer for
vector-valued node features. Given per-node vectors \(x_i \in R^d\) and a
vector-valued diffusion operator \(Q \in R^{(Nd) x (Nd)}\) (typically a sparse
random-walk-style operator assembled for the *vector track*), the layer computes
multi-scale diffusion scattering coefficients and predicts an updated vector at
each node using a shared MLP.

## Architecture (high-level)

- **Inputs**:
  - Node vectors `x` from `batch[vector_feat_key]`, shape (N, d).
  - Sparse diffusion operator `Q` from `batch[vector_operator_key]`, shape (N*d, N*d).
  - Optional graph connectivity `edge_index` / `edge_weight` (only used when
    neighbor concatenation is enabled).

- **Diffusion scattering features**:
  - Flatten vector signals to (N*d, 1) and compute first-order diffusion wavelet
    filtrations using the `geo_scat.batch_scatter` routine (dyadic or custom
    diffusion times).
  - Reshape back to per-node vector coefficients and concatenate wavelet channels
    into a per-node feature vector of size d * (#filters [+ lowpass]).
  - Optionally include **0th-order** coefficients (the raw vectors) as a skip
    feature.

- **Optional ordered neighbor augmentation (wind experiments)**:
  - For each node, select the top-k **incoming** neighbors (sources into the
    node), optionally ordering by `edge_weight`, and concatenate their raw
    vectors (and optionally their edge weights) to the scattering feature.

- **Prediction + update**:
  - A per-node `ProjectionMLP` maps the concatenated features back to a vector in
    R^d.
  - Optionally apply a **residual update** `x <- x + f(x)` when shapes match.
  - Multiple blocks can be stacked; each block re-computes scattering features
    from the current vectors.

## Wind hyperparameters

For the wind dataset, this layer is configured via `config/yaml_files/wind/vdw_layer.yaml`
(note: `config/yaml_files/wind/egnn.yaml` selects the EGNN comparison baseline, not this class).

Outputs:
    Returns a dict with `node_vector` and `preds`, both set to the final per-node
    vector features of shape (N, d).
"""

from __future__ import annotations

from typing import Dict, Optional, List, Literal

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_sparse import SparseTensor

from models.base_module import BaseModule
from models import nn_utilities as nnu
from geo_scat import ScatterMLP
from pyg_utilities import infer_device_from_batch


class VDWLayer(BaseModule):
    """
    Lightweight vector-only VDW layer stack using diffusion scattering + MLP.

    Each layer:
    - Scatters vector features with diffusion wavelets.
    - Optionally includes 0th-order coefficients.
    - Optionally concatenates ordered neighbor raw vectors and edge weights.
    - Applies a vanilla MLP to predict a new vector per node.
    """

    def __init__(
        self,
        *,
        base_module_kwargs: Dict[str, object],
        vector_feat_key: str,
        vector_operator_key: str,
        vector_dim: int,
        diffusion_scales: Optional[torch.Tensor],
        scales_type: Literal["dyadic", "custom"],
        J: int,
        include_lowpass: bool,
        scatter_mlp_layers: int,
        scatter_mlp_hidden: List[int],
        scatter_mlp_dropout: float,
        scatter_mlp_use_batch_norm: bool,
        scatter_mlp_activation: type[nn.Module],
        scatter_include_zero_order: bool,
        scatter_use_residual: bool,
        scatter_use_neighbor_concat: bool,
        scatter_neighbor_k: Optional[int],
        scatter_neighbor_include_edge_weight: bool,
        k_neighbors: Optional[int] = None,
    ) -> None:
        super().__init__(**base_module_kwargs)
        self.vector_feat_key = str(vector_feat_key)
        self.vector_operator_key = str(vector_operator_key)
        self.vector_dim = int(vector_dim)
        self.scales_type = scales_type
        self.diffusion_scales = diffusion_scales
        self.J = int(J)
        self.include_lowpass = bool(include_lowpass)
        self.scatter_include_zero_order = bool(scatter_include_zero_order)
        self.scatter_use_residual = bool(scatter_use_residual)
        self.scatter_use_neighbor_concat = bool(scatter_use_neighbor_concat)
        self.scatter_neighbor_k = scatter_neighbor_k
        self.scatter_neighbor_include_edge_weight = bool(scatter_neighbor_include_edge_weight)
        self.k_neighbors = k_neighbors

        self.layers = nn.ModuleList(
            [
                ScatterMLP(
                    is_vector_feature=True,
                    diffusion_scales=self.diffusion_scales,
                    scales_type=self.scales_type,
                    J=self.J,
                    include_lowpass=self.include_lowpass,
                    include_zero_order=self.scatter_include_zero_order,
                    mlp_hidden=scatter_mlp_hidden,
                    mlp_dropout=scatter_mlp_dropout,
                    mlp_use_batch_norm=scatter_mlp_use_batch_norm,
                    mlp_activation=scatter_mlp_activation,
                )
                for _ in range(max(int(scatter_mlp_layers), 1))
            ]
        )

    def _build_neighbor_concat(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        *,
        k_neighbors: int,
    ) -> torch.Tensor:
        n_nodes, d = x.shape
        device = x.device
        # Use incoming edges: neighbors are sources pointing into each node
        row = edge_index[1]
        col = edge_index[0]

        def _stable_argsort(values: torch.Tensor, descending: bool = False) -> torch.Tensor:
            try:
                return torch.argsort(values, descending=descending, stable=True)
            except TypeError:
                return torch.argsort(values, descending=descending)

        if edge_weight is not None:
            order_by_weight = _stable_argsort(edge_weight, descending=True)
            row = row[order_by_weight]
            col = col[order_by_weight]
            edge_weight = edge_weight[order_by_weight]

        order_by_row = _stable_argsort(row, descending=False)
        row = row[order_by_row]
        col = col[order_by_row]
        if edge_weight is not None:
            edge_weight = edge_weight[order_by_row]

        counts = torch.bincount(row, minlength=n_nodes)
        if counts.numel() == 0:
            flat_vecs = torch.zeros((n_nodes, k_neighbors * d), device=device, dtype=x.dtype)
            if self.scatter_neighbor_include_edge_weight:
                weights_dtype = edge_weight.dtype if edge_weight is not None else x.dtype
                neighbor_weights = torch.zeros((n_nodes, k_neighbors), device=device, dtype=weights_dtype)
                return torch.cat([flat_vecs, neighbor_weights], dim=1)
            return flat_vecs

        starts = torch.cumsum(counts, dim=0) - counts
        edge_pos = torch.arange(row.numel(), device=device) - starts[row]
        mask = edge_pos < k_neighbors

        row_sel = row[mask]
        col_sel = col[mask]
        pos_sel = edge_pos[mask]

        neighbor_vectors = torch.zeros(
            (n_nodes, k_neighbors, d),
            device=device,
            dtype=x.dtype,
        )
        neighbor_vectors[row_sel, pos_sel] = x[col_sel]

        flat_vecs = neighbor_vectors.reshape(n_nodes, k_neighbors * d)
        if self.scatter_neighbor_include_edge_weight:
            weights_dtype = edge_weight.dtype if edge_weight is not None else x.dtype
            neighbor_weights = torch.zeros(
                (n_nodes, k_neighbors),
                device=device,
                dtype=weights_dtype,
            )
            if edge_weight is not None:
                neighbor_weights[row_sel, pos_sel] = edge_weight[mask]
            return torch.cat([flat_vecs, neighbor_weights], dim=1)
        return flat_vecs

    def forward(
        self,
        batch: Batch,
    ) -> Dict[str, torch.Tensor]:
        device = infer_device_from_batch(
            batch,
            feature_keys=[self.vector_feat_key],
            operator_keys=[self.vector_operator_key],
        )
        x = getattr(batch, self.vector_feat_key).to(device)
        Q = getattr(batch, self.vector_operator_key).to(device)
        # nnu.raise_if_nonfinite_tensor(x, name="vdw_layer: x")

        edge_index = getattr(batch, "edge_index", None)
        edge_weight = getattr(batch, "edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        k_neighbors = self.scatter_neighbor_k
        if k_neighbors is None:
            k_neighbors = self.k_neighbors
        k_neighbors = int(k_neighbors) if k_neighbors is not None else None

        for layer in self.layers:
            extra_features = None
            if self.scatter_use_neighbor_concat:
                if edge_index is None:
                    raise ValueError("Neighbor concat requested, but edge_index is missing.")
                if k_neighbors <= 0:
                    raise ValueError("scatter_neighbor_k must be > 0 for neighbor concatenation.")
                extra_features = self._build_neighbor_concat(
                    x=x,
                    edge_index=edge_index.to(device),
                    edge_weight=edge_weight,
                    k_neighbors=k_neighbors,
                )
            out = layer(
                x=x,
                P_sparse=Q,
                extra_features=extra_features,
            )
            if self.scatter_use_residual \
            and out.shape == x.shape:
                x = x + out
            else:
                x = out

        outputs: Dict[str, torch.Tensor] = {
            "node_vector": x,
            "preds": x,
        }
        return outputs
