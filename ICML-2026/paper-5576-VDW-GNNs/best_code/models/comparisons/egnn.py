"""
E-GNN model implementation adapted from Geometric GNN Dojo (MIT license).
https://github.com/chaitjo/geometric-gnn-dojo/blob/main/models/egnn.py

This file extends the GNN Dojo's EGNN implementation to support datasets (e.g., wind)
where the graph geometry used to define neighbors is decoupled from the vector
space where we want equivariant learning.

Wind-style setup:
- **Geographic neighbors**: `edge_index` is built from Earth coordinates stored
  on the PyG `Data.pos` field during dataset construction.
- **Equivariant geometry in vector-feature space**: at runtime, the comparison
  wrapper (`models/comparisons/comparison_module.py`) can temporarily replace
  `batch.pos` with a different tensor (e.g., wind vectors) when the model exposes
  `pos_input_key`. This makes EGNN distances and coordinate updates operate in
  wind-vector space while keeping neighbor structure fixed.
- **Optional edge strength feature**: for wind/wind_rot runs we enable
  `use_edge_weight=True`, which passes `batch.edge_weight` through to each
  `EGNNLayer` as a scalar edge-strength feature.
"""

import torch
from typing import Optional
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from .egnn_layer import EGNNLayer


class EGNNModel(torch.nn.Module):
    """
    E-GNN model from "E(n) Equivariant Graph Neural Networks".
    """
    ALLOWED_POOLS = {"sum", "mean", "max"}

    def __init__(
        self,
        num_layers: int = 5,
        emb_dim: int = 128,
        in_dim: Optional[int] = 1,
        out_dim: int = 1,
        activation: str = "relu",
        norm: str = "layer",
        aggr: str = "sum",
        pool_types = None,
        residual: bool = True,
        equivariant_pred: bool = False,
        use_bias_if_no_atoms: bool = True,
        predict_per_node: bool = False,
        vector_target: bool = False,
        use_edge_weight: bool = False,
    ):
        """
        Initializes an instance of the EGNNModel class with the provided parameters.

        Parameters:
        - num_layers (int): Number of layers in the model (default: 5)
        - emb_dim (int): Dimension of the node embeddings (default: 128)
        - in_dim (Optional[int]): Vocabulary size of scalar node feature 
            (e.g., atom types). If None or 0, will not create an embedding 
            table (default: 1)
        - out_dim (int): Output dimension of the model (default: 1)
        - activation (str): Activation function to be used (default: "relu")
        - norm (str): Normalization method to be used (default: "layer")
        - aggr (str): Aggregation method to be used (default: "sum")
        - pool (str): Global pooling method to be used (default: "sum")
        - residual (bool): Whether to use residual connections (default: True)
        - equivariant_pred (bool): Whether it is an equivariant prediction task (default: False)
        - use_bias_if_no_atoms (bool): If True and `batch.atoms` is absent/None, use a learnable bias vector to create constant node features of size `emb_dim` (default: True)
        - predict_per_node (bool): If True, produce per-node predictions (no global pooling). If False, produce graph-level predictions via pooling. (default: False)
        - vector_target (bool): If True and `equivariant_pred` is True, produce a single vector output with dimensionality equal to
            `pos.size(-1)` (per node or per graph depending on `predict_per_node`). If False and `equivariant_pred` is True, produce
            a scalar invariant by combining `h` with invariant statistics (mean norm and mean squared norm) of an equivariant vector
            constructed from `pos`. (default: False)
        """
        super().__init__()
        self.equivariant_pred = equivariant_pred
        self.residual = residual
        self.use_bias_if_no_atoms = use_bias_if_no_atoms
        self.predict_per_node = predict_per_node
        self.vector_target = vector_target
        self.use_edge_weight = bool(use_edge_weight)
        self.out_dim = int(out_dim)

        # Embedding lookup for initial node features (optional)
        self.emb_in = None
        if in_dim is not None and int(in_dim) > 0:
            self.emb_in = torch.nn.Embedding(int(in_dim), emb_dim)

        # Learnable bias used to mimic a single constant node feature when no atoms are provided
        # Initialized to zeros so it does not inject information at initialization time
        self.bias_h = torch.nn.Parameter(torch.zeros(emb_dim))

        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            disable_pos = (layer_idx == num_layers - 1) and (not self.equivariant_pred) and (not self.predict_per_node)
            self.convs.append(
                EGNNLayer(
                    emb_dim,
                    activation,
                    norm,
                    aggr,
                    disable_position_mlp=disable_pos,
                    use_edge_weight=self.use_edge_weight,
                )
            )

        # Perform setup via inner functions
        self._setup_pooling(pool_types)
        self._setup_heads(emb_dim, out_dim)

    
    # -------------------------------
    # Inner setup functions
    # -------------------------------
    def _setup_pooling(self, pool_types_param):
        # Pooling configuration (invariant paths can use multiple poolings)
        if pool_types_param is None:
            pool_types_param = ["sum"]
        if isinstance(pool_types_param, str):
            pool_types_param = [pool_types_param]
        self.pool_types = [str(p).lower() for p in pool_types_param]
        for p in self.pool_types:
            if p not in self.ALLOWED_POOLS:
                raise ValueError(
                    f"Unsupported pooling type '{p}'. Allowed: {sorted(list(self.ALLOWED_POOLS))}"
                )
        # Primary pool fn for equivariant vector outputs (single vector)
        primary = self.pool_types[0]
        self._pool_map = {
            "sum": global_add_pool,
            "mean": global_mean_pool,
            "max": global_max_pool,
        }
        self.pool = self._pool_map.get(primary, global_add_pool)


    def _setup_heads(self, emb_dim, out_dim):
        if self.equivariant_pred:
            # Equivariant path always learns a scalar gate s(h)
            self.scale_head = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, 1),
            )
            if not self.vector_target:
                if self.predict_per_node:
                    self.inv_pred_node = torch.nn.Sequential(
                        torch.nn.Linear(emb_dim + 1, emb_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(emb_dim, out_dim),
                    )
                else:
                    num_pools_local = len(self.pool_types)
                    graph_in_dim = emb_dim * num_pools_local + 2 * num_pools_local
                    self.inv_pred_graph = torch.nn.Sequential(
                        torch.nn.Linear(graph_in_dim, emb_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(emb_dim, out_dim),
                    )
            self.pred = None
        else:
            if self.predict_per_node:
                self.pred_scalar_node = torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(emb_dim, out_dim)
                )
            else:
                num_pools_local = len(self.pool_types)
                self.pred_scalar_graph = torch.nn.Sequential(
                    torch.nn.Linear(emb_dim * num_pools_local, emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(emb_dim, out_dim)
                )


    # -------------------------------
    # Forward pass
    # -------------------------------
    def forward(self, batch):

        # Node feature initialization
        if hasattr(batch, "atoms") and (batch.atoms is not None) and (self.emb_in is not None):
            h = self.emb_in(batch.atoms)  # (n,) -> (n, d)
        else:
            if not self.use_bias_if_no_atoms:
                raise ValueError(
                    "EGNNModel expected `batch.atoms` but it was missing/None, and `use_bias_if_no_atoms` is False."
                )
            num_nodes = batch.pos.size(0)
            h = self.bias_h.unsqueeze(0).expand(num_nodes, -1)  # (n, d), constant learnable vector
        pos = batch.pos  # (n, 3)
        edge_weight = getattr(batch, "edge_weight", None)

        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, batch.edge_index, edge_weight=edge_weight)

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update 

            # Update node coordinates (no residual) (n, 3) -> (n, 3)
            pos = pos_update
    
        # Decide prediction granularity
        if self.predict_per_node:
            # Node-level predictions: do not pool
            if not self.equivariant_pred:
                out = h  # (n, d)
                return self.pred_scalar_node(out)  # (n, out_dim)
            else:
                # y_i = s(h_i) * pos_i preserves rotation equivariance
                scale = self.scale_head(h)  # (n, 1)
                vec = scale * pos  # (n, pos_dim)
                if self.vector_target:
                    return vec
                else:
                    # Invariant per-node feature: concat(h_i, ||vec_i||)
                    vec_norm = torch.norm(vec, dim=-1, keepdim=True)
                    inv_feat = torch.cat([h, vec_norm], dim=-1)
                    return self.inv_pred_node(inv_feat)
        else:
            # Graph-level predictions: pool then predict
            if not self.equivariant_pred:
                pooled_h_multi = self._concat_pooled_features(h, batch.batch)  # (bs, d*num_pools)
                return self.pred_scalar_graph(pooled_h_multi)  # (batch_size, out_dim)
            else:
                scale = self.scale_head(h)  # (n, 1)
                vec = scale * pos  # (n, pos_dim)
                if self.vector_target:
                    pooled_vec = self.pool(vec, batch.batch)  # (batch_size, pos_dim)
                    return pooled_vec
                else:
                    # Invariant graph-level features: concat(multi-pool(h), multi-pool(||vec||), multi-pool(||vec||^2))
                    pooled_h_multi = self._concat_pooled_features(h, batch.batch)  # (bs, d*num_pools)
                    vec_norm = torch.norm(vec, dim=-1, keepdim=True)  # (n, 1)
                    vec_sqnorm = (vec * vec).sum(dim=-1, keepdim=True)  # (n, 1)
                    pooled_vec_norm_multi = self._concat_pooled_features(vec_norm, batch.batch)  # (bs, num_pools)
                    pooled_vec_sqnorm_multi = self._concat_pooled_features(vec_sqnorm, batch.batch)  # (bs, num_pools)
                    inv_feat = torch.cat([pooled_h_multi, pooled_vec_norm_multi, pooled_vec_sqnorm_multi], dim=-1)
                    return self.inv_pred_graph(inv_feat)


    # -------------------------------
    # Helper functions
    # -------------------------------
    def _concat_pooled_features(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Apply each pooling type in self.pool_types to x and concatenate along feature dim.

        Args:
            x: Node features (N, F)
            batch_index: Graph assignment (N,)
        Returns:
            (batch_size, F * num_pools) for F>1, or (batch_size, num_pools) for F==1
        """
        pooled_list = []
        for p in self.pool_types:
            if p in self._pool_map:
                pooled = self._pool_map[p](x, batch_index)
            else:
                raise ValueError(f"Unsupported pooling type '{p}'")
            pooled_list.append(pooled)
        return torch.cat(pooled_list, dim=-1)