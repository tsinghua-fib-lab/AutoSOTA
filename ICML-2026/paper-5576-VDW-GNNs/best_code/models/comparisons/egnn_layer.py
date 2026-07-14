"""
EGNN layer implementation adapted from Geometric GNN Dojo (MIT license).
https://github.com/chaitjo/geometric-gnn-dojo/blob/main/models/layers/egnn_layer.py

NOTE: this implementation can optionally use a scalar edge feature via `edge_weight`.
When `use_edge_weight=True`, `edge_weight` is concatenated into the message MLP input
(as a single scalar per edge), and the message MLP input dimension is increased by 1.

This layer is used in a wind-style setup where:
- `edge_index` encodes geographic neighbors (built from Earth coordinates).
- `pos` may be swapped at runtime to a vector feature (e.g., wind velocity) so the
  equivariant geometry used in message passing is defined in that vector space.
"""

import torch
from torch.nn import Linear, ReLU, SiLU, Sequential
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_scatter import scatter


class EGNNLayer(MessagePassing):
    """E(n) Equivariant GNN Layer

    Paper: E(n) Equivariant Graph Neural Networks, Satorras et al.
    """
    def __init__(
        self,
        emb_dim,
        activation="relu",
        norm="layer",
        aggr="add",
        disable_position_mlp: bool = False,
        use_edge_weight: bool = False,
    ):
        """
        Args:
            emb_dim: (int) - hidden dimension `d`
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.use_edge_weight = bool(use_edge_weight)
        self.activation = {"swish": SiLU(), "relu": ReLU()}[activation]
        self.norm = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d}[norm]

        # MLP `\psi_h` for computing messages `m_ij`
        msg_in_dim = (2 * emb_dim) + 1 + (1 if self.use_edge_weight else 0)
        self.mlp_msg = Sequential(
            Linear(msg_in_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )
        # MLP `\psi_x` for computing messages `\overrightarrow{m}_ij`
        # Optionally disabled (e.g., final layer for invariant graph-level tasks)
        if disable_position_mlp:
            self.mlp_pos = None
        else:
            self.mlp_pos = Sequential(
                Linear(emb_dim, emb_dim), self.norm(emb_dim), self.activation, Linear(emb_dim, 1)
            )
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )

    def forward(self, h, pos, edge_index, edge_weight=None):
        """
        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_weight: (e,) or (e, 1) - optional edge strength feature (used only if `use_edge_weight=True`)
        Returns:
            out: [(n, d),(n,3)] - updated node features
        """
        out = self.propagate(edge_index, h=h, pos=pos, edge_weight=edge_weight)
        return out

    def message(self, h_i, h_j, pos_i, pos_j, edge_weight=None):
        # Compute messages
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1).unsqueeze(1)
        if self.use_edge_weight:
            if edge_weight is None:
                raise ValueError("EGNNLayer expected edge_weight when use_edge_weight=True.")
            ew = edge_weight
            if ew.dim() == 1:
                ew = ew.unsqueeze(1)
            elif ew.dim() != 2 or ew.shape[1] != 1:
                raise ValueError("EGNNLayer expected edge_weight shape (e,) or (e,1).")
            if ew.shape[0] != dists.shape[0]:
                raise ValueError("EGNNLayer edge_weight must align with number of edges.")
            ew = ew.to(device=dists.device, dtype=dists.dtype)
            msg = torch.cat([h_i, h_j, dists, ew], dim=-1)
        else:
            msg = torch.cat([h_i, h_j, dists], dim=-1)
        msg = self.mlp_msg(msg)
        # Scale magnitude of displacement vector (if enabled)
        if self.mlp_pos is not None:
            pos_diff = pos_diff * self.mlp_pos(msg)
        else:
            pos_diff = pos_diff * 0.0
        # NOTE: some papers divide pos_diff by (dists + 1) to stabilise model.
        # NOTE: lucidrains clamps pos_diff between some [-n, +n], also for stability.
        return msg, pos_diff

    def aggregate(self, inputs, index):
        msgs, pos_diffs = inputs
        # Aggregate messages
        msg_aggr = scatter(msgs, index, dim=self.node_dim, reduce=self.aggr)
        # Aggregate displacement vectors
        pos_aggr = scatter(pos_diffs, index, dim=self.node_dim, reduce="mean")
        return msg_aggr, pos_aggr

    def update(self, aggr_out, h, pos):
        msg_aggr, pos_aggr = aggr_out
        upd_out = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
        upd_pos = pos + pos_aggr
        return upd_out, upd_pos

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"


class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim, activation="relu", norm="layer", aggr="add"):
        """Vanilla Message Passing GNN layer
        
        Args:
            emb_dim: (int) - hidden dimension `d`
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.activation = {"swish": SiLU(), "relu": ReLU()}[activation]
        self.norm = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d}[norm]

        # MLP `\psi_h` for computing messages `m_ij`
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )

    def forward(self, h, edge_index):
        """
        Args:
            h: (n, d) - initial node features
            edge_index: (e, 2) - pairs of edges (i, j)
        Returns:
            out: (n, d) - updated node features
        """
        out = self.propagate(edge_index, h=h)
        return out

    def message(self, h_i, h_j):
        # Compute messages
        msg = torch.cat([h_i, h_j], dim=-1)
        msg = self.mlp_msg(msg)
        return msg

    def aggregate(self, inputs, index):
        # Aggregate messages
        msg_aggr = scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
        return msg_aggr

    def update(self, aggr_out, h):
        upd_out = self.mlp_upd(torch.cat([h, aggr_out], dim=-1))
        return upd_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"