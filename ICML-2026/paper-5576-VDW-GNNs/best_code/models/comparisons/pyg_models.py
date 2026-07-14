import inspect
import torch
from torch import nn
from typing import List, Optional, Literal
from torch_scatter import scatter_add, scatter_max
from torch_geometric.data import Data
from torch_geometric.nn.models import GIN as PyGGIN
from torch_geometric.nn.models import GCN as PyGGCN
from torch_geometric.nn.models import GAT as PyGGAT


class PyGModel(nn.Module):
    """
    Unified adapter around PyG's GIN/GCN/GAT models with a hidden-embedding output
    and MLP readout.

    - Node-level regression: apply MLP to hidden node embeddings directly
    - Graph-level regression: pool (sum, max) per graph, concatenate, then MLP
    """

    def __init__(
        self,
        in_dim: int,
        hidden_channels: int,
        out_dim: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.2,
        feature_attr: str = 'x',
        pool_types: List[Literal['sum', 'max']] | None = None,
        predict_per_node: bool = False,
        mlp_head_hidden_dims: Optional[List[int]] = [128, 64, 32, 16],
        mlp_dropout_p: float = 0.0,
        activation: Literal['relu', 'silu', 'tanh'] = 'silu',
        backbone: Literal['gin', 'gcn', 'gat'] = 'gin',
        heads: int = 1,
    ) -> None:
        super().__init__()
        self.feature_attr = str(feature_attr)
        self.predict_per_node = bool(predict_per_node)
        self.hidden_channels = int(hidden_channels)
        self.out_dim = int(out_dim)
        backbone = str(backbone).lower()
        self.mlp_dropout_p = mlp_dropout_p
        # Backbone returns hidden node embeddings of size hidden_channels
        if backbone == 'gin':
            self.backbone = PyGGIN(
                in_channels=int(in_dim),
                hidden_channels=int(hidden_channels),
                out_channels=int(hidden_channels),
                num_layers=int(num_layers),
                dropout=float(dropout),
            )
        elif backbone == 'gcn':
            self.backbone = PyGGCN(
                in_channels=int(in_dim),
                hidden_channels=int(hidden_channels),
                out_channels=int(hidden_channels),
                num_layers=int(num_layers),
                dropout=float(dropout),
                jk=None,
            )
        elif backbone == 'gat':
            self.backbone = PyGGAT(
                in_channels=int(in_dim),
                hidden_channels=int(hidden_channels),
                out_channels=int(hidden_channels),
                num_layers=int(num_layers),
                dropout=float(dropout),
                heads=int(heads),
            )
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'. Use one of ['gin','gcn','gat'].")

        # Pooling configuration
        if pool_types is None:
            pool_types = ['sum', 'max']
        self.pool_types = [str(p).lower() for p in pool_types]
        for p in self.pool_types:
            if p not in {'sum', 'max'}:
                raise ValueError("pool_types must be drawn from ['sum','max']")

        # Readout MLP
        act = (
            nn.ReLU
            if str(activation).lower() == 'relu'
            else (nn.Tanh if str(activation).lower() == 'tanh' else nn.SiLU)
        )
        in_dim_head = self.hidden_channels if self.predict_per_node else (self.hidden_channels * len(self.pool_types))
        layers: List[nn.Module] = []
        cur = in_dim_head
        for h in mlp_head_hidden_dims:
            layers.append(nn.Linear(cur, int(h)))
            if self.mlp_dropout_p > 0.0:
                layers.append(nn.Dropout(self.mlp_dropout_p))
            layers.append(act())
            cur = int(h)
        layers.append(nn.Linear(cur, self.out_dim))
        self.mlp_head = nn.Sequential(*layers)

    def _pool_concat(self, x: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        if batch is None:
            parts = []
            if 'sum' in self.pool_types:
                parts.append(x.sum(dim=0, keepdim=True))
            if 'max' in self.pool_types:
                parts.append(x.max(dim=0, keepdim=True).values)
            return torch.cat(parts, dim=-1)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        parts = []
        if 'sum' in self.pool_types:
            parts.append(scatter_add(x, batch, dim=0, dim_size=num_graphs))
        if 'max' in self.pool_types:
            parts.append(scatter_max(x, batch, dim=0, dim_size=num_graphs)[0])
        return torch.cat(parts, dim=-1)

    def forward(self, data: Data) -> torch.Tensor:
        x = getattr(data, self.feature_attr, None)
        if x is None:
            raise ValueError(
                f"PyGModel expected feature '{self.feature_attr}' on data but it was missing."
            )
        if (not isinstance(x, torch.Tensor)) or x.dim() != 2:
            raise ValueError(f"Feature '{self.feature_attr}' must be a 2D torch.Tensor of shape (N, F)")

        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is not None:
            sig = inspect.signature(self.backbone.forward)
            params = sig.parameters
            if "edge_weight" in params:
                h = self.backbone(x, data.edge_index, edge_weight=edge_weight)
            elif "edge_attr" in params:
                h = self.backbone(x, data.edge_index, edge_attr=edge_weight)
            else:
                h = self.backbone(x, data.edge_index)
        else:
            h = self.backbone(x, data.edge_index)

        if self.predict_per_node:
            return self.mlp_head(h)
        pooled = self._pool_concat(h, getattr(data, 'batch', None))
        return self.mlp_head(pooled)


