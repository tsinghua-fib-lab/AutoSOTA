from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    global_add_pool,
    global_mean_pool,
)

try:
    from ogb.graphproppred.mol_encoder import AtomEncoder
except Exception:
    AtomEncoder = None

try:
    from .rade_convs_gc import (
        BatchedGraphCache,
        RADEGATConvGC,
        RADEGCNConvGC,
        RADEGINConvGC,
    )
except Exception:
    from rade_convs_gc import (
        BatchedGraphCache,
        RADEGATConvGC,
        RADEGCNConvGC,
        RADEGINConvGC,
    )

try:
    from .dropmessage_convs_gc import (
        DropMessageGATConvGC,
        DropMessageGCNConvGC,
        DropMessageGINConvGC,
    )
except Exception:
    from dropmessage_convs_gc import (
        DropMessageGATConvGC,
        DropMessageGCNConvGC,
        DropMessageGINConvGC,
    )


EdgeIndexObj = Union[torch.Tensor, List[torch.Tensor]]


@dataclass
class GraphMPNNConfig:
    gnn: str = "gin"
    pooling: str = "sum"
    local_layers: int = 3
    hidden_channels: int = 64
    dropout: float = 0.0
    bn: bool = False
    use_ogb_encoders: bool = False
    use_edge_attr: bool = False
    pre_linear: bool = False
    ep_correction: bool = False
    rade_variant: str = "rade-of"
    correct_self_loop: bool = True
    linear: bool = False
    aug_tech: str = "rade"
    global_num_nodes_total: int = 0
    gat_heads: int = 1
    gat_concat: bool = True
    gat_negative_slope: float = 0.2
    gat_moment_chunk_size: int = 1024
    gat_moment_samples: int = 256
    gat_nonedge_samples: int = 256
    gat_moment_seed: int = 0
    gat_ep_corr_clip: float = 2.0

    def __post_init__(self) -> None:
        self.gnn = str(self.gnn).lower().strip()
        self.pooling = str(self.pooling).lower().strip()
        self.rade_variant = str(self.rade_variant).lower().strip()
        self.aug_tech = str(self.aug_tech).lower().strip()

        if self.gnn not in {"gin", "gcn", "gat"}:
            raise ValueError(f"gnn must be 'gin', 'gcn', or 'gat'. Got {self.gnn}")
        if self.pooling not in {"sum", "mean"}:
            raise ValueError(f"pooling must be 'sum' or 'mean'. Got {self.pooling}")
        if self.rade_variant not in {"rade-of", "rade-ofs"}:
            raise ValueError(f"rade_variant must be 'rade-of' or 'rade-ofs'. Got {self.rade_variant}")
        if self.aug_tech not in {"rade", "dropout", "dropedge", "dropmessage", "dropnode", "none"}:
            raise ValueError(
                "aug_tech must be one of {rade, dropout, dropedge, dropmessage, dropnode, none}. "
                f"Got {self.aug_tech}"
            )
        if int(self.gat_heads) <= 0:
            raise ValueError(f"gat_heads must be positive. Got {self.gat_heads}")
        if self.linear:
            self.bn = False
            self.dropout = 0.0


def _make_gin_mlp(in_dim: int, out_dim: int, *, hidden_dim: int, dropout: float, linear: bool) -> nn.Module:
    if linear:
        return nn.Linear(in_dim, out_dim)
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    if float(dropout) > 0.0:
        layers.append(nn.Dropout(p=float(dropout)))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class GraphMPNN(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int, cfg: GraphMPNNConfig):
        super().__init__()
        self.cfg = cfg

        hidden_channels = int(cfg.hidden_channels)
        local_layers = int(cfg.local_layers)
        if local_layers <= 0:
            raise ValueError(f"local_layers must be >= 1. Got {local_layers}")
        if cfg.gnn == "gat" and bool(cfg.gat_concat) and (hidden_channels % int(cfg.gat_heads) != 0):
            raise ValueError(
                "For GAT with gat_concat=True, hidden_channels must be divisible by gat_heads. "
                f"Got hidden_channels={hidden_channels}, gat_heads={cfg.gat_heads}."
            )

        if bool(cfg.use_ogb_encoders):
            if AtomEncoder is None:
                raise ImportError("OGB AtomEncoder is required when use_ogb_encoders=True.")
            self.atom_encoder = AtomEncoder(hidden_channels)
        else:
            self.atom_encoder = None

        self.lin_in = nn.Linear(in_channels, hidden_channels) if bool(cfg.pre_linear) and self.atom_encoder is None else None
        first_in = hidden_channels if (self.atom_encoder is not None or self.lin_in is not None) else int(in_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for layer_idx in range(local_layers):
            in_dim = first_in if layer_idx == 0 else hidden_channels
            if cfg.aug_tech == "rade":
                if cfg.gnn == "gcn":
                    conv = RADEGCNConvGC(
                        in_dim,
                        hidden_channels,
                        bias=True,
                        correct_self_loop=cfg.correct_self_loop,
                        rade_variant=cfg.rade_variant,
                        ep_correction=cfg.ep_correction,
                    )
                elif cfg.gnn == "gin":
                    conv = RADEGINConvGC(
                        mlp=_make_gin_mlp(
                            in_dim,
                            hidden_channels,
                            hidden_dim=hidden_channels,
                            dropout=cfg.dropout,
                            linear=cfg.linear,
                        ),
                        eps=0.0,
                        train_eps=False,
                        rade_variant=cfg.rade_variant,
                        ep_correction=cfg.ep_correction,
                    )
                else:
                    conv = RADEGATConvGC(
                        in_dim,
                        hidden_channels,
                        heads=int(cfg.gat_heads),
                        concat=bool(cfg.gat_concat),
                        negative_slope=float(cfg.gat_negative_slope),
                        bias=True,
                        rade_variant=cfg.rade_variant,
                        ep_correction=cfg.ep_correction,
                        moment_chunk_size=int(cfg.gat_moment_chunk_size),
                        moment_samples=int(cfg.gat_moment_samples),
                        nonedge_samples=int(cfg.gat_nonedge_samples),
                        moment_seed=int(cfg.gat_moment_seed),
                        ep_corr_clip=float(cfg.gat_ep_corr_clip),
                    )
            elif cfg.aug_tech == "dropmessage":
                if cfg.gnn == "gcn":
                    conv = DropMessageGCNConvGC(in_dim, hidden_channels, bias=True)
                elif cfg.gnn == "gin":
                    conv = DropMessageGINConvGC(
                        mlp=_make_gin_mlp(
                            in_dim,
                            hidden_channels,
                            hidden_dim=hidden_channels,
                            dropout=cfg.dropout,
                            linear=cfg.linear,
                        ),
                        eps=0.0,
                        train_eps=False,
                    )
                else:
                    conv = DropMessageGATConvGC(
                        in_dim,
                        hidden_channels // int(cfg.gat_heads) if bool(cfg.gat_concat) else hidden_channels,
                        heads=int(cfg.gat_heads),
                        concat=bool(cfg.gat_concat),
                        negative_slope=float(cfg.gat_negative_slope),
                        bias=True,
                    )
            else:
                if cfg.gnn == "gcn":
                    conv = GCNConv(in_dim, hidden_channels, cached=False, normalize=True)
                elif cfg.gnn == "gin":
                    conv = GINConv(
                        _make_gin_mlp(
                            in_dim,
                            hidden_channels,
                            hidden_dim=hidden_channels,
                            dropout=cfg.dropout,
                            linear=cfg.linear,
                        )
                    )
                else:
                    conv = GATConv(
                        in_dim,
                        hidden_channels // int(cfg.gat_heads) if bool(cfg.gat_concat) else hidden_channels,
                        heads=int(cfg.gat_heads),
                        concat=bool(cfg.gat_concat),
                        negative_slope=float(cfg.gat_negative_slope),
                        add_self_loops=False,
                    )

            self.convs.append(conv)
            if cfg.bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.pred = nn.Linear(hidden_channels, int(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.atom_encoder is not None and hasattr(self.atom_encoder, "reset_parameters"):
            self.atom_encoder.reset_parameters()
        if self.lin_in is not None:
            self.lin_in.reset_parameters()
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.pred.reset_parameters()

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.cfg.pooling == "sum":
            return global_add_pool(x, batch)
        if self.cfg.pooling == "mean":
            return global_mean_pool(x, batch)
        raise ValueError(f"Unsupported pooling={self.cfg.pooling}")

    @staticmethod
    def _apply_dropnode(x: torch.Tensor, dropnode_rate: float, training: bool) -> torch.Tensor:
        rate = float(dropnode_rate)
        if (not training) or rate <= 0.0:
            return x
        if rate >= 1.0:
            raise ValueError(f"dropnode_rate must be in [0,1). Got {rate}")
        keep = 1.0 - rate
        mask = torch.empty((x.size(0), 1), device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * (mask / keep)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        *,
        edge_attr: Optional[torch.Tensor] = None,
        edge_index_keep: Optional[EdgeIndexObj] = None,
        edge_index_add: Optional[EdgeIndexObj] = None,
        p: float = 0.0,
        q: float = 0.0,
        dropmessage_rate: float = 0.0,
        dropnode_rate: float = 0.0,
        node_ids: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        _ = edge_attr
        edge_index = edge_index.to(torch.long)
        batch = batch.to(torch.long)
        if self.atom_encoder is not None:
            x = self.atom_encoder(x.to(torch.long))
        elif not torch.is_floating_point(x):
            x = x.float()

        if self.lin_in is not None:
            x = self.lin_in(x)
            if (not self.cfg.linear) and float(self.cfg.dropout) > 0.0:
                x = F.dropout(x, p=float(self.cfg.dropout), training=self.training)

        if self.cfg.aug_tech == "dropnode":
            x = self._apply_dropnode(x, dropnode_rate=float(dropnode_rate), training=self.training)

        needs_cache = any(hasattr(conv, "set_graph") for conv in self.convs)
        if needs_cache:
            num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0
            graph_cache = BatchedGraphCache.from_edge_index(
                edge_index,
                batch,
                num_graphs=num_graphs,
                node_ids=node_ids,
                global_num_nodes_total=int(self.cfg.global_num_nodes_total) if self.cfg.global_num_nodes_total > 0 else None,
            )
            for conv in self.convs:
                if hasattr(conv, "set_graph"):
                    conv.set_graph(graph_cache)

        num_layers = len(self.convs)

        def _pick_layer_edges(obj: Optional[EdgeIndexObj], idx: int) -> Optional[torch.Tensor]:
            if obj is None:
                return None
            if isinstance(obj, (list, tuple)):
                if len(obj) != num_layers:
                    raise ValueError(f"Expected {num_layers} per-layer masks, got {len(obj)}")
                tensor = obj[idx]
                return None if tensor is None else tensor.to(torch.long)
            return obj.to(torch.long)

        h = x
        for layer_idx, conv in enumerate(self.convs):
            keep_l = _pick_layer_edges(edge_index_keep, layer_idx)
            add_l = _pick_layer_edges(edge_index_add, layer_idx)

            if hasattr(conv, "set_graph"):
                if keep_l is None and add_l is None:
                    h = conv(h, edge_index, p=float(p), q=float(q))
                else:
                    h = conv(h, edge_index, edge_index_keep=keep_l, edge_index_add=add_l, p=float(p), q=float(q))
            elif getattr(conv, "supports_dropmessage", False):
                h = conv(h, edge_index, drop_rate=float(dropmessage_rate))
            else:
                h = conv(h, edge_index)

            if not self.cfg.linear:
                if self.cfg.bn:
                    h = self.bns[layer_idx](h)
                h = F.relu(h)
                if float(self.cfg.dropout) > 0.0:
                    h = F.dropout(h, p=float(self.cfg.dropout), training=self.training)

        node_emb = h
        pooled = self._pool(node_emb, batch)
        logits = self.pred(pooled)
        if return_embeddings:
            return logits, node_emb
        return logits
