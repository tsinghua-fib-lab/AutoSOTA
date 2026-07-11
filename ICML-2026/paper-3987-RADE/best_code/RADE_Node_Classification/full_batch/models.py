import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv
from typing import Optional, Union, List

from rade_convs import GraphCache, RADEGATConv, RADEGCNConv, RADEGINConv
from dropmessage_convs import DropMessageGATConv, DropMessageGCNConv, DropMessageGINConv


EdgeIndexObj = Union[torch.Tensor, List[torch.Tensor]]


class MPNNs(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        local_layers=3,
        dropout=0.5,
        pre_linear=False,
        res=False,
        ln=False,
        bn=False,
        jk=False,
        gnn="gcn",
        ep_correction: bool = False,
        rade_variant: str = "rade-of",
        linear: bool = False,
        aug_tech: str = "rade",
        gat_heads: int = 1,
        gat_concat: bool = True,
        gat_negative_slope: float = 0.2,
        gat_moment_samples: int = 256,
        gat_nonedge_samples: int = 256,
        gat_moment_seed: int = 0,
        gat_ep_corr_clip: float = 2.0,
    ):
        super(MPNNs, self).__init__()

        assert gnn in ("gcn", "gin", "gat"), "gnn must be 'gcn', 'gin', or 'gat'"
        assert local_layers >= 1, "local_layers must be >= 1"
        if rade_variant not in ("rade-of", "rade-ofs"):
            raise ValueError(
                f"rade_variant must be one of {{'rade-of', 'rade-ofs'}}. Got {rade_variant}"
            )
        self.linear = bool(linear)
        if gnn == "gat" and self.linear:
            raise ValueError(
                "linear=True is unavailable for gnn='gat' because GAT attention is "
                "feature-dependent and nonlinear. Use linear=False for GAT, or use "
                "the sgc / gin-linear CLI backbones for strict linear baselines."
            )

        # Enforce strict linearity at the architecture level
        if self.linear:
            dropout = 0.0
            ln = False
            bn = False

        if ln and bn:
            raise ValueError("Choose at most one normalization: ln=True or bn=True (not both).")

        self.dropout = float(dropout)
        self.pre_linear = bool(pre_linear)
        self.res = bool(res)
        self.ln = bool(ln)
        self.bn = bool(bn)
        self.jk = bool(jk)
        self.gnn = gnn
        self.gat_heads = int(gat_heads)
        self.gat_concat = bool(gat_concat)
        self.gat_negative_slope = float(gat_negative_slope)
        self.gat_moment_samples = int(gat_moment_samples)
        self.gat_nonedge_samples = int(gat_nonedge_samples)
        self.gat_moment_seed = int(gat_moment_seed)
        self.gat_ep_corr_clip = max(0.0, float(gat_ep_corr_clip))

        self.ep_correction = bool(ep_correction)
        self.rade_variant = str(rade_variant)

        self.aug_tech = str(aug_tech).lower().strip()
        if self.aug_tech not in {"rade", "dropout", "dropedge", "dropmessage", "dropnode", "none"}:
            raise ValueError(
                f"aug_tech must be one of {{rade, dropout, dropedge, dropmessage, dropnode, none}}. Got {self.aug_tech}"
            )

        self.lin_in = torch.nn.Linear(in_channels, hidden_channels) if self.pre_linear else None

        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList() if self.res else None
        self.lns = torch.nn.ModuleList() if self.ln else None
        self.bns = torch.nn.ModuleList() if self.bn else None

        def make_gin_mlp(in_dim: int, out_dim: int) -> nn.Module:
            if self.linear:
                return nn.Linear(in_dim, out_dim)
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )

        first_in = hidden_channels if self.pre_linear else in_channels
        L = int(local_layers)

        # -------------------------
        # Build layers
        # -------------------------
        for layer_idx in range(L):
            in_dim = first_in if layer_idx == 0 else hidden_channels
            out_dim = hidden_channels

            if gnn == "gin":
                mlp = make_gin_mlp(in_dim, out_dim)

                if self.aug_tech == "rade":
                    conv = RADEGINConv(
                        mlp,
                        rade_variant=self.rade_variant,
                        ep_correction=self.ep_correction,
                    )
                elif self.aug_tech == "dropmessage":
                    conv = DropMessageGINConv(mlp)
                else:
                    conv = GINConv(mlp)

            elif gnn == "gcn":
                if self.aug_tech == "rade":
                    conv = RADEGCNConv(
                        in_dim,
                        out_dim,
                        bias=True,
                        correct_self_loop=True,
                        rade_variant=self.rade_variant,
                        ep_correction=self.ep_correction,
                    )
                elif self.aug_tech == "dropmessage":
                    conv = DropMessageGCNConv(in_dim, out_dim, bias=True)
                else:
                    conv = GCNConv(in_dim, out_dim, cached=False, normalize=True)
            else:
                if self.aug_tech == "rade":
                    conv = RADEGATConv(
                        in_dim,
                        out_dim,
                        heads=self.gat_heads,
                        concat=self.gat_concat,
                        negative_slope=self.gat_negative_slope,
                        bias=True,
                        rade_variant=self.rade_variant,
                        ep_correction=self.ep_correction,
                        moment_samples=self.gat_moment_samples,
                        nonedge_samples=self.gat_nonedge_samples,
                        moment_seed=self.gat_moment_seed,
                        ep_corr_clip=self.gat_ep_corr_clip,
                    )
                elif self.aug_tech == "dropmessage":
                    if self.gat_concat and out_dim % self.gat_heads != 0:
                        raise ValueError(
                            "For GAT concat=True, hidden_channels must be divisible by gat_heads. "
                            f"Got hidden_channels={out_dim}, gat_heads={self.gat_heads}."
                        )
                    gat_out = out_dim // self.gat_heads if self.gat_concat else out_dim
                    conv = DropMessageGATConv(
                        in_dim,
                        gat_out,
                        heads=self.gat_heads,
                        concat=self.gat_concat,
                        negative_slope=self.gat_negative_slope,
                        bias=True,
                    )
                else:
                    if self.gat_concat and out_dim % self.gat_heads != 0:
                        raise ValueError(
                            "For GAT concat=True, hidden_channels must be divisible by gat_heads. "
                            f"Got hidden_channels={out_dim}, gat_heads={self.gat_heads}."
                        )
                    gat_out = out_dim // self.gat_heads if self.gat_concat else out_dim
                    conv = GATConv(
                        in_dim,
                        gat_out,
                        heads=self.gat_heads,
                        concat=self.gat_concat,
                        negative_slope=self.gat_negative_slope,
                        dropout=0.0,
                        add_self_loops=False,
                        bias=True,
                    )

            self.local_convs.append(conv)

            if self.res:
                self.lins.append(nn.Linear(in_dim, out_dim))
            if self.ln:
                self.lns.append(nn.LayerNorm(out_dim))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(out_dim))

        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.local_convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        if self.lins is not None:
            for lin in self.lins:
                lin.reset_parameters()
        if self.lns is not None:
            for ln in self.lns:
                ln.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()
        if self.lin_in is not None:
            self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()

    def set_graph(self, edge_index_clean: torch.Tensor, num_nodes: int) -> None:
        cache = GraphCache.from_edge_index(edge_index_clean, num_nodes)
        for conv in self.local_convs:
            if hasattr(conv, "set_graph"):
                conv.set_graph(cache)

    def reset_ep_stats(self, p: float, q: float) -> None:
        for conv in self.local_convs:
            if hasattr(conv, "reset_ep_stats"):
                conv.reset_ep_stats(p=float(p), q=float(q))

    def update_ep_stats(
        self,
        *,
        edge_index_keep: Optional[EdgeIndexObj],
        edge_index_add: Optional[EdgeIndexObj],
        p: float,
        q: float,
    ) -> None:
        def _is_list(t):
            return isinstance(t, (list, tuple))

        L = len(self.local_convs)

        def _pick_layer_edges(obj: Optional[EdgeIndexObj], i: int) -> Optional[torch.Tensor]:
            if obj is None:
                return None
            if _is_list(obj):
                if len(obj) != L:
                    raise ValueError(f"Expected list length={L}, got {len(obj)}")
                t = obj[i]
                return None if t is None else t
            return obj

        for i, conv in enumerate(self.local_convs):
            if hasattr(conv, "update_ep_stats"):
                layer_keep = _pick_layer_edges(edge_index_keep, i)
                layer_add = _pick_layer_edges(edge_index_add, i)
                conv.update_ep_stats(
                    edge_index_keep=layer_keep,
                    edge_index_add=layer_add,
                    p=float(p),
                    q=float(q),
                )

    @staticmethod
    def _apply_dropnode(x: torch.Tensor, dropnode_rate: float, training: bool) -> torch.Tensor:
        """
        DropNode baseline: node-wise feature masking.
        """
        r = float(dropnode_rate)
        if (not training) or r <= 0.0:
            return x
        if r >= 1.0:
            raise ValueError(f"dropnode_rate must be in [0,1). Got {r}")
        keep = 1.0 - r
        mask = torch.empty((x.size(0), 1), device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * (mask / keep)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_clean: torch.Tensor,
        *,
        edge_index_keep: Optional[EdgeIndexObj] = None,
        edge_index_add: Optional[EdgeIndexObj] = None,
        p: float = 0.0,
        q: float = 0.0,
        dropmessage_rate: float = 0.0,
        dropnode_rate: float = 0.0,
        return_embeddings: bool = False,
    ):
        # Optional input projection
        if self.lin_in is not None:
            x = self.lin_in(x)
            if (not self.linear) and (self.dropout > 0.0):
                x = F.dropout(x, p=self.dropout, training=self.training)

        # DropNode baseline
        if self.aug_tech == "dropnode":
            x = self._apply_dropnode(x, dropnode_rate=float(dropnode_rate), training=self.training)

        def _is_list(t):
            return isinstance(t, (list, tuple))

        L = len(self.local_convs)

        # Active when training on sampled RADE views.
        rade_active = (
            self.aug_tech == "rade"
            and self.training
            and (edge_index_keep is not None or edge_index_add is not None)
            and (p > 0.0 or q > 0.0)
        )

        def _pick_layer_edges(obj: Optional[EdgeIndexObj], i: int) -> Optional[torch.Tensor]:
            if obj is None:
                return None
            if _is_list(obj):
                if len(obj) != L:
                    raise ValueError(f"Expected list length={L}, got {len(obj)}")
                t = obj[i]
                return None if t is None else t
            return obj

        x_final = None

        for i, conv in enumerate(self.local_convs):
            x_in = x  # for residual path

            apply_rade_here = rade_active and hasattr(conv, "set_graph")

            if apply_rade_here:
                layer_keep = _pick_layer_edges(edge_index_keep, i)
                layer_add = _pick_layer_edges(edge_index_add, i)
                x = conv(
                    x,
                    edge_index_clean,
                    edge_index_keep=layer_keep,
                    edge_index_add=layer_add,
                    p=p,
                    q=q,
                )
            else:
                # RADE convs may still use p,q in eval for RADE-OFS inference correction.
                if hasattr(conv, "set_graph"):
                    x = conv(x, edge_index_clean, p=p, q=q)
                elif getattr(conv, "supports_dropmessage", False):
                    x = conv(x, edge_index_clean, drop_rate=float(dropmessage_rate))
                else:
                    x = conv(x, edge_index_clean)

            if self.res:
                x = x + self.lins[i](x_in)

            if not self.linear:
                if self.ln:
                    x = self.lns[i](x)
                elif self.bn:
                    x = self.bns[i](x)

                x = F.relu(x)
                if self.dropout > 0.0:
                    x = F.dropout(x, p=self.dropout, training=self.training)

            if self.jk:
                x_final = x if x_final is None else (x_final + x)
            else:
                x_final = x

        logits = self.pred_local(x_final)
        if return_embeddings:
            return logits, x_final
        return logits
