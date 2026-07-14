"""
Core geometric scattering layers and methods.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_scatter import scatter_logsumexp, scatter_sum
from torch_sparse import SparseTensor, matmul
from torch.func import functional_call, stack_module_state, vmap

from models.nn_utilities import ProjectionMLP
from models import nn_utilities as nnu
from .utils import ensure_sparse_tensor, subset_second_order_wavelets

__all__ = [
    "LearnableP",
    "MultiLearnableP",
    "MultiViewScatter",
    "LearnableMahalanobisTopK",
    "ScatterMLP",
    "VectorBatchNorm",
    "batch_scatter",
    "vector_multiorder_scatter",
    "materialize_learned_diffusion_ops",
    "build_learnable_diffusion_ops",
]


class VectorBatchNorm(nn.Module):
    """
    Batch normalization for vector-valued features that preserves SO(d) equivariance.

    Given inputs of shape (batch, 1, d, W) representing wavelet coefficients per
    vector component and wavelet channel, this layer normalizes each channel's
    vector norm to have mean 1 while leaving directions unchanged.
    """

    def __init__(
        self,
        num_wavelets: int,
        momentum: float = 0.1,
        eps: float = 1e-6,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_wavelets = int(num_wavelets)
        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats

        if track_running_stats:
            self.register_buffer("running_mean_norm", torch.ones(num_wavelets))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean_norm", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"VectorBatchNorm expects 4D input (batch, 1, d, W), got {x.shape}")
        if x.shape[-1] != self.num_wavelets:
            raise ValueError(
                f"Expected {self.num_wavelets} wavelet channels, received {x.shape[-1]}"
            )

        norms = torch.norm(x.squeeze(1), p=2, dim=1)  # (batch, W)
        if self.training and self.track_running_stats:
            batch_mean = norms.mean(dim=0)
            if int(self.num_batches_tracked) == 0:
                self.running_mean_norm = batch_mean.detach()
            else:
                self.running_mean_norm = (
                    (1 - self.momentum) * self.running_mean_norm
                    + self.momentum * batch_mean.detach()
                )
            self.num_batches_tracked += 1
            mean_norm = batch_mean
        else:
            if self.track_running_stats and self.running_mean_norm is not None:
                mean_norm = self.running_mean_norm
            else:
                mean_norm = norms.mean(dim=0)

        scale = mean_norm.to(x.device).view(1, 1, 1, self.num_wavelets) + self.eps
        return x / scale


class ScatterMLP(nn.Module):
    """
    Scattering followed by a vanilla MLP for scalar or vector features.

    - Compute 1st-order diffusion wavelets via batch_scatter (2nd-order not yet supported).
    - Optionally include 0th-order coefficients (original features).
    - Optionally concatenate extra features (e.g., neighbor vectors/weights).
    - Predict a new per-node feature with the same channel dimension as input.
    """

    def __init__(
        self,
        *,
        is_vector_feature: bool = False,
        diffusion_scales: Optional[torch.Tensor] = None,
        scales_type: Literal["dyadic", "custom"] = "custom",
        J: int = 4,
        include_lowpass: bool = True,
        include_zero_order: bool = True,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: float = 0.0,
        mlp_use_batch_norm: bool = False,
        mlp_activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.vector_feature = bool(is_vector_feature)
        self.diffusion_scales = diffusion_scales
        self.scales_type = scales_type
        self.J = int(J)
        self.include_lowpass = bool(include_lowpass)
        self.include_zero_order = bool(include_zero_order)
        self.mlp_hidden = mlp_hidden or [128, 128]
        self.mlp_dropout = float(mlp_dropout)
        self.mlp_use_batch_norm = bool(mlp_use_batch_norm)
        self.mlp_activation = mlp_activation
        self._mlp: Optional[ProjectionMLP] = None
        self._mlp_in_dim: Optional[int] = None

    def _ensure_mlp(
        self,
        in_dim: int,
        out_dim: int,
        device: torch.device,
    ) -> None:
        if (self._mlp is None) or (self._mlp_in_dim != in_dim):
            self._mlp_in_dim = int(in_dim)
            self._mlp = ProjectionMLP(
                in_dim=int(in_dim),
                hidden_dim=list(self.mlp_hidden),
                embedding_dim=int(out_dim),
                activation=self.mlp_activation,
                use_batch_norm=self.mlp_use_batch_norm,
                dropout_p=self.mlp_dropout,
            ).to(device)

    def forward(
        self,
        x: torch.Tensor,
        P_sparse: torch.Tensor | SparseTensor,
        *,
        extra_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"ScatterMLP expects x with shape (N, C); got {x.shape}")
        n_nodes, c = x.shape
        if self.vector_feature:
            x_flat = x.reshape(n_nodes * c, 1)
        else:
            x_flat = x
        W = batch_scatter(
            x=x_flat,
            P_sparse=P_sparse,
            scales_type=self.scales_type,
            diffusion_scales=self.diffusion_scales,
            J=self.J,
            include_lowpass=self.include_lowpass,
            filter_stack_dim=-1,
        )  # (N*d, 1, W)
        if self.vector_feature:
            W = W.view(n_nodes, c, -1)  # (N, d, W)
            scat_flat = W.reshape(n_nodes, -1)  # (N, d*W)
        else:
            scat_flat = W.reshape(n_nodes, -1)  # (N, C*W)
        feats = [scat_flat]
        if self.include_zero_order:
            feats.append(x)
        if extra_features is not None:
            feats.append(extra_features)
        mlp_in = torch.cat(feats, dim=1)
        self._ensure_mlp(
            in_dim=int(mlp_in.shape[1]),
            out_dim=int(c),
            device=mlp_in.device,
        )
        if self._mlp is None:
            raise RuntimeError("ScatterMLP MLP was not initialized.")
        return self._mlp(mlp_in)


class LearnableP(nn.Module):
    """
    Learnable parameterization of the lazy random walk diffusion operator P.

    The module predicts per-edge transition probabilities as well as an optional
    learnable laziness (diagonal) coefficient. Off-diagonal weights are produced
    by an MLP over custom edge features and normalized so that the outgoing
    probabilities for each node sum to (1 - alpha).
    """

    def __init__(
        self,
        learn_laziness: bool = True,
        edge_mlp_kwargs: Optional[Dict[str, Any]] = None,
        laziness_init: float = 0.5,
        use_softmax: bool = True,
        softmax_temp: float = 1.0,
        use_attention: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not (0.0 < laziness_init < 1.0):
            raise ValueError(
                f"LearnableP laziness_init must be in (0,1), received {laziness_init}."
            )
        default_mlp_kwargs: Dict[str, Any] = {
            'hidden_dim': [64, 64],
            'embedding_dim': 1,
            'activation': nn.ReLU,
            'use_batch_norm': False,
        }
        self.use_softmax = use_softmax
        self.use_attention = bool(use_attention)
        # Learnable log temperature to ensure positivity
        self.softmax_temp_param = nn.Parameter(
            torch.log(torch.tensor(float(softmax_temp), dtype=torch.float32))
        )
        self.edge_mlp_kwargs = default_mlp_kwargs
        if edge_mlp_kwargs is not None:
            self.edge_mlp_kwargs = {**default_mlp_kwargs, **edge_mlp_kwargs}
        self.edge_mlp: Optional[ProjectionMLP] = None
        self.learn_laziness = learn_laziness
        self.eps = eps
        logit = math.log(laziness_init) - math.log(1.0 - laziness_init)
        alpha_tensor = torch.tensor(logit, dtype=torch.float32)
        if learn_laziness:
            self.alpha_param = nn.Parameter(alpha_tensor)
        else:
            self.register_buffer('alpha_param', alpha_tensor)

        default_attention = {
            "num_heads": 1,
            "dropout": 0.0,
        }
        self.attention_kwargs = {
            **default_attention, 
            **(attention_kwargs or {})
        }
        self.attn_q: Optional[nn.Linear] = None
        self.attn_k: Optional[nn.Linear] = None

        # Cache the alpha value for the current batch
        self._cached_alpha: Optional[torch.Tensor] = None

    def _lazy_init_edge_mlp(self, input_dim: int, device: torch.device) -> None:
        kwargs = dict(self.edge_mlp_kwargs)
        self.edge_mlp = ProjectionMLP(
            in_dim=input_dim,
            hidden_dim=kwargs.get('hidden_dim', [128, 128]),
            embedding_dim=kwargs.get('embedding_dim', 1),
            activation=kwargs.get('activation', nn.ReLU),
            use_batch_norm=kwargs.get('use_batch_norm', False),
        ).to(device)

    def _lazy_init_attention_projs(self, embed_dim: int, device: torch.device) -> None:
        num_heads = int(self.attention_kwargs.get("num_heads", 1))
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Attention requires embedding_dim divisible by num_heads. "
                f"Got embed_dim={embed_dim}, num_heads={num_heads}."
            )
        if self.attn_q is None:
            self.attn_q = nn.Linear(embed_dim, embed_dim, bias=False).to(device)
        if self.attn_k is None:
            self.attn_k = nn.Linear(embed_dim, embed_dim, bias=False).to(device)

    def current_alpha(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Returns the laziness coefficient alpha in (0,1).
        """
        alpha = torch.sigmoid(self.alpha_param)
        if device is not None and alpha.device != device:
            alpha = alpha.to(device)
        return alpha

    def current_temperature(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Returns a positive softmax temperature.
        """
        temp = torch.exp(self.softmax_temp_param)
        if device is not None and temp.device != device:
            temp = temp.to(device)
        return temp

    def normalize_with_softmax_temperature(
        self,
        edge_logits: torch.Tensor,
        edge_index: torch.Tensor,
        alpha: float,
        temperature: float,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Normalize edge logits with temperature-scaled softmax.
        Note that alpha and temperature are antagonistic parameters:
        a larger alpha slows smoothing, while a larger temperature
        promotes smoothing.
        Args:
            edge_logits: logits for each edge, shape [num_edges]
            edge_index: edge indices, shape [2, num_edges]
            alpha: laziness coefficient, in (0,1)
            temperature: temperature (tau) for softmax, > 0 
                (larger tau = smoother softmax)
            num_nodes: number of nodes in the graph
        Returns:
            normalized edge weights, shape [num_edges]
        """
        # edge_logits has shape [num_edges]
        row = edge_index[0]

        # Scale logits by temperature
        scaled_logits = edge_logits / temperature

        # For each source node, compute logsumexp over its outgoing edges
        # scatter_logsumexp returns a tensor of shape [num_nodes]
        logsumexp_per_node = scatter_logsumexp(
            scaled_logits,
            row,
            dim=0,
            dim_size=num_nodes,
        )

        # Compute normalized probabilities
        # p_ij = (1 - alpha) * softmax(logits / tau)
        normalized = torch.exp(scaled_logits - logsumexp_per_node[row])
        normalized = (1.0 - alpha) * normalized

        return normalized

    def attention_normalize(
        self,
        edge_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        alpha: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Normalize edge weights using per-node queries derived from outgoing edges.
        """
        params = dict(self.attention_kwargs)
        num_heads = int(params.get("num_heads", 1))
        dropout = float(params.get("dropout", 0.0))
        embed_dim = int(edge_embeddings.shape[-1])
        head_dim = embed_dim // num_heads
        device = edge_embeddings.device
        self._lazy_init_attention_projs(embed_dim=embed_dim, device=device)

        q_tokens = self.attn_q(edge_embeddings).view(-1, num_heads, head_dim)
        k_tokens = self.attn_k(edge_embeddings).view(-1, num_heads, head_dim)

        row = edge_index[0].to(torch.long)
        node_queries = torch.zeros(
            num_nodes,
            num_heads,
            head_dim,
            device=device,
            dtype=edge_embeddings.dtype,
        )
        node_queries.index_add_(0, row, q_tokens)
        counts = torch.bincount(row, minlength=num_nodes).clamp_min(1)
        node_queries = node_queries / counts.view(-1, 1, 1).to(node_queries.dtype)
        q_aligned = node_queries[row]  # (E, num_heads, head_dim)

        scores = (q_aligned * k_tokens).sum(dim=-1) / math.sqrt(head_dim)
        scores = scores.mean(dim=1)  # (E,)
        logsumexp = scatter_logsumexp(scores, row, dim=0, dim_size=num_nodes)
        attn = torch.exp(scores - logsumexp[row])
        if dropout > 0 and self.training:
            attn = F.dropout(attn, p=dropout)
        attn = (1.0 - alpha) * attn
        return attn

    def forward(
        self,
        num_nodes: int,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        device: Optional[torch.device] = None,
        attention_mode: Optional[bool] = None,
    ) -> torch.Tensor:

        # Validate input shapes
        if edge_features.dim() != 2:
            raise ValueError(
                f"edge_features must be 2D (E, F); received shape {edge_features.shape}."
            )
        if edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape (2, E); received {edge_index.shape}."
            )

        # Move inputs to device
        if device is None:
            device = edge_features.device
        edge_features = edge_features.to(device)
        edge_index = edge_index.to(device)

        # Pass edge features through MLP and rescaling transform to get edge weights
        if self.edge_mlp is None:
            self._lazy_init_edge_mlp(int(edge_features.shape[-1]), device)

        edge_repr = self.edge_mlp(edge_features)
        embed_dim = int(edge_repr.shape[-1])
        alpha = self.current_alpha(device=edge_repr.device)
        use_attention = attention_mode if attention_mode is not None else self.use_attention
        temperature = self.current_temperature(device=edge_repr.device)

        if use_attention:
            normalized = self.attention_normalize(
                edge_embeddings=edge_repr,
                edge_index=edge_index,
                alpha=alpha,
                num_nodes=num_nodes,
            )
        elif self.use_softmax:
            if embed_dim != 1:
                raise ValueError(
                    "Softmax normalization expects edge MLP embedding_dim=1 "
                    f"(got {embed_dim})."
                )
            edge_logits = edge_repr.view(-1)
            normalized = self.normalize_with_softmax_temperature(
                edge_logits=edge_logits,
                edge_index=edge_index,
                alpha=alpha,
                temperature=float(temperature),
                num_nodes=num_nodes,
            )
        else: # use sigmoid
            edge_weights = torch.sigmoid(edge_repr).view(-1)
            row = edge_index[0].to(torch.long)
            degree = torch.zeros(
                int(num_nodes),
                dtype=edge_weights.dtype,
                device=edge_weights.device,
            )
            degree.index_add_(0, row, edge_weights)
            row_degree = degree[row].clamp_min(self.eps)
            normalized = edge_weights * (1.0 - alpha) / row_degree

        # Cache the alpha value for the current batch
        self._cached_alpha = alpha

        # Return normalized edge weights
        return normalized  # shape [num_edges]


class MultiLearnableP(nn.Module):
    """
    Ensemble of LearnableP modules to model multiple diffusion views.

    Each view shares the same topology (edge_index) but owns independent
    edge MLPs, laziness, normalization temperature, and attention projections. The
    outputs are stacked into a tensor of shape (num_edges, num_views),
    ready to be integrated into multi-view scattering pipelines such as
    VDW SupCon models. Views are currently evaluated sequentially for
    clarity; once the project standardizes on a PyTorch release with stable
    ``torch.vmap`` support for modules, this class can be upgraded to evaluate
    views in parallel without breaking its API.
    """

    def __init__(
        self,
        num_views: int = 3,
        *,
        learn_laziness: bool = True,
        laziness_inits: Optional[Sequence[float]] = None,
        softmax_temps: Optional[Sequence[float]] = None,
        edge_mlp_kwargs: Optional[Dict[str, Any]] = None,
        use_softmax: bool = True,
        use_attention: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if num_views < 1:
            raise ValueError("MultiLearnableP requires at least one view.")
        self.num_views = int(num_views)
        self.use_softmax = bool(use_softmax)
        self.use_attention = bool(use_attention)

        laziness_defaults = laziness_inits or (0.75, 0.6, 0.4)
        temp_defaults = softmax_temps or (0.1, 0.3, 1.0)
        laziness_vals = self._match_hparam_length(
            laziness_defaults,
            target=self.num_views,
            name="laziness_inits",
        )
        temp_vals = self._match_hparam_length(
            temp_defaults,
            target=self.num_views,
            name="softmax_temps",
        )

        self.views = nn.ModuleList(
            [
                LearnableP(
                    learn_laziness=learn_laziness,
                    edge_mlp_kwargs=edge_mlp_kwargs,
                    laziness_init=float(laziness_vals[i]),
                    use_softmax=self.use_softmax,
                    softmax_temp=float(temp_vals[i]),
                    use_attention=self.use_attention,
                    attention_kwargs=attention_kwargs,
                )
                for i in range(self.num_views)
            ]
        )

    def current_alpha(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Returns a tensor of shape (num_views,) containing each view's laziness.
        """
        alphas = [
            view.current_alpha(device=device)
            for view in self.views
        ]
        return torch.stack(alphas)

    @staticmethod
    def _match_hparam_length(
        values: Sequence[float],
        *,
        target: int,
        name: str,
    ) -> Sequence[float]:
        if len(values) == 0:
            raise ValueError(f"{name} must provide at least one value.")
        if len(values) == target:
            return values
        if len(values) > target:
            return values[:target]
        # pad by repeating last value
        pad_value = values[-1]
        padded = list(values)
        padded.extend([pad_value] * (target - len(values)))
        return padded

    def forward(
        self,
        num_nodes: int,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
        attention_mode: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            num_nodes: number of nodes in the graph.
            edge_features: tensor of shape (E, F).
            edge_index: tensor of shape (2, E).
            device: optional device override for the views.
            attention_mode: override for self.use_attention per forward call.
        Returns:
            Tensor of shape (E, num_views) with view-specific edge weights.
        """
        use_attention_flag = (
            attention_mode if attention_mode is not None else self.use_attention
        )

        if self.num_views == 1:
            single = self.views[0](
                num_nodes=num_nodes,
                edge_features=edge_features,
                edge_index=edge_index,
                device=device,
                attention_mode=attention_mode,
            )
            return single.unsqueeze(-1)

        target_device = device or edge_features.device
        input_dim = int(edge_features.shape[-1])
        for view in self.views:
            view._lazy_init_edge_mlp(input_dim, target_device)
            if view.use_attention and view.attn_q is None:
                embed_dim = int(view.edge_mlp_kwargs.get("embedding_dim", 1))
                view._lazy_init_attention_projs(embed_dim, target_device)

        # Run views sequentially (avoid vmap issues in both attention and non-attention modes).
        outputs = []
        for view in self.views:
            out = view(
                num_nodes=num_nodes,
                edge_features=edge_features,
                edge_index=edge_index,
                device=target_device,
                attention_mode=attention_mode,
            )
            outputs.append(out)
        return torch.stack(outputs, dim=1)  # (E, V)

    @staticmethod
    def compute_view_similarities(
        ops: Sequence[SparseTensor],
    ) -> list[tuple[int, int, float]]:
        """
        Compute pairwise Frobenius norms between view-specific diffusion operators.

        Assumes all SparseTensor ops share the same sparsity pattern.
        Returns list of tuples: (i, j, frob_norm).
        """
        sims: list[tuple[int, int, float]] = []
        if len(ops) < 2:
            return sims

        def _values_from_op(op: SparseTensor | torch.Tensor) -> torch.Tensor:
            if isinstance(op, SparseTensor):
                coalesced = op.coalesce()
                return coalesced.storage.value()
            # Fallback: convert to torch_sparse SparseTensor
            sp = ensure_sparse_tensor(op)
            sp = sp.coalesce()
            return sp.storage.value()

        vals = torch.stack([_values_from_op(op) for op in ops], dim=0)  # (V, E)
        # Pairwise Frobenius norms across views (shared sparsity -> compare values)
        pairwise = torch.cdist(vals, vals, p=2)  # (V, V)
        V = pairwise.shape[0]
        for i in range(V):
            for j in range(i + 1, V):
                sims.append((i, j, float(pairwise[i, j].item())))
        return sims


class MultiViewScatter(nn.Module):
    """
    Apply scattering per diffusion view and combine outputs.

    combine_mode:
        - 'concat': concatenate along the wavelet dimension (default for multi-view)
        - 'sum': sum across views
        - 'mean': average across views
    """

    def __init__(self, combine_mode: Literal["concat", "sum", "mean"] = "concat") -> None:
        super().__init__()
        self.combine_mode = combine_mode

    def forward(
        self,
        x: torch.Tensor,
        diffusions: SparseTensor | torch.Tensor | Sequence[SparseTensor | torch.Tensor],
        *,
        diffusion_kwargs: Dict[str, Any],
        num_scattering_layers: int,
    ) -> torch.Tensor:
        if isinstance(diffusions, (SparseTensor, torch.Tensor)):
            diffusions = [diffusions]
        if len(diffusions) == 0:
            raise ValueError("MultiViewScatter received an empty diffusion list.")

        # Scatter per view
        coeffs_list = [
            vector_multiorder_scatter(
                x,
                op,
                diffusion_kwargs,
                num_scattering_layers,
            )
            for op in diffusions
        ]

        if len(coeffs_list) == 1:
            return coeffs_list[0]
        if self.combine_mode == "concat":
            return torch.cat(coeffs_list, dim=-1)
        if self.combine_mode == "sum":
            return torch.stack(coeffs_list, dim=0).sum(dim=0)
        if self.combine_mode == "mean":
            return torch.stack(coeffs_list, dim=0).mean(dim=0)

        raise ValueError(
            f"Unknown combine_mode '{self.combine_mode}'. "
            "Choose from {'concat', 'sum', 'mean', 'first'}."
        )


def materialize_learned_diffusion_ops(
    *,
    data: Data,
    normalized_weights: torch.Tensor,
    alpha: torch.Tensor,
    vector_dim: int,
    device: torch.device,
) -> List[SparseTensor]:
    """
    Convert learned transition probabilities into realized diffusion operators.

    Args:
        data: PyG Batch/Data object containing diffusion scaffolding attributes
            ('Q_unwt', 'Q_block_pairs', 'Q_block_edge_ids').
        normalized_weights: Tensor of shape (E,) or (E, num_views) holding
            per-edge transition probabilities.
        alpha: Scalar or tensor of shape (num_views,) encoding laziness.
        vector_dim: Dimensionality of the vector track (d).
        device: Target device for sparse tensors.
    Returns:
        List of torch_sparse SparseTensor diffusion operators, one per view.
    """
    Q_unwt = getattr(data, "Q_unwt", None)
    block_pairs = getattr(data, "Q_block_pairs", None)
    block_edge_ids = getattr(data, "Q_block_edge_ids", None)
    if Q_unwt is None or block_pairs is None or block_edge_ids is None:
        raise ValueError(
            "Data object is missing required attributes ('Q_unwt', 'Q_block_pairs', "
            "'Q_block_edge_ids') for learnable diffusion materialization."
        )
    if isinstance(Q_unwt, SparseTensor):
        Q_unwt = Q_unwt.to_torch_sparse_coo_tensor()
    Q_unwt = Q_unwt.coalesce().to(device)
    block_pairs = block_pairs.to(device)
    block_edge_ids = block_edge_ids.to(device)

    values = Q_unwt.values()
    indices = Q_unwt.indices()
    block_size = int(vector_dim) * int(vector_dim)
    num_blocks = values.numel() // block_size
    reshaped_vals = values.view(num_blocks, block_size)
    diag_mask = block_pairs[0] == block_pairs[1]
    off_diag_mask = ~diag_mask

    weights = normalized_weights.to(device)
    if weights.dim() == 1:
        weights = weights.unsqueeze(-1)
    elif weights.dim() != 2:
        raise ValueError(
            "normalized_weights must be 1D or 2D tensor; "
            f"received shape {normalized_weights.shape}."
        )
    num_views = weights.shape[-1]

    alpha_tensor = alpha.to(device)
    if alpha_tensor.dim() == 0:
        alpha_tensor = alpha_tensor.repeat(num_views)
    elif alpha_tensor.numel() == 1 and num_views > 1:
        alpha_tensor = alpha_tensor.repeat(num_views)
    elif alpha_tensor.numel() != num_views:
        raise ValueError(
            f"alpha (size={alpha_tensor.numel()}) does not match view count ({num_views})."
        )

    ops: List[SparseTensor] = []
    for view_idx in range(num_views):
        weighted_vals = reshaped_vals.clone()
        alpha_val = alpha_tensor[view_idx]
        weights_view = weights[:, view_idx]

        if diag_mask.any():
            weighted_vals[diag_mask] = weighted_vals[diag_mask] * alpha_val
        if off_diag_mask.any():
            off_idx = torch.nonzero(off_diag_mask, as_tuple=False).view(-1)
            edge_ids = block_edge_ids[off_idx].to(torch.long)
            num_edges = int(weights_view.shape[0])

            valid_mask = (edge_ids >= 0) & (edge_ids < num_edges)
            if valid_mask.any():
                edge_ids_valid = edge_ids[valid_mask]
                gathered = weights_view[edge_ids_valid].unsqueeze(-1)
                weighted_vals[off_idx[valid_mask]] = (
                    weighted_vals[off_idx[valid_mask]] * gathered
                )
            if (~valid_mask).any():
                # Zero out off-diagonal blocks that no longer have a corresponding edge
                weighted_vals[off_idx[~valid_mask]] = 0.0

        view_values = weighted_vals.view(-1)
        sparse = ensure_sparse_tensor(
            torch.sparse_coo_tensor(
                indices,
                view_values,
                size=Q_unwt.size(),
            )
        ).to(device)
        ops.append(sparse)
    return ops


def build_learnable_diffusion_ops(
    *,
    learnable_module: nn.Module,
    data: Data,
    edge_index: torch.Tensor,
    edge_features: torch.Tensor,
    vector_dim: int,
    device: torch.device,
    attention_mode: Optional[bool] = None,
) -> List[SparseTensor]:
    """
    Run a learnable diffusion module (single or multi-view) and return operators.

    Args:
        learnable_module: LearnableP or MultiLearnableP instance.
        data: Data/Batch carrying diffusion scaffolding tensors.
        edge_index: Edge indices (2, E).
        edge_features: Edge feature matrix (E, F).
        vector_dim: Vector channel dimension (d).
        device: Target device for computation.
        attention_mode: Optional override for LearnableP attention mode.
    Returns:
        List of SparseTensor diffusion operators (length equals num_views).
    """
    weights = learnable_module(
        num_nodes=int(data.num_nodes),
        edge_features=edge_features,
        edge_index=edge_index,
        device=device,
        attention_mode=attention_mode,
    )
    alpha = learnable_module.current_alpha(device=device)
    return materialize_learned_diffusion_ops(
        data=data,
        normalized_weights=weights,
        alpha=alpha,
        vector_dim=vector_dim,
        device=device,
    )

def batch_scatter(
    x: Batch,
    P_sparse: torch.Tensor | SparseTensor,
    scales_type: Literal['dyadic', 'custom'] = 'dyadic',
    diffusion_scales: Optional[torch.Tensor] = None,
    J: int = 4,
    include_lowpass: bool = True,
    filter_stack_dim: int = -1,
    rescale_filtrations: bool = False,
    # rescale_method: Literal['standardize', 'minmax'] = 'standardize',
    # vector_dim: Optional[int] = None
) -> torch.Tensor:
    r"""
    Computes diffusion wavelet filtrations on a (disconnected) 
    graph, using recursive sparse matrix multiplication of a 
    diffusion operator matrix P. This method skips computing 
    increasingly dense powers of P, which get denser with as 
    the power increases, by these steps:
    
    1. Compute $y_t = P^t x$ recursively via $y_t = P y_{t-1}$,
       (only using P, and not its powers, which grow denser).
    2. Subtract $y_{2^{j-1}} - y_{2^{j}}$ [dyadic scales]. 
        The result is $W_j x = (P^{2^{j-1}} - P^{2^j}) x$.
        (Thus, we never form the matrices P^t, t > 1, which get 
        denser with as the power increases.)

    Note that if 'diffusion_scales' is not None, using its custom
    scales will override the 'scales_type' parameter.
    
    Args:
        x: stacked node-by-channel (N, c) data matrix for a 
            disconnected  batch graph of a pytorch geometric 
            Batch object. 
        P_sparse: sparse diffusion operator matrix 
            for disconnected batch graph of a pytorch
            geometric Batch object (should be row-normalized
            for multiplication with x on the right).
        scales_type: 'dyadic' or 'custom' or None for fixed P^1.
        diffusion_scales: tensor of shape (n_scale_split_ts)
            or (n_channels, n_scale_split_ts) for calculating custom
            wavelet scales, containing the indices of ts 0...max($t$).
            Scales are constructed uniquely for each channel of x from
            $t$s with adjacent indices in rows of this tensor. If None,
            this function defaults to dyadic scales.
        J: max wavelet filter order, for dyadic scales. For example,
            $J = 4$ will give $T = 2^4 = 16$ max diffusion step.
        include_lowpass: whether to include the 
            'lowpass' filtration, $P^{2^J} x$.
        filter_stack_dim: new dimension in which to 
           stack Wjx (filtration) tensors.
        rescale_filtrations: whether to rescale the filtrations tensor. 
            Note that wavelet versus low-pass filtrations can have
            very different scales, so it can be useful to rescale. Note
            that this uses per-batch statistics, not global statistics, 
            or learnable as with nn.BatchNorm[n]d modules.
        rescale_method: [DEPRECATED]'standardize' (mean 0 std 1) or 'minmax' (onto
            interval [-1,1]).
        vector_dim: [DEPRECATED] if provided and the input corresponds to flattened
            vector features (shape (N*d, ...)), rescaling is applied
            independently for each coordinate dimension so that each
            component (e.g. x, y, z) keeps its own statistics.
    Returns:
        Dense tensor of shape (batch_total_nodes, n_channels,
        n_filtrations) = 'Ncj'.
    """
    # Ensure x is a 2D tensor and has same dtype as the diffusion operator
    if x.ndim == 1:
        x = x.unsqueeze(1)
    P_sparse = ensure_sparse_tensor(P_sparse)
    P_sparse = P_sparse.to(device=x.device)
    if P_sparse.dtype() != x.dtype:
        P_sparse = P_sparse.to(dtype=x.dtype)
    x = x.to(P_sparse.dtype())
    if isinstance(P_sparse, SparseTensor):
        nnu.raise_if_nonfinite_tensor(
            P_sparse.storage.value(),
            name="batch_scatter: P_sparse values",
        )
    nnu.raise_if_nonfinite_tensor(x, name="batch_scatter: x")

    # --- Helper function for shared powers ---
    def _get_Wjxs_from_shared_powers(
        x: torch.Tensor,
        P_sparse: torch.Tensor,
        powers_to_save: torch.Tensor,
        range_upper_lim: int,
        device: str
    ) -> torch.Tensor:
        Ptxs = [x.to(device)]
        Ptx = x.to(device)
        
        # calc P^t x for t \in 1...T, saving only needed P^txs
        # print('P_sparse.shape', P_sparse.shape)
        # print('Ptx.shape', Ptx.shape)
        for j in range(1, powers_to_save[-1] + 1):
            try:
                Ptx = matmul(P_sparse, Ptx)
                nnu.raise_if_nonfinite_tensor(Ptx, name=f"batch_scatter: P^t x (t={j})")
                if j in powers_to_save:
                    # print(f"j={j}")
                    # it's possible the same power is in a
                    # custom 'diffusion_scales' more than once
                    if diffusion_scales is not None:
                        j_ct = (powers_to_save == j).sum().item()
                        for _ in range(j_ct):
                            Ptxs.append(Ptx.to(device))
                    else:
                        Ptxs.append(Ptx.to(device))
            except Exception as e:
                print(f"Error in _get_Wjxs_from_shared_powers (j={j}): {e}")
                raise e

        # print('len(Ptxs):', len(Ptxs))
        Wjxs = [Ptxs[j - 1] - Ptxs[j] for j in range(1, range_upper_lim)] # J + 2)]
        if include_lowpass:
            Wjxs.append(Ptxs[-1])
        Wjxs = torch.stack(Wjxs, dim=filter_stack_dim).to(device)
        return Wjxs
    
    # --- Main function body ---
    device = x.device

    # First option: custom unique scales for each channel
    # print('get_Batch_Wjxs: diffusion_scales:', diffusion_scales)
    if (diffusion_scales is not None) \
    and (diffusion_scales != 'dyadic') \
    and (diffusion_scales.dim() == 2):
        Ptxs = [x.to(device)]
        Ptx = x.to(device)
        
        # calc P^t x for t \in 1...T, saving all powers of t
        custom_scales_max_t = int(2 ** J) # e.g. J = 5 -> 32
        for j in range(1, custom_scales_max_t + 1):
            Ptx = matmul(P_sparse, Ptx)
            Ptxs.append(Ptx.to_dense().to(device))

        # Compute filtrations ('Wjxs')
        # Note that filter (P^u - P^v)x = (P^u x) - (P^v x)
        # here indexes for (P^u x) and (P^v x) within 'Ptxs' for each
        # channel are adjacent entries in each channel's 't_is'
        Wjxs = torch.stack([
            torch.stack([
                # as of Nov 2024, bracket slicing doesn't work with sparse tensors
                # patch: entries of 'Ptxs' made dense above, when added to Ptxs
                Ptxs[t_is[t_i - 1]][:, c_i] - Ptxs[t_is[t_i]][:, c_i] \
                for t_i in range(1, len(t_is))
            ], dim=-1) \
            for c_i, t_is in enumerate(diffusion_scales)
        ], dim=1).to(device)
        
        '''
        Wjxs = [None] * diffusion_scales.shape[0]
        for c_i, t_is in enumerate(diffusion_scales):
            channel_Wjxs = [None] * (len(t_is) - 1)
            c_i_tensor = torch.tensor([c_i]).to(device)
            
            for t_i in range(1, len(t_is)):
                Pu = torch.index_select(Ptxs[t_is[t_i - 1]], 1, c_i_tensor)
                Pv = torch.index_select(Ptxs[t_is[t_i]], 1, c_i_tensor)
                channel_Wjxs[t_i - 1] = (Pu - Pv).squeeze()
                # print('channel_Wjxs.shape:', channel_Wjxs[t_i - 1].shape)
                
            channel_Wjxs = torch.stack(channel_Wjxs, dim=-1)
            Wjxs[c_i] = channel_Wjxs
        Wjxs = torch.stack(Wjxs, dim=1)
        print('Wjxs.shape:', Wjxs.shape)
        '''
        
        # lowpass = P^T x, for all channels
        if include_lowpass:
            # print('Ptxs[-1].shape:', Ptxs[-1].shape)
            Wjxs = torch.concatenate(
                (Wjxs, Ptxs[-1].unsqueeze(dim=-1).to(device)), 
                dim=-1
            )
        # Wjxs shape (N, n_channels, n_filters)
        # print('Wjxs.shape:', Wjxs.shape)

    # Second option: one set of custom scales shared by all channels
    # (custom is priority if passed)
    elif (diffusion_scales is not None) \
    and (diffusion_scales != 'dyadic') \
    and (diffusion_scales.dim() == 1):
        powers_to_save = diffusion_scales
        range_upper_lim = diffusion_scales.shape[0]
        # print('shared_powers_to_save:', shared_powers_to_save)
        # print('range_upper_lim:', range_upper_lim)
        Wjxs = _get_Wjxs_from_shared_powers(
            x=x,
            P_sparse=P_sparse,
            powers_to_save=powers_to_save,
            range_upper_lim=range_upper_lim,
            device=device
        )

    # Third option: dyadic scales shared by all channels
    elif (scales_type == 'dyadic'):
        powers_to_save = 2 ** torch.arange(J + 1)
        range_upper_lim = J + 2
        Wjxs = _get_Wjxs_from_shared_powers(
            x=x,
            P_sparse=P_sparse,
            powers_to_save=powers_to_save,
            range_upper_lim=range_upper_lim,
            device=device
        )

    # Fourth option: fixed P^1
    elif (diffusion_scales is None) and (scales_type is None):
        Ptx = x.to(device)
        Ptx = matmul(P_sparse, Ptx)
        Wjxs = Ptx.unsqueeze(dim=-1).to(device)
    else:
        raise NotImplementedError(
            f"No method implemented for scales_type={scales_type}"
        )
        
    # -----------------------------------------------------
    # Vector features: rescale per coordinate + filter
    #   Wjxs shape (N*d, C, F) -> (N, d, C, F)
    #   Reduce dims = (0, 2)  (nodes, channels)
    # Scalars (or unknown d): reduce dims = (0, 1)
    # -----------------------------------------------------

    if rescale_filtrations:
        raise NotImplementedError(
            "Rescaling filtrations within 'get_Batch_Wjxs' has been deprecated."
        )
        '''
        # --- Rescaling helper functions ---
        def _standardize(
            t: torch.Tensor, 
            dims: Tuple[int] = (0, 1)
        ) -> torch.Tensor:
            """
            Standardize tensor *per filter* (dim=-1) across the 
            given dims.
            """
            mean_val = t.mean(dim=dims, keepdim=True)
            std_val = t.std(dim=dims, keepdim=True)
            std_val = torch.where(std_val == 0, torch.ones_like(std_val), std_val)
            return (t - mean_val) / std_val

        def _minmax_scale(
            t: torch.Tensor, 
            dims: Tuple[int] = (0, 1),
            center_at_zero: bool = True
        ) -> torch.Tensor:
            """
            Min-max scale tensor *per filter* (dim=-1) across the 
            given dims. Center at zero if 'center_at_zero' is True.
            """
            min_val = t.min(dim=dims, keepdim=True).values
            max_val = t.max(dim=dims, keepdim=True).values
            range_val = max_val - min_val
            range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
            t = (t - min_val) / range_val  # -> [0,1]
            if center_at_zero:
                t = 2. * t - 1.            # -> [-1,1]
            return t

        # Vector case: rescale per coordinates and filters
        if (vector_dim is not None) and (Wjxs.shape[0] % vector_dim == 0):
            print("WARNING: rescaling vector features per-coordinate is probably not what you want...")
            N = Wjxs.shape[0] // vector_dim
            C, F = Wjxs.shape[1], Wjxs.shape[2]
            Wjxs_reshaped = Wjxs.view(N, vector_dim, C, F)
            reduce_dims = (0, 2)  # nodes, channels

            if rescale_method == 'standardize':
                Wjxs_reshaped = _standardize(Wjxs_reshaped, dims=reduce_dims)
            elif rescale_method == 'minmax':
                Wjxs_reshaped = _minmax_scale(Wjxs_reshaped, dims=reduce_dims)
            else:
                raise ValueError(
                    f"Unknown rescale option '{rescale_method}'. Choose 'standardize' or 'minmax'."
                )

            Wjxs = Wjxs_reshaped.view(N * vector_dim, C, F)

        else:
            # Scalar case: reduce over nodes and channels per filter
            if rescale_method == 'standardize':
                Wjxs = _standardize(Wjxs, dims=(0, 1))
            elif rescale_method == 'minmax':
                Wjxs = _minmax_scale(Wjxs, dims=(0, 1))
            else:
                raise ValueError(
                    f"Unknown rescale option '{rescale_method}'. Choose 'standardize' or 'minmax'."
                )
    '''

    return Wjxs


def vector_multiorder_scatter(
    x_v: torch.Tensor,
    diffusion_op: torch.Tensor,
    diffusion_kwargs: Dict[str, Any],
    num_scattering_layers: int = 1,
) -> torch.Tensor:
    """
    Apply 0th-, 1st-, and optional 2nd-order scattering to vector signals,
    using batch_scatter. Returns coefficients of shape (N, d, W_total).

    Args:
        x_v: vector features of shape (N, d)
        diffusion_op: vector diffusion operator of shape (Nd, Nd)
        diffusion_kwargs: keyword arguments for batch_scatter
        num_scattering_layers: number of scattering layers to apply
    Returns:
        coefficients of shape (N, d, W_total)
        where W_total is the total number of wavelet coefficients.
    """
    N, d = x_v.shape
    flat = x_v.reshape(N * d, 1)

    # Allow only batch_scatter kwargs that are supported; drop extras like 'scattering_k'
    _batch_scatter_allowed = {
        "scales_type",
        "diffusion_scales",
        "J",
        "include_lowpass",
        "filter_stack_dim",
        "rescale_filtrations",
    }
    scatter_kwargs = {
        key: val for key, val in diffusion_kwargs.items() if key in _batch_scatter_allowed
    }

    # First-order scattering
    diffusion_sparse = ensure_sparse_tensor(diffusion_op).to(device=x_v.device)
    W1 = batch_scatter(
        x=flat,
        P_sparse=diffusion_sparse,
        **scatter_kwargs,
    )  # (N*d, 1, W1)

    coeffs = [flat.unsqueeze(-1), W1]

    if num_scattering_layers > 1:
        nW = int(W1.shape[-1])
        if nW > 1:
            x_second = W1.squeeze(1)  # (N*d, W1)
            W2raw = batch_scatter(
                x=x_second,
                P_sparse=diffusion_sparse,
                **scatter_kwargs,
            )  # (N*d, W1, W1)
            Nd = x_second.shape[0]
            W2raw = W2raw.view(Nd, nW, -1)  # (N*d, W1, W1)
            W2 = subset_second_order_wavelets(
                W2raw,
                feature_type='vector',
            )  # (N*d, 1, W2)
            coeffs.append(W2)

    W_tot = torch.cat(coeffs, dim=-1)  # (N*d, 1, W_total)
    return W_tot.view(N, d, -1)  # (N, d, W_total)


class LearnableMahalanobisTopK(nn.Module):
    """
    Learn a Mahalanobis-like neighbor scoring and prune to top-k edges.

    Given node features x (N, d) and a candidate edge_index (2, E), this layer:
        1) Projects x with a learned linear map (no bias).
        2) Scores neighbors with a diagonal Mahalanobis metric.
        3) Applies a per-node softmax to get attention weights.
        4) Selects top-k neighbors per source node; forward uses truncated,
           renormalized weights while gradients flow through the dense softmax
           (straight-through style).
    Returns the pruned edge_index and the associated attention weights.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: Optional[int] = None,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0 (got {temperature}).")
        self.in_dim = int(in_dim)
        self.proj_dim = int(proj_dim) if proj_dim is not None else int(in_dim)
        self.eps = float(eps)
        self.proj = nn.Linear(self.in_dim, self.proj_dim, bias=False)
        # Positive diagonal metric via softplus
        self.log_diag = nn.Parameter(torch.zeros(self.proj_dim))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(float(temperature))))

    def current_temperature(self, device: Optional[torch.device] = None) -> torch.Tensor:
        temp = torch.exp(self.log_temperature)
        if device is not None and temp.device != device:
            temp = temp.to(device)
        return temp

    def _mah_dist_scores(self, x_proj: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        diff = x_proj[row] - x_proj[col]  # (E, proj_dim)
        diag = F.softplus(self.log_diag) + self.eps  # ensure positivity
        dist_sq = (diff * diag) * diff
        dist_sq = dist_sq.sum(dim=-1)  # (E,)
        return -dist_sq  # higher is closer

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        topk: int,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError(f"x must be 2D (N, d); got shape {x.shape}.")
        if edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape (2, E); got {edge_index.shape}.")
        if topk <= 0:
            raise ValueError(f"topk must be positive; got {topk}.")

        device = x.device
        edge_index = edge_index.to(device)
        x_proj = self.proj(x)
        scores = self._mah_dist_scores(x_proj, edge_index)

        temp = float(temperature) if temperature is not None else float(self.current_temperature(device))
        scaled_scores = scores / temp

        row = edge_index[0].to(torch.long)
        num_nodes = int(x.shape[0])

        # Softmax per source node (log-sum-exp for stability)
        logsumexp = scatter_logsumexp(scaled_scores, row, dim=0, dim_size=num_nodes)
        base_weights = torch.exp(scaled_scores - logsumexp[row])  # (E,), sums to 1 per row

        # Top-k per node (using row-grouped indices)
        counts = torch.bincount(row, minlength=num_nodes)
        order = torch.argsort(row)  # group by source node
        sorted_scores = scaled_scores[order]

        selected_indices: List[torch.Tensor] = []
        offset = 0
        for count in counts.tolist():
            if count == 0:
                continue
            segment_scores = sorted_scores[offset : offset + count]
            k = min(int(topk), int(count))
            local_topk = torch.topk(segment_scores, k=k, largest=True).indices
            selected_indices.append(order[offset + local_topk])
            offset += count
        if selected_indices:
            selected_indices = torch.cat(selected_indices, dim=0)
        else:
            # No edges; return empty tensors
            empty = edge_index.new_zeros((2, 0))
            return empty, empty.new_zeros((0,), dtype=x.dtype)

        topk_mask = torch.zeros_like(base_weights, dtype=torch.bool)
        topk_mask[selected_indices] = True

        selected_weights = base_weights * topk_mask.to(base_weights.dtype)
        row_sums = scatter_sum(selected_weights, row, dim=0, dim_size=num_nodes)
        row_sums = row_sums[row].clamp_min(self.eps)
        renorm = selected_weights / row_sums

        # Straight-through: forward uses renormed top-k; gradients from base softmax
        weights_st = base_weights + (renorm - base_weights).detach()

        pruned_edge_index = edge_index[:, topk_mask]
        pruned_edge_weight = weights_st[topk_mask]
        return pruned_edge_index, pruned_edge_weight