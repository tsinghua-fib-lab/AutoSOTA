"""Optimized transformer layers for Graphormer with sparse attention support.

This module provides memory-efficient transformer layers with:
- Sparse attention using scatter/gather operations (fully vectorized)
- Cached RBF features across layers
- Segment-wise softmax for grouped edges
- Support for both dense and sparse modes
- Optional flash_attn package support for additional speedups
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .nn_utils import RotaryPositionalEncoding

# Try to import flash_attn package for additional optimizations
try:
    from flash_attn import flash_attn_func  # type: ignore[import-untyped]

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class BlockSequential(nn.Sequential):
    """Sequential container with keyword argument support.

    Allows passing additional arguments (like edge indices) through the sequence.
    """

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        """Forward pass with keyword arguments.

        Args:
            input: Input tensor
            **kwargs: Additional arguments (dist, edge_index, edge_dist, edge_features, dist_features)

        Returns:
            Output tensor
        """
        for module in self:
            # Pass kwargs only to Block and GraphormerBlock modules
            if isinstance(module, (Block, GraphormerBlock)):
                input = module(input, **kwargs)
            else:
                input = module(input)
        return input


class SelfAttention(nn.Module):
    """Multi-head self-attention with optional Flash Attention support."""

    def __init__(self, nembed: int, nhead: int, dropout: float = 0.0) -> None:
        """Initialize self-attention layer.

        Args:
            nembed: Embedding dimension
            nhead: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.c_attn = nn.Linear(nembed, nembed * 3, bias=False)  # Combined Q, K, V projection
        self.proj = nn.Linear(nembed, nembed, bias=False)  # Output projection
        self.n_embed = nembed
        self.n_head = nhead
        self.dropout = dropout
        self.flash = hasattr(F, "scaled_dot_product_attention")
        self.has_flash_attn_pkg = HAS_FLASH_ATTN
        # Store reference to avoid linter issues
        self._scaled_dot_product_attention = getattr(F, "scaled_dot_product_attention", None)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        rotary_pos_emb: RotaryPositionalEncoding | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input [B, T, C]
            attn_mask: Optional attention mask
            rotary_pos_emb: Optional rotary positional encoding

        Returns:
            Output [B, T, C]
        """
        B, T, C = x.shape
        head_dim = C // self.n_head

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

        # Reshape and apply rotary embeddings if provided
        q = q.view(B, T, self.n_head, head_dim)
        k = k.view(B, T, self.n_head, head_dim)
        v = v.view(B, T, self.n_head, head_dim)

        if rotary_pos_emb is not None:
            # Check if rotary_pos_emb expects positions or fixed mode
            # If default/fixed mode, no need to pass extra args
            q, k = rotary_pos_emb(q).transpose(1, 2), rotary_pos_emb(k).transpose(1, 2)
        else:
            q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = v.transpose(1, 2)  # [B, H, T, D]

        # Attention computation
        # Use flash_attn package if available and no custom mask (it's faster)
        if self.has_flash_attn_pkg and attn_mask is None:
            # flash_attn expects [B, T, H, D] format
            q_fa = q.transpose(1, 2).contiguous()  # [B, T, H, D]
            k_fa = k.transpose(1, 2).contiguous()
            v_fa = v.transpose(1, 2).contiguous()
            y = flash_attn_func(q_fa, k_fa, v_fa, dropout_p=self.dropout if self.training else 0, causal=False)
            y = y.transpose(1, 2)  # Back to [B, H, T, D]
        elif self.flash and self._scaled_dot_product_attention is not None:
            # PyTorch's built-in scaled_dot_product_attention
            y = self._scaled_dot_product_attention(  # type: ignore[misc]
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
            if attn_mask is not None:
                att = att.masked_fill(attn_mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    """Standard transformer block."""

    def __init__(self, nembed: int, nhead: int, hidden_size: int) -> None:
        """Initialize transformer block.

        Args:
            nembed: Embedding dimension
            nhead: Number of attention heads
            hidden_size: MLP hidden size
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(nembed, eps=1e-6)
        self.ln2 = nn.LayerNorm(nembed, eps=1e-6)
        self.attn = SelfAttention(nembed, nhead, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(nembed, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, nembed),
        )

    def forward(self, x: Tensor, rotary_pos_emb: RotaryPositionalEncoding | None = None) -> Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input [B, T, C]
            rotary_pos_emb: Optional rotary positional encoding

        Returns:
            Output [B, T, C]
        """
        x = x + self.attn(self.ln1(x), rotary_pos_emb=rotary_pos_emb)
        x = x + self.mlp(self.ln2(x))
        return x


class GraphormerAttention(nn.Module):
    """Attention with spatial bias for Graphormer.

    Supports both dense (O(N²)) and sparse (O(E)) attention modes with RBF distance encoding.
    """

    def __init__(
        self,
        nembed: int,
        nhead: int,
        dropout: float = 0.0,
        num_kernels: int = 128,
        dist_feature_extractor: str = "rbf",
        shared_dist_kernels: nn.Parameter | None = None,
        shared_dist_scale: nn.Parameter | None = None,
    ):
        """Initialize Graphormer attention.

        Args:
            nembed: Embedding dimension
            nhead: Number of attention heads
            dropout: Dropout rate
            num_kernels: Number of RBF kernels
            dist_feature_extractor: Distance encoding type ("rbf")
            shared_dist_kernels: Optional shared RBF kernel parameters across layers
            shared_dist_scale: Optional shared RBF scale parameter across layers
        """
        super().__init__()
        self.self_attn = SelfAttention(nembed, nhead, dropout)
        self.nembed = nembed
        self.nhead = nhead
        self.num_kernels = num_kernels
        self.dist_feature_extractor = dist_feature_extractor

        if dist_feature_extractor == "rbf":
            # Use shared parameters if provided, otherwise create new ones
            if shared_dist_kernels is not None and shared_dist_scale is not None:
                self.dist_kernels = shared_dist_kernels
                self.dist_scale = shared_dist_scale
            else:
                self.dist_kernels = nn.Parameter(torch.randn(num_kernels))
                self.dist_scale = nn.Parameter(torch.ones(1))
            self.dist_proj = nn.Linear(num_kernels, nhead)
        else:
            raise NotImplementedError(f"Extractor '{dist_feature_extractor}' not implemented")

    def _compute_dist_features(self, dist: Tensor) -> Tensor:
        """Compute RBF features for distances (dense mode).

        Args:
            dist: Distances [B, N, N]

        Returns:
            RBF features [B, N, N, K]
        """
        if self.dist_feature_extractor == "rbf":
            dist_expanded = dist.unsqueeze(-1)  # [B, N, N, 1]
            # Optimized: use in-place operations where possible
            diff = dist_expanded - self.dist_kernels  # [B, N, N, K]
            return torch.exp(-(diff * diff) / self.dist_scale)
        return None

    def _compute_dist_features_sparse(self, edge_dist: Tensor) -> Tensor:
        """Compute RBF features for edge distances (sparse mode).

        Args:
            edge_dist: Edge distances [B, E]

        Returns:
            RBF features [B, E, K]
        """
        if self.dist_feature_extractor == "rbf":
            dist_expanded = edge_dist.unsqueeze(-1)  # [B, E, 1]
            # Optimized: use in-place operations where possible
            diff = dist_expanded - self.dist_kernels  # [B, E, K]
            return torch.exp(-(diff * diff) / self.dist_scale)
        return None

    def forward(
        self,
        x: Tensor,
        dist: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_dist: Tensor | None = None,
        edge_features: Tensor | None = None,
        dist_features: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass supporting both dense and sparse modes.

        Args:
            x: Input [B, N, C]
            dist: Distance matrix [B, N, N] (dense mode)
            edge_index: Edge indices [2, E] (sparse mode)
            edge_dist: Edge distances [B, E] (sparse mode)
            edge_features: Pre-computed RBF features [B, E, K] (sparse mode, optional)
            dist_features: Pre-computed RBF features [B, N, N, K] (dense mode, optional)
            attn_mask: Optional attention mask

        Returns:
            Output [B, N, C]
        """
        if (edge_index is not None) and (edge_dist is not None):
            return self._forward_sparse(x, edge_index, edge_dist, edge_features)
        else:
            return self._forward_dense(x, dist, dist_features, attn_mask)

    def _forward_dense(
        self,
        x: Tensor,
        dist: Tensor,
        dist_features: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Dense attention computation (O(N²)).

        Args:
            x: Input [B, N, C]
            dist: Distance matrix [B, N, N]
            dist_features: Pre-computed RBF features [B, N, N, K] (optional, cached)
            attn_mask: Optional mask

        Returns:
            Output [B, N, C]
        """
        B, N, C = x.shape
        head_dim = C // self.nhead

        # Compute Q, K, V and reshape for multi-head attention
        q, k, v = self.self_attn.c_attn(x).split(self.nembed, dim=2)
        q = q.view(B, N, self.nhead, head_dim).transpose(1, 2)  # [B, H, N, D]
        k = k.view(B, N, self.nhead, head_dim).transpose(1, 2)
        v = v.view(B, N, self.nhead, head_dim).transpose(1, 2)

        # Compute attention scores with distance bias
        # Optimized: use matmul with better memory layout
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # Use cached dist_features if provided, otherwise compute
        if dist_features is None:
            dist_features = self._compute_dist_features(dist)

        if dist_features is not None:
            attn_bias = self.dist_proj(dist_features).permute(0, 3, 1, 2)  # [B, H, N, N]
            scores = scores + attn_bias

        # Apply softmax and compute output
        # Use scaled_dot_product_attention if no bias (faster), otherwise manual
        if dist_features is None and attn_mask is None and self.self_attn.flash:
            # Can use flash attention when no custom bias
            scaled_attn = getattr(F, "scaled_dot_product_attention", None)
            if scaled_attn is not None:
                y = scaled_attn(q, k, v, dropout_p=0.0, is_causal=False)  # type: ignore[misc]
            else:
                att = F.softmax(scores, dim=-1)
                if attn_mask is not None:
                    att = att.masked_fill(attn_mask == 0, 0.0)
                y = torch.matmul(att, v)
        else:
            att = F.softmax(scores, dim=-1)
            if attn_mask is not None:
                att = att.masked_fill(attn_mask == 0, 0.0)
            y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, N, C)
        return self.self_attn.proj(y)

    def _forward_sparse(
        self, x: Tensor, edge_index: Tensor, edge_dist: Tensor, edge_features: Tensor | None = None
    ) -> Tensor:
        """Sparse attention computation (O(E), fully vectorized).

        Args:
            x: Input [B, N, C]
            edge_index: Edge indices [2, E]
            edge_dist: Edge distances [B, E]
            edge_features: Pre-computed RBF features [B, E, K]

        Returns:
            Output [B, N, C]
        """
        B, N, C = x.shape
        head_dim = C // self.nhead

        # Compute Q, K, V and reshape [B, H, N, D]
        q, k, v = self.self_attn.c_attn(x).split(self.nembed, dim=2)
        q = q.view(B, N, self.nhead, head_dim).transpose(1, 2)
        k = k.view(B, N, self.nhead, head_dim).transpose(1, 2)
        v = v.view(B, N, self.nhead, head_dim).transpose(1, 2)

        # Compute or use cached RBF features
        if edge_features is None:
            edge_features = self._compute_dist_features_sparse(edge_dist)
        edge_bias = self.dist_proj(edge_features).transpose(1, 2)  # [B, H, E]

        # Gather Q, K, V along edges
        src, dst = edge_index
        q_src = q[:, :, src, :]  # [B, H, E, D]
        k_dst = k[:, :, dst, :]
        v_dst = v[:, :, dst, :]

        # Compute attention scores for edges
        # Optimized: use einsum or bmm for better performance
        scores = torch.einsum("bhed,bhed->bhe", q_src, k_dst) / math.sqrt(head_dim) + edge_bias  # [B, H, E]

        # Apply segment-wise softmax
        attn_weights = self._segment_softmax(scores, src, N)  # [B, H, E]

        # Compute and aggregate messages
        messages = attn_weights.unsqueeze(-1) * v_dst  # [B, H, E, D]
        output = torch.zeros(
            B, self.nhead, N, head_dim, dtype=messages.dtype, device=messages.device
        )  # [B, H, N, D]
        src_expanded = src.view(1, 1, -1, 1).expand(B, self.nhead, -1, head_dim)
        output = output.scatter_add(2, src_expanded, messages)  # [B, H, N, D]

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, C]
        return self.self_attn.proj(output)

    def _segment_softmax(self, scores: Tensor, segment_ids: Tensor, num_segments: int) -> Tensor:
        """Apply softmax grouped by segment IDs (fully vectorized).

        Computes softmax over edges grouped by source node using scatter operations.

        Args:
            scores: Attention scores [B, H, E]
            segment_ids: Segment ID for each edge [E]
            num_segments: Total number of segments (N)

        Returns:
            Normalized attention weights [B, H, E]
        """
        B, H, _E = scores.shape

        # Expand segment IDs for broadcasting
        seg_ids = segment_ids.view(1, 1, -1).expand(B, H, -1)

        # Compute max per segment for numerical stability
        max_vals = torch.full((B, H, num_segments), -1e9, dtype=scores.dtype, device=scores.device)
        max_vals = max_vals.scatter_reduce(2, seg_ids, scores, reduce="amax", include_self=False)
        max_per_edge = torch.gather(max_vals, 2, seg_ids)

        # Compute exp(score - max)
        exp_scores = torch.exp(scores - max_per_edge)

        # Sum exp scores per segment
        exp_sums = torch.zeros_like(max_vals)
        exp_sums = exp_sums.scatter_add(2, seg_ids, exp_scores)
        exp_sum_per_edge = torch.gather(exp_sums, 2, seg_ids)

        # Compute final softmax with numerical stability
        return exp_scores / torch.clamp(exp_sum_per_edge, min=1e-12)


class GraphormerBlock(nn.Module):
    """Graphormer block with edge features and spatial attention."""

    def __init__(
        self,
        nembed: int,
        nhead: int,
        hidden_size: int,
        num_kernels: int = 128,
        dist_feature_extractor: str = "rbf",
        shared_dist_kernels: nn.Parameter | None = None,
        shared_dist_scale: nn.Parameter | None = None,
    ):
        """Initialize Graphormer block.

        Args:
            nembed: Embedding dimension
            nhead: Number of attention heads
            hidden_size: MLP hidden size
            num_kernels: Number of RBF kernels
            dist_feature_extractor: Distance encoding type
            shared_dist_kernels: Optional shared RBF kernel parameters across layers
            shared_dist_scale: Optional shared RBF scale parameter across layers
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(nembed)
        self.ln2 = nn.LayerNorm(nembed)
        self.attn = GraphormerAttention(
            nembed,
            nhead,
            num_kernels=num_kernels,
            dist_feature_extractor=dist_feature_extractor,
            shared_dist_kernels=shared_dist_kernels,
            shared_dist_scale=shared_dist_scale,
        )
        self.mlp = nn.Sequential(
            nn.Linear(nembed, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, nembed),
        )

    def forward(
        self,
        x: Tensor,
        dist: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_dist: Tensor | None = None,
        edge_features: Tensor | None = None,
        dist_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass supporting both dense and sparse modes.

        Args:
            x: Node features [B, N, C]
            dist: Distance matrix [B, N, N] (dense mode)
            edge_index: Edge indices [2, E] (sparse mode)
            edge_dist: Edge distances [B, E] (sparse mode)
            edge_features: Pre-computed RBF features [B, E, K] (sparse mode, optional)
            dist_features: Pre-computed RBF features [B, N, N, K] (dense mode, optional)

        Returns:
            Output [B, N, C]
        """
        if (edge_index is not None) and (edge_dist is not None):
            x = self._forward_sparse(x, edge_index, edge_dist, edge_features)
        else:
            x = self._forward_dense(x, dist, dist_features)

        return x + self.mlp(self.ln2(x))

    def _forward_sparse(
        self, x: Tensor, edge_index: Tensor, edge_dist: Tensor, edge_features: Tensor | None
    ) -> Tensor:
        """Sparse mode forward pass.

        Args:
            x: Node features [B, N, C]
            edge_index: Edge indices [2, E]
            edge_dist: Edge distances [B, E]
            edge_features: Pre-computed RBF features [B, E, K]

        Returns:
            Output after attention [B, N, C]
        """
        # Apply attention with edge features as bias
        return x + self.attn(
            self.ln1(x), edge_index=edge_index, edge_dist=edge_dist, edge_features=edge_features
        )

    def _forward_dense(
        self, x: Tensor, dist: Tensor, dist_features: Tensor | None = None
    ) -> Tensor:
        """Dense mode forward pass.

        Args:
            x: Node features [B, N, C]
            dist: Distance matrix [B, N, N]
            dist_features: Pre-computed RBF features [B, N, N, K] (optional, cached)

        Returns:
            Output after attention [B, N, C]
        """
        # Apply attention with distance features as bias (cached if provided)
        return x + self.attn(self.ln1(x), dist=dist, dist_features=dist_features)
