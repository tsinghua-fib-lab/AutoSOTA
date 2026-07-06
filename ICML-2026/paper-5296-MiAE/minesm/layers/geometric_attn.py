from math import sqrt

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class GeometricReasoningFlashImpl(nn.Module):
    """
    FlashAttention-style reimplementation of GeometricReasoningOriginalImpl.

    Keeps the same API and numerics (up to floating point), but uses
    torch.nn.functional.scaled_dot_product_attention for the softmax attention.
    """

    def __init__(
        self,
        c_s: int,
        v_heads: int,
        num_vector_messages: int = 1,
        mask_and_zero_frameless: bool = True,
        divide_residual_by_depth: bool = False,  # unused; kept for API parity
        bias: bool = False,
    ):
        super().__init__()
        self.c_s = c_s
        self.v_heads = v_heads
        self.num_vector_messages = num_vector_messages
        self.mask_and_zero_frameless = mask_and_zero_frameless

        self.s_norm = nn.LayerNorm(c_s, bias=bias)

        # 2*(q,k) for rotation (3D each) + 2*(q,k) for distance (3D each),
        # plus values: heads * num_vector_messages * 3D
        dim_proj = 4 * self.v_heads * 3 + self.v_heads * 3 * self.num_vector_messages
        self.proj = nn.Linear(c_s, dim_proj, bias=bias)

        channels_out = self.v_heads * 3 * self.num_vector_messages
        self.out_proj = nn.Linear(channels_out, c_s, bias=bias)

        # Per-head positive scales (via softplus) for distance and rotation
        self.distance_scale_per_head = nn.Parameter(torch.zeros((self.v_heads)))
        self.rotation_scale_per_head = nn.Parameter(torch.zeros((self.v_heads)))

    def _pairwise_distance_bias(self, q_dist, k_dist, w_dist):
        """
        Compute per-head additive bias:  - w_dist * ||q_dist - k_dist|| / sqrt(3)

        q_dist: (B, H, S_q, 3)
        k_dist: (B, H, S_k, 3)
        w_dist: (B, H, 1, 1)  broadcastable
        returns: (B, H, S_q, S_k)
        """
        # ||a-b|| = sqrt(||a||^2 + ||b||^2 - 2 a·b), keep clamp for numerical stability
        # a·b term:
        # (B,H,S_q,3) @ (B,H,3,S_k) -> (B,H,S_q,S_k)
        ab = torch.matmul(q_dist, k_dist.transpose(-1, -2))  # dot along last dim (3)

        # ||a||^2 and ||b||^2 terms, then broadcast
        qa2 = (q_dist * q_dist).sum(dim=-1, keepdim=True)  # (B,H,S_q,1)
        kb2 = (
            (k_dist * k_dist).sum(dim=-1, keepdim=True).transpose(-1, -2)
        )  # (B,H,1,S_k)

        # distance matrix
        dist2 = (qa2 + kb2 - 2.0 * ab).clamp_min(0.0)
        dist = torch.sqrt(dist2 + 1e-10)  # small epsilon for stability
        dist = dist / sqrt(3.0)

        return -w_dist * dist

    def forward(self, s, attention_mask, affine, affine_mask, sequence_id, chain_id):
        """
        s:            (B, S, c_s)
        affine:       has .rot (rotation) and .apply/.invert() that operate on (...,3) vectors
        affine_mask:  (B, S) boolean
        sequence_id:  (B, S) int64 (None -> zeros)
        chain_id:     (B, S) int64
        """
        B, S, _ = s.shape
        if sequence_id is None:
            sequence_id = torch.zeros_like(s[..., 0], dtype=torch.int64)

        # Build cross-sequence / padding / chain masks -> additive bias
        # attn_bias = (sequence_id.unsqueeze(-1) == sequence_id.unsqueeze(-2))  # (B,S,S)
        # attn_bias = attn_bias.unsqueeze(1).float()  # (B,1,S,S)
        attn_bias = torch.zeros(
            attention_mask.shape[0],
            1,
            attention_mask.shape[1],
            attention_mask.shape[1],
            device=s.device,
        )  # [B * L, 1, S, S]
        # mask padding on keys
        attn_bias = attn_bias.masked_fill(
            ~(affine_mask & attention_mask)[:, None, None, :],
            torch.finfo(attn_bias.dtype).min,
        )
        # prevent cross-chain attention
        chain_id_mask = chain_id.unsqueeze(1) != chain_id.unsqueeze(2)  # (B,S,S)
        attn_bias = attn_bias.masked_fill(
            chain_id_mask.unsqueeze(1), torch.finfo(s.dtype).min
        )

        ns = self.s_norm(s)
        vec_rot, vec_dist = self.proj(ns).split(
            [
                self.v_heads * 2 * 3 + self.v_heads * 3 * self.num_vector_messages,
                self.v_heads * 2 * 3,
            ],
            dim=-1,
        )

        # Rotate queries/keys/values for rotation term & message vectors
        # Values are rotated only (no translation), matching the original.
        query_rot, key_rot, value = (
            affine.rot[..., None]
            .apply(rearrange(vec_rot, "... (h c) -> ... h c", c=3))
            .split(
                [self.v_heads, self.v_heads, self.v_heads * self.num_vector_messages],
                dim=-2,
            )
        )
        # Rotate+Translate for distance term
        query_dist, key_dist = (
            affine[..., None]
            .apply(rearrange(vec_dist, "... (h c) -> ... h c", c=3))
            .chunk(2, dim=-2)
        )

        # Shapes for SDPA
        # Rotation Q/K head_dim=3 (SDPA scales by 1/sqrt(3) automatically)
        q_rot = rearrange(query_rot, "b s h d -> b h s d")
        k_rot = rearrange(key_rot, "b s h d -> b h s d")
        # Values: concatenate per-message 3D vectors
        v = rearrange(value, "b s (h m) d -> b h s (m d)", m=self.num_vector_messages)

        # Distance vectors
        q_dist = rearrange(query_dist, "b s h d -> b h s d")
        k_dist = rearrange(key_dist, "b s h d -> b h s d")

        # Per-head positive weights via softplus, shape (B,H,1,1) after broadcast
        # (weights are per head and same across batch; we expand to B for convenience)
        w_dist = F.softplus(self.distance_scale_per_head)  # (H,)
        w_rot = F.softplus(self.rotation_scale_per_head)  # (H,)
        # Expand to (B,H,1,1) for bias math, and (B,H,1,1) is broadcastable to SDPA
        w_dist_b = w_dist.view(1, -1, 1, 1).expand(B, -1, 1, 1)
        # For rotation weight inside QK^T, multiply BOTH Q and K by sqrt(w_rot)
        wrot_sqrt = torch.sqrt(w_rot + 1e-10).view(1, -1, 1, 1)  # (1,H,1,1)

        q = q_rot * wrot_sqrt
        k = k_rot * wrot_sqrt

        # Build additive bias: distance penalty + original attention bias
        dist_bias = self._pairwise_distance_bias(q_dist, k_dist, w_dist_b)  # (B,H,S,S)

        # Crop attn_bias if needed (bin-packing cases)
        s_q = q.size(-2)
        s_k = k.size(-2)
        _s_q = max(0, attn_bias.size(2) - s_q)
        _s_k = max(0, attn_bias.size(3) - s_k)
        attn_bias_cropped = attn_bias[:, :, _s_q:, _s_k:]  # (B,1 or H,S,S) -> (B,1,S,S)

        # Combine: both are additive pre-softmax logits
        # Broadcast (B,1,S,S) over heads to (B,H,S,S)
        combined_bias = dist_bias + attn_bias_cropped

        # SDPA: uses FlashAttention kernels when available (dtype/shape permitting)
        # We pass `combined_bias` as additive mask.
        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            ]
        ):
            attn_out = F.scaled_dot_product_attention(
                q,  # (B,H,S,3)
                k,  # (B,H,S,3)
                v,  # (B,H,S, 3 * num_vector_messages)
                attn_mask=combined_bias,  # (B,H,S,S) additive bias
                is_causal=False,
            )  # -> (B,H,S, 3 * num_vector_messages)

        # Bring back to input frame by inverse rotation; keep "no translation" like original
        attn_out_vecs = rearrange(
            attn_out, "b h s (m d) -> b s (h m) d", d=3, m=self.num_vector_messages
        )
        attn_out_inv = affine.rot[..., None].invert().apply(attn_out_vecs)

        s_out = rearrange(
            attn_out_inv, "b s (h m) d -> b s (h m d)", m=self.num_vector_messages
        )

        if self.mask_and_zero_frameless:
            s_out = s_out.masked_fill(~affine_mask[..., None], 0.0)

        return self.out_proj(s_out)
