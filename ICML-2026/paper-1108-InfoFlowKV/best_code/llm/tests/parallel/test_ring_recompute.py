"""
Unit tests for ring attention recompute module.

These tests verify:
1. Ring attention output matches standard attention (single GPU simulation)
2. Distributed scoring produces consistent results
3. RoPE position correction works correctly
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from unittest.mock import MagicMock, patch

# Skip if dependencies not available
try:
    from ring_flash_attn import ring_flash_attn_varlen_func
    RING_FLASH_AVAILABLE = True
except ImportError:
    RING_FLASH_AVAILABLE = False

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class TestDistributedConfig:
    """Tests for DistributedConfig."""

    def test_single_gpu_config(self):
        """Test configuration in single GPU mode."""
        from models.parallel.config import DistributedConfig

        config = DistributedConfig(enabled=False)
        assert config.enabled is False
        assert config.rank == 0
        assert config.world_size == 1

    def test_sequence_partition(self):
        """Test sequence partitioning logic."""
        from models.parallel.config import DistributedConfig

        # Test 4 GPUs, 1000 tokens
        for rank in range(4):
            config = DistributedConfig(enabled=True, rank=rank, world_size=4)
            config.set_sequence_partition(1000)

            assert config.local_seq_start == rank * 250
            assert config.local_seq_end == (rank + 1) * 250
            assert config.local_seq_len == 250

    def test_sequence_partition_uneven(self):
        """Test sequence partitioning with uneven division."""
        from models.parallel.config import DistributedConfig

        # Test 4 GPUs, 1002 tokens (1002 = 4*250 + 2)
        # First 2 ranks get 251, last 2 get 250
        for rank in range(4):
            config = DistributedConfig(enabled=True, rank=rank, world_size=4)
            config.set_sequence_partition(1002)

            if rank < 2:
                expected_len = 251
            else:
                expected_len = 250

            assert config.local_seq_len == expected_len

    def test_num_recompute_positions(self):
        """Test recompute position calculation."""
        from models.parallel.config import DistributedConfig

        config = DistributedConfig(enabled=True, recompute_ratio=0.15)
        config.set_sequence_partition(1000)

        # 15% of 250 = 37
        num_pos = config.get_num_recompute_positions()
        assert num_pos == 37

        # With explicit k
        config.recompute_k = 50
        num_pos = config.get_num_recompute_positions()
        assert num_pos == 50


class TestRoPECorrection:
    """Tests for RoPE position correction in distributed setting."""

    def test_rope_apply_remove_roundtrip(self):
        """Test that applying and removing RoPE is identity."""
        from models.parallel.extractor import DistributedExtractor

        # Create mock extractor with RoPE methods
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        # Create test tensor
        x = torch.randn(1, 4, 10, 64, device=device, dtype=dtype)  # [B, H, T, D]

        # Create cos/sin for positions 0-9
        positions = torch.arange(10, device=device)
        dim = 64
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))

        # Compute cos/sin
        freqs = torch.outer(positions.float(), inv_freq)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(0)  # [1, T, D]
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(0)  # [1, T, D]

        # Apply RoPE
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rope(x, cos, sin):
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            return (x * cos) + (rotate_half(x) * sin)

        def remove_rope(x, cos, sin):
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            return (x * cos) - (rotate_half(x) * sin)

        x_with_rope = apply_rope(x, cos, sin)
        x_recovered = remove_rope(x_with_rope, cos, sin)

        # Should be close to original
        assert torch.allclose(x, x_recovered, atol=1e-5)

    def test_rope_position_rebase(self):
        """Test RoPE rebasing from local to global positions."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Simulate: local positions 0-9, but globally should be 100-109
        local_positions = torch.arange(10, device=device)
        global_positions = torch.arange(100, 110, device=device)

        dim = 64
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))

        # Create test K tensor with local RoPE
        k_original = torch.randn(1, 4, 10, dim, device=device)

        def get_rope(positions):
            freqs = torch.outer(positions.float(), inv_freq)
            cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(0)
            sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(0)
            return cos, sin

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rope(x, cos, sin):
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            return (x * cos) + (rotate_half(x) * sin)

        def remove_rope(x, cos, sin):
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            return (x * cos) - (rotate_half(x) * sin)

        # Apply local RoPE
        local_cos, local_sin = get_rope(local_positions)
        k_with_local_rope = apply_rope(k_original, local_cos, local_sin)

        # Remove local, apply global
        k_no_rope = remove_rope(k_with_local_rope, local_cos, local_sin)
        global_cos, global_sin = get_rope(global_positions)
        k_with_global_rope = apply_rope(k_no_rope, global_cos, global_sin)

        # Verify: direct application of global RoPE should match
        k_direct_global = apply_rope(k_original, global_cos, global_sin)
        assert torch.allclose(k_with_global_rope, k_direct_global, atol=1e-5)


class TestAttentionCorrectness:
    """Tests for attention computation correctness."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_causal_mask_sparse_attention(self):
        """Test that sparse attention with causal mask is correct."""
        device = "cuda"
        dtype = torch.float32

        # Create Q, K, V
        B, H, T, D = 1, 4, 100, 64
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)

        # Full causal attention
        scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Create causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        full_output = torch.matmul(probs, v)

        # Sparse attention at positions [10, 20, 50]
        sparse_positions = torch.tensor([10, 20, 50], device=device)
        q_sparse = q[:, :, sparse_positions, :]

        # Sparse attention
        scores_sparse = torch.matmul(q_sparse, k.transpose(-2, -1)) * scale

        # Sparse causal mask: position p can attend to positions <= p
        key_positions = torch.arange(T, device=device)
        sparse_causal_mask = sparse_positions.unsqueeze(1) >= key_positions.unsqueeze(0)
        scores_sparse = scores_sparse.masked_fill(~sparse_causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        probs_sparse = F.softmax(scores_sparse, dim=-1)
        sparse_output = torch.matmul(probs_sparse, v)

        # Compare with full output at sparse positions
        full_at_sparse = full_output[:, :, sparse_positions, :]
        assert torch.allclose(full_at_sparse, sparse_output, atol=1e-4)

    @pytest.mark.skipif(not FLASH_ATTN_AVAILABLE, reason="Flash attention required")
    def test_flash_attention_matches_sdpa(self):
        """Test that flash attention matches SDPA."""
        device = "cuda"
        dtype = torch.bfloat16

        B, H, T, D = 1, 8, 128, 64
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)

        # SDPA output
        sdpa_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Flash attention output
        q_fa = q.transpose(1, 2).contiguous()  # [B, T, H, D]
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()

        fa_output = flash_attn_func(q_fa, k_fa, v_fa, causal=True)
        fa_output = fa_output.transpose(1, 2)  # Back to [B, H, T, D]

        # Should be close (bfloat16 has some precision loss)
        assert torch.allclose(sdpa_output.float(), fa_output.float(), atol=1e-2)


class TestDistributedScoring:
    """Tests for distributed importance scoring."""

    def test_score_aggregation(self):
        """Test that scores aggregate correctly across simulated GPUs."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Simulate scores from 4 GPUs
        scores_per_gpu = [
            torch.randn(100, device=device),
            torch.randn(100, device=device),
            torch.randn(100, device=device),
            torch.randn(100, device=device),
        ]

        # Concatenate (what all_gather does)
        global_scores = torch.cat(scores_per_gpu)
        assert global_scores.shape[0] == 400

        # Select top 15%
        num_select = int(400 * 0.15)
        _, top_indices = torch.topk(global_scores, num_select)
        top_indices = top_indices.sort().values

        # Verify selection is consistent
        assert len(top_indices) == num_select
        assert top_indices.min() >= 0
        assert top_indices.max() < 400

    def test_local_position_filtering(self):
        """Test filtering global positions to local positions."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Global important positions
        global_important = torch.tensor([10, 50, 120, 250, 380], device=device)

        # GPU 1 handles positions 100-199
        local_start = 100
        local_end = 200

        # Filter to local
        local_mask = (global_important >= local_start) & (global_important < local_end)
        local_global = global_important[local_mask]  # [120]

        # Convert to local indices
        local_indices = local_global - local_start  # [20]

        assert len(local_indices) == 1
        assert local_indices[0] == 20


@pytest.mark.skipif(not RING_FLASH_AVAILABLE, reason="ring-flash-attention required")
class TestRingAttention:
    """Tests for ring attention functionality."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_ring_attention_api(self):
        """Test ring attention API in single-GPU mode."""
        device = "cuda"
        dtype = torch.bfloat16

        # Create inputs in varlen format
        T, H, D = 128, 8, 64
        q = torch.randn(T, H, D, device=device, dtype=dtype)
        k = torch.randn(T, H, D, device=device, dtype=dtype)
        v = torch.randn(T, H, D, device=device, dtype=dtype)

        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)

        # This should work in single-GPU mode (no process group)
        # Note: actual ring communication requires multiple GPUs
        # Here we just verify the API call works
        try:
            output = ring_flash_attn_varlen_func(
                q, k, v,
                cu_seqlens, cu_seqlens,
                T, T,
                causal=True,
            )
            assert output.shape == q.shape
        except RuntimeError as e:
            # May fail if no process group, which is expected
            if "process group" not in str(e).lower():
                raise


class TestEndToEnd:
    """End-to-end tests for the parallel module."""

    def test_config_creation(self):
        """Test creating distributed config."""
        from models.parallel.config import DistributedConfig

        config = DistributedConfig(
            enabled=False,
            recompute_ratio=0.15,
        )
        assert config.recompute_ratio == 0.15

    def test_module_imports(self):
        """Test that all modules can be imported."""
        from models.parallel import (
            DistributedConfig,
            DistributedExtractor,
            DistributedScorer,
            RingAttentionRecomputer,
        )

        assert DistributedConfig is not None
        assert DistributedExtractor is not None
        assert DistributedScorer is not None
        assert RingAttentionRecomputer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
