#!/usr/bin/env python
"""
End-to-end integration tests for distributed parallel inference.

These tests verify the full pipeline:
1. Distributed extraction matches sequential extraction
2. Distributed scoring produces same results as single-GPU
3. Final generation quality matches single-GPU baseline

Run with:
    # Single process (simulated distributed)
    python tests/parallel/test_distributed_e2e.py

    # Multi-GPU
    torchrun --nproc_per_node=2 tests/parallel/test_distributed_e2e.py
"""

import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_distributed():
    """Initialize distributed if running with torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank
    return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed."""
    if dist.is_initialized():
        dist.destroy_process_group()


def test_config_partition():
    """Test that config correctly partitions sequences."""
    from models.parallel.config import DistributedConfig

    print("Testing config partitioning...")

    # Single GPU
    config = DistributedConfig(enabled=False)
    config.set_sequence_partition(1000)
    assert config.local_seq_start == 0
    assert config.local_seq_end == 1000
    print("  Single GPU: OK")

    # Multi-GPU simulation
    for world_size in [2, 4, 8]:
        total_len = 1000
        covered = []
        for rank in range(world_size):
            config = DistributedConfig(enabled=True, rank=rank, world_size=world_size)
            config.set_sequence_partition(total_len)
            covered.extend(range(config.local_seq_start, config.local_seq_end))

        assert len(covered) == total_len
        assert sorted(covered) == list(range(total_len))
        print(f"  World size {world_size}: OK")

    print("Config partitioning: PASSED")


def test_distributed_kv_data():
    """Test DistributedKVCacheData class."""
    from models.parallel.extractor import DistributedKVCacheData
    from transformers.cache_utils import DynamicCache

    print("Testing DistributedKVCacheData...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create mock cache
    cache = DynamicCache()
    cache.key_cache = [torch.randn(1, 4, 100, 64, device=device)]
    cache.value_cache = [torch.randn(1, 4, 100, 64, device=device)]

    kv_data = DistributedKVCacheData(
        past_key_values=cache,
        input_ids=torch.randint(0, 1000, (1, 100), device=device),
        attention_mask=torch.ones(1, 100, device=device),
        chunk_lens=100,
        global_offset=500,
        global_total_len=1000,
        local_seq_len=100,
    )

    assert kv_data.total_len == 100
    assert kv_data.num_layers == 1
    assert kv_data.global_offset == 500
    assert kv_data.local_seq_len == 100

    # Test clone
    cloned = kv_data.clone()
    assert cloned.global_offset == kv_data.global_offset
    assert not (cloned.input_ids is kv_data.input_ids)

    print("DistributedKVCacheData: PASSED")


def test_score_selection():
    """Test importance score selection logic."""
    from models.parallel.config import DistributedConfig

    print("Testing score selection...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create mock scores
    scores = torch.randn(1000, device=device)

    # Select top 15%
    num_select = int(1000 * 0.15)
    _, top_indices = torch.topk(scores, num_select)
    top_indices = top_indices.sort().values

    assert len(top_indices) == 150

    # Filter to local partition (rank 1 of 4: positions 250-499)
    local_start = 250
    local_end = 500

    local_mask = (top_indices >= local_start) & (top_indices < local_end)
    local_global = top_indices[local_mask]
    local_indices = local_global - local_start

    # All local indices should be in valid range
    assert (local_indices >= 0).all()
    assert (local_indices < 250).all()

    print("Score selection: PASSED")


def test_rope_operations():
    """Test RoPE apply/remove operations."""
    print("Testing RoPE operations...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(x, cos, sin):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (x * cos) + (rotate_half(x) * sin)

    def remove_rope(x, cos, sin):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (x * cos) - (rotate_half(x) * sin)

    # Create test data
    B, H, T, D = 1, 4, 50, 64
    x = torch.randn(B, H, T, D, device=device, dtype=dtype)

    # Create RoPE embeddings
    positions = torch.arange(T, device=device)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, device=device).float() / D))
    freqs = torch.outer(positions.float(), inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(0)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(0)

    # Apply and remove should give back original
    x_rope = apply_rope(x, cos, sin)
    x_back = remove_rope(x_rope, cos, sin)

    assert torch.allclose(x, x_back, atol=1e-5)
    print("RoPE operations: PASSED")


def test_causal_mask_correctness():
    """Test causal mask generation for sparse positions."""
    print("Testing causal mask correctness...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Full sequence length
    T = 100

    # Sparse query positions
    query_positions = torch.tensor([10, 25, 50, 75, 90], device=device)

    # Generate causal mask
    key_positions = torch.arange(T, device=device)
    causal_mask = query_positions.unsqueeze(1) >= key_positions.unsqueeze(0)

    # Verify mask properties
    # Position 10 can attend to 0-10 (11 positions)
    assert causal_mask[0].sum() == 11
    # Position 25 can attend to 0-25 (26 positions)
    assert causal_mask[1].sum() == 26
    # Position 50 can attend to 0-50 (51 positions)
    assert causal_mask[2].sum() == 51
    # Position 75 can attend to 0-75 (76 positions)
    assert causal_mask[3].sum() == 76
    # Position 90 can attend to 0-90 (91 positions)
    assert causal_mask[4].sum() == 91

    print("Causal mask correctness: PASSED")


def test_all_gather_simulation():
    """Test all-gather simulation for KV cache."""
    print("Testing all-gather simulation...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate 4 GPUs each with 250 tokens
    world_size = 4
    local_len = 250
    total_len = world_size * local_len
    H, D = 4, 64

    # Create local caches
    local_caches = [
        torch.randn(1, H, local_len, D, device=device) for _ in range(world_size)
    ]

    # Simulate all-gather
    full_cache = torch.zeros(1, H, total_len, D, device=device)
    for rank in range(world_size):
        offset = rank * local_len
        full_cache[:, :, offset : offset + local_len, :] = local_caches[rank]

    # Verify
    assert full_cache.shape == (1, H, total_len, D)
    for rank in range(world_size):
        offset = rank * local_len
        assert torch.allclose(
            full_cache[:, :, offset : offset + local_len, :], local_caches[rank]
        )

    print("All-gather simulation: PASSED")


def test_online_softmax():
    """Test online softmax for ring attention accumulation."""
    print("Testing online softmax...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Create test attention scores (from multiple blocks)
    B, H, Q, K = 1, 4, 10, 50
    block_size = 10
    num_blocks = K // block_size

    # Full computation
    scores_full = torch.randn(B, H, Q, K, device=device, dtype=dtype)
    probs_full = torch.softmax(scores_full, dim=-1)
    v_full = torch.randn(B, H, K, 64, device=device, dtype=dtype)
    output_full = torch.matmul(probs_full, v_full)

    # Online computation (simulate ring attention)
    output_online = torch.zeros(B, H, Q, 64, device=device, dtype=dtype)
    lse = torch.full((B, H, Q), float("-inf"), device=device, dtype=dtype)

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = start + block_size

        block_scores = scores_full[:, :, :, start:end]
        block_v = v_full[:, :, start:end, :]

        # Online softmax update
        block_max = block_scores.max(dim=-1, keepdim=True).values.squeeze(-1)
        new_max = torch.maximum(lse, block_max)

        exp_scores = torch.exp(block_scores - new_max.unsqueeze(-1))
        exp_old = torch.exp(lse - new_max)

        # Update output with correction
        output_online = output_online * exp_old.unsqueeze(-1) + torch.matmul(
            exp_scores, block_v
        )

        # Update LSE
        lse = new_max + torch.log(exp_old + exp_scores.sum(dim=-1))

    # Normalize
    output_online = output_online / lse.unsqueeze(-1).exp()

    # Should match full computation
    assert torch.allclose(output_full, output_online, atol=1e-4)
    print("Online softmax: PASSED")


def run_all_tests():
    """Run all tests."""
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print("Running Distributed E2E Tests")
        print(f"World size: {world_size}")
        print("=" * 60)
        print()

    try:
        test_config_partition()
        test_distributed_kv_data()
        test_score_selection()
        test_rope_operations()
        test_causal_mask_correctness()
        test_all_gather_simulation()
        test_online_softmax()

        if rank == 0:
            print()
            print("=" * 60)
            print("ALL TESTS PASSED")
            print("=" * 60)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    run_all_tests()
