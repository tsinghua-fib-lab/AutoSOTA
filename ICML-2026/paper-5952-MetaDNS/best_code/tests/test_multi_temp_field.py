#!/usr/bin/env python
"""Test script for multi-temperature/field extension."""

import numpy as np
import torch

from ..utils_ising import get_temp_field_batch, ising2d_ham, reward_fn_ising


def test_get_temp_field_batch():
    """Test temperature/field batch generation."""
    print("Testing get_temp_field_batch...")
    
    # Test 1: Single temp, single field
    temps = [2.0]
    fields = [0.0]
    batch_size = 128
    beta_batch, h_batch, num_temps, num_fields, batchsize_slice = get_temp_field_batch(
        temps, fields, batch_size
    )
    assert beta_batch.shape == (batch_size,), "Expected shape ({},), got {}".format(batch_size, beta_batch.shape)
    assert h_batch.shape == (batch_size,), "Expected shape ({},), got {}".format(batch_size, h_batch.shape)
    assert num_temps == 1, "Expected 1 temp, got {}".format(num_temps)
    assert num_fields == 1, "Expected 1 field, got {}".format(num_fields)
    assert batchsize_slice == 128, "Expected 128 samples per condition, got {}".format(batchsize_slice)
    assert torch.allclose(beta_batch, torch.tensor(1.0/2.0)), "Beta should be 1/2.0 = 0.5"
    print("Test 1 passed: Single temp/field")
    
    # Test 2: Multiple temps, single field
    temps = [2.0, 2.5, 3.0]
    fields = [0.0]
    batch_size = 120  # 120 = 40 * 3 * 1
    beta_batch, h_batch, num_temps, num_fields, batchsize_slice = get_temp_field_batch(
        temps, fields, batch_size
    )
    assert beta_batch.shape == (batch_size,)
    assert num_temps == 3
    assert num_fields == 1
    assert batchsize_slice == 40
    # With broadcasting and C-order flattening, the pattern cycles through temps
    # Pattern: [T0, T1, T2, T0, T1, T2, ...] repeated batchsize_slice times
    # So indices 0, 3, 6, ... have T=2.0 (beta=0.5)
    # indices 1, 4, 7, ... have T=2.5 (beta=0.4)
    # indices 2, 5, 8, ... have T=3.0 (beta=1/3)
    expected_beta_0 = 1.0 / 2.0  # 0.5
    expected_beta_1 = 1.0 / 2.5  # 0.4
    expected_beta_2 = 1.0 / 3.0   # ~0.333
    assert torch.allclose(beta_batch[0::3], torch.tensor(expected_beta_0)), \
        "Every 3rd sample starting at 0 should have beta=0.5"
    assert torch.allclose(beta_batch[1::3], torch.tensor(expected_beta_1)), \
        "Every 3rd sample starting at 1 should have beta=0.4"
    assert torch.allclose(beta_batch[2::3], torch.tensor(expected_beta_2)), \
        "Every 3rd sample starting at 2 should have beta=1/3"
    # Check that all samples have one of the three beta values
    unique_betas = torch.unique(beta_batch)
    assert len(unique_betas) == 3, "Should have exactly 3 unique beta values"
    print("Test 2 passed: Multiple temps, single field")
    
    # Test 3: Multiple temps and fields
    temps = [2.0, 2.5]
    fields = [-0.1, 0.1]
    batch_size = 120  # 120 = 30 * 2 * 2
    beta_batch, h_batch, num_temps, num_fields, batchsize_slice = get_temp_field_batch(
        temps, fields, batch_size
    )
    assert beta_batch.shape == (batch_size,)
    assert num_temps == 2
    assert num_fields == 2
    assert batchsize_slice == 30
    # With broadcasting and C-order flattening, pattern cycles through all combinations
    # Shape [30, 2, 2] flattens as: [T0,F0], [T0,F1], [T1,F0], [T1,F1], [T0,F0], ...
    # So pattern repeats every 4 samples: [T0,F0], [T0,F1], [T1,F0], [T1,F1]
    expected_beta_t0 = 1.0 / 2.0  # 0.5
    expected_beta_t1 = 1.0 / 2.5  # 0.4
    # Check first few samples match expected pattern
    assert torch.allclose(beta_batch[0], torch.tensor(expected_beta_t0)), "Sample 0 should be T=2.0"
    assert torch.allclose(h_batch[0], torch.tensor(-0.1)), "Sample 0 should be h=-0.1"
    assert torch.allclose(beta_batch[1], torch.tensor(expected_beta_t0)), "Sample 1 should be T=2.0"
    assert torch.allclose(h_batch[1], torch.tensor(0.1)), "Sample 1 should be h=0.1"
    assert torch.allclose(beta_batch[2], torch.tensor(expected_beta_t1)), "Sample 2 should be T=2.5"
    assert torch.allclose(h_batch[2], torch.tensor(-0.1)), "Sample 2 should be h=-0.1"
    assert torch.allclose(beta_batch[3], torch.tensor(expected_beta_t1)), "Sample 3 should be T=2.5"
    assert torch.allclose(h_batch[3], torch.tensor(0.1)), "Sample 3 should be h=0.1"
    # Check pattern repeats
    assert torch.allclose(beta_batch[4], torch.tensor(expected_beta_t0)), "Pattern should repeat at index 4"
    print("Test 3 passed: Multiple temps and fields")
    
    print("All get_temp_field_batch tests passed!\n")


def test_ising2d_ham_per_sample():
    """Test ising2d_ham with per-sample h values."""
    print("Testing ising2d_ham with per-sample h...")
    
    # Create test samples: 4x4 lattice, batch of 4
    L = 4
    B = 4
    S = torch.randint(0, 2, (B, L*L)) * 2 - 1  # Random spins in {-1, 1}
    
    # Test scalar h
    H_scalar = ising2d_ham(S, J=1.0, h=0.5)
    assert H_scalar.shape == (B,), "Expected shape ({},), got {}".format(B, H_scalar.shape)
    
    # Test per-sample h
    h_batch = torch.tensor([0.0, 0.1, 0.2, 0.3])
    H_per_sample = ising2d_ham(S, J=1.0, h=h_batch)
    assert H_per_sample.shape == (B,), "Expected shape ({},), got {}".format(B, H_per_sample.shape)
    
    # Verify they're different when h varies
    assert not torch.allclose(H_scalar, H_per_sample), "Results should differ with per-sample h"
    
    print("ising2d_ham per-sample h test passed!\n")


def test_reward_fn_ising_per_sample():
    """Test reward_fn_ising with per-sample beta and h."""
    print("Testing reward_fn_ising with per-sample beta/h...")
    
    # Create test samples: 4x4 lattice, batch of 4
    L = 4
    B = 4
    S = torch.randint(0, 2, (B, L*L))  # Random samples in {0, 1}
    
    # Test scalar beta/h
    rewards_scalar = reward_fn_ising(S, beta=0.5, J=1.0, h=0.0)
    assert rewards_scalar.shape == (B,), "Expected shape ({},), got {}".format(B, rewards_scalar.shape)
    
    # Test per-sample beta
    beta_batch = torch.tensor([0.4, 0.5, 0.6, 0.7])
    rewards_per_beta = reward_fn_ising(S, beta=beta_batch, J=1.0, h=0.0)
    assert rewards_per_beta.shape == (B,)
    assert not torch.allclose(rewards_scalar, rewards_per_beta), "Results should differ with per-sample beta"
    
    # Test per-sample h
    h_batch = torch.tensor([0.0, 0.1, 0.2, 0.3])
    rewards_per_h = reward_fn_ising(S, beta=0.5, J=1.0, h=h_batch)
    assert rewards_per_h.shape == (B,)
    assert not torch.allclose(rewards_scalar, rewards_per_h), "Results should differ with per-sample h"
    
    # Test both per-sample
    rewards_both = reward_fn_ising(S, beta=beta_batch, J=1.0, h=h_batch)
    assert rewards_both.shape == (B,)
    
    print("reward_fn_ising per-sample beta/h test passed!\n")


def test_backward_compatibility():
    """Test that single temperature mode still works."""
    print("Testing backward compatibility (single temperature)...")
    
    # Simulate old-style usage: single beta
    temps = [1.0 / 0.28]  # Convert beta=0.28 to temperature
    fields = [0.0]
    batch_size = 128
    
    beta_batch, h_batch, num_temps, num_fields, _ = get_temp_field_batch(
        temps, fields, batch_size
    )
    
    # All samples should have same beta
    expected_beta = 0.28
    assert torch.allclose(beta_batch, torch.tensor(expected_beta)), \
        "All samples should have beta={}".format(expected_beta)
    assert torch.allclose(h_batch, torch.tensor(0.0)), "All samples should have h=0.0"
    
    print("Backward compatibility test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Multi-Temperature/Field Extension")
    print("=" * 60 + "\n")
    
    try:
        test_get_temp_field_batch()
        test_ising2d_ham_per_sample()
        test_reward_fn_ising_per_sample()
        test_backward_compatibility()
        
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
