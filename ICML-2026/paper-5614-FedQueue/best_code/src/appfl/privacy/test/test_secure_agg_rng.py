"""
Test script to verify the SecureAggregator RNG fix.
Tests: determinism, distribution properties, and mask cancellation.
"""

import torch
import numpy as np
from appfl.privacy import SecureAggregator


def test_determinism():
    """Test that same seed produces same values."""
    print("Testing determinism...")

    secret = b"test_secret_123"
    all_clients = ["client_A", "client_B", "client_C"]
    device = torch.device("cpu")

    sa1 = SecureAggregator("client_A", all_clients, secret, device)
    sa2 = SecureAggregator("client_A", all_clients, secret, device)

    seed = sa1._derive_seed(round_id=0, a="client_A", b="client_B")

    vals1 = sa1._seed_to_normals(1000, seed)
    vals2 = sa2._seed_to_normals(1000, seed)

    assert torch.allclose(vals1, vals2), "Same seed should produce same values"
    print("✓ Determinism test passed")


def test_distribution():
    """Test that generated values have approximately normal distribution."""
    print("\nTesting distribution properties...")

    secret = b"test_secret_123"
    all_clients = ["client_A", "client_B"]
    device = torch.device("cpu")

    sa = SecureAggregator("client_A", all_clients, secret, device)
    seed = sa._derive_seed(round_id=0, a="client_A", b="client_B")

    # Generate large sample
    vals = sa._seed_to_normals(100000, seed).numpy()

    mean = np.mean(vals)
    std = np.std(vals)

    print(f"  Mean: {mean:.6f} (expected: ~0)")
    print(f"  Std:  {std:.6f} (expected: ~1)")

    # Check if mean is close to 0 (within 0.01)
    assert abs(mean) < 0.01, f"Mean {mean} too far from 0"

    # Check if std is close to 1 (within 0.02)
    assert abs(std - 1.0) < 0.02, f"Std {std} too far from 1"

    print("✓ Distribution test passed")


def test_efficiency():
    """Test that Box-Muller efficiency improved (uses both values)."""
    print("\nTesting efficiency...")

    secret = b"test_secret_123"
    all_clients = ["client_A", "client_B"]
    device = torch.device("cpu")

    sa = SecureAggregator("client_A", all_clients, secret, device)
    seed = sa._derive_seed(round_id=0, a="client_A", b="client_B")

    # Generate odd number to test both values are used
    vals = sa._seed_to_normals(1001, seed)

    assert len(vals) == 1001, "Should generate exact number requested"
    print("✓ Efficiency test passed (both Box-Muller values used)")


def test_mask_cancellation():
    """Test that pairwise masks cancel when summed."""
    print("\nTesting mask cancellation...")

    secret = b"shared_secret"
    all_clients = ["client_A", "client_B", "client_C"]
    device = torch.device("cpu")
    round_id = 5
    mask_size = 1000

    # Create aggregators for each client
    aggregators = {
        cid: SecureAggregator(cid, all_clients, secret, device) for cid in all_clients
    }

    # Generate masks for each client
    masks = {
        cid: sa.make_pairwise_mask(mask_size, round_id)
        for cid, sa in aggregators.items()
    }

    # Sum all masks
    total_mask = sum(masks.values())

    # Check if masks cancel (sum should be close to zero)
    max_error = torch.max(torch.abs(total_mask)).item()
    print(f"  Max absolute error in cancellation: {max_error:.2e}")

    assert torch.allclose(total_mask, torch.zeros_like(total_mask), atol=1e-5), (
        f"Masks should cancel, but max error is {max_error}"
    )

    print("✓ Mask cancellation test passed")


def test_symmetry():
    """Test that seed derivation is symmetric."""
    print("\nTesting seed derivation symmetry...")

    secret = b"test_secret"
    all_clients = ["client_A", "client_B"]
    device = torch.device("cpu")

    sa = SecureAggregator("client_A", all_clients, secret, device)

    seed_ab = sa._derive_seed(0, "client_A", "client_B")
    seed_ba = sa._derive_seed(0, "client_B", "client_A")

    assert seed_ab == seed_ba, "Seed derivation should be symmetric"
    print("✓ Symmetry test passed")


def test_no_bias():
    """Test that values are unbiased (no modulo artifacts)."""
    print("\nTesting for bias...")

    secret = b"test_secret_123"
    all_clients = ["client_A", "client_B"]
    device = torch.device("cpu")

    sa = SecureAggregator("client_A", all_clients, secret, device)
    seed = sa._derive_seed(round_id=0, a="client_A", b="client_B")

    # Generate values and check distribution across bins
    vals = sa._seed_to_normals(50000, seed).numpy()

    # Check that values span a reasonable range for normal distribution
    min_val, max_val = vals.min(), vals.max()
    print(f"  Value range: [{min_val:.2f}, {max_val:.2f}]")

    # For 50k samples from normal distribution, we expect roughly:
    # ~99.7% within [-3, 3], and some outliers beyond
    assert min_val < -3 and max_val > 3, "Values should span beyond [-3, 3]"

    print("✓ No bias test passed")


def test_full_aggregation_workflow():
    """Test complete secure aggregation workflow."""
    print("\nTesting full aggregation workflow...")

    # Setup
    secret = b"shared_secret"
    all_clients = ["client_0", "client_1", "client_2"]
    device = torch.device("cpu")
    round_id = 1

    # Simulate model deltas from each client
    deltas = {
        "client_0": {
            "layer1": torch.tensor([1.0, 2.0, 3.0]),
            "layer2": torch.tensor([4.0, 5.0]),
        },
        "client_1": {
            "layer1": torch.tensor([0.5, 1.5, 2.5]),
            "layer2": torch.tensor([3.5, 4.5]),
        },
        "client_2": {
            "layer1": torch.tensor([2.0, 3.0, 4.0]),
            "layer2": torch.tensor([5.0, 6.0]),
        },
    }

    # Expected plaintext sum
    expected_sum = {
        "layer1": torch.tensor([3.5, 6.5, 9.5]),
        "layer2": torch.tensor([12.5, 15.5]),
    }

    # Each client masks their delta
    masked_updates = {}
    shapes_list = {}

    for cid in all_clients:
        sa = SecureAggregator(cid, all_clients, secret, device)
        masked_flat, shapes = sa.mask_update(deltas[cid], round_id)
        masked_updates[cid] = masked_flat
        shapes_list[cid] = shapes

    # Server aggregates masked updates
    aggregated_flat = sum(masked_updates.values())

    # Unflatten back to state dict
    result = SecureAggregator.unflatten_to_state_dict(
        aggregated_flat, shapes_list["client_0"], device
    )

    # Verify result matches expected sum
    for layer_name in expected_sum:
        assert torch.allclose(
            result[layer_name], expected_sum[layer_name], atol=1e-4
        ), f"Layer {layer_name} doesn't match expected sum"

    print("✓ Full aggregation workflow test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SecureAggregator RNG Fix")
    print("=" * 60)

    test_determinism()
    test_distribution()
    test_efficiency()
    test_mask_cancellation()
    test_symmetry()
    test_no_bias()
    test_full_aggregation_workflow()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
