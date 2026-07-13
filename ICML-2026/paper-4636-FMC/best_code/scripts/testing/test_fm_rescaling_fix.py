#!/usr/bin/env python3
"""Test that FM post transform training functions correctly use rescaling parameter."""

import torch
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from baselines.trainers import train_fm_post_transform
from simulator import get_simulator

def test_rescaling_parameter():
    """Test that rescaling parameter from config is actually used."""

    print("="*80)
    print("TESTING FM POST TRANSFORM RESCALING FIX")
    print("="*80)

    # Create simple test data
    torch.manual_seed(42)
    n_samples = 100
    theta = torch.randn(n_samples, 2) * 2.0 + 5.0  # Mean ~5, std ~2
    x = torch.randn(n_samples, 3) * 3.0 + 10.0     # Mean ~10, std ~3
    y = torch.randn(n_samples, 3) * 3.0 + 10.0     # Mean ~10, std ~3

    print(f"\nTest data statistics:")
    print(f"  theta: mean={theta.mean().item():.2f}, std={theta.std().item():.2f}")
    print(f"  x: mean={x.mean().item():.2f}, std={x.std().item():.2f}")
    print(f"  y: mean={y.mean().item():.2f}, std={y.std().item():.2f}")

    # Setup paths
    device = torch.device("cpu")
    model_path = Path("logs/rescaling_test")
    model_path.mkdir(parents=True, exist_ok=True)

    # Save a dummy simulator config
    import json
    sim_config = {
        "name": "pure_gaussian",
        "params": {
            "theta_dim": 2,
            "obs_dim": 3,
            "prior_var_scale": 2,
            "likelihood_var_scale": 1,
            "noisy_var_scale": 0.1,
            "seed": 0
        }
    }
    with open(model_path / "simulator.json", "w") as f:
        json.dump(sim_config, f)

    # Create minimal config
    config = {
        "flow_theta": {
            "conditional": True,
            "probability_path": "ot2",
            "prior": "power",
            "base_dist": "gaussian",
            "params": {
                "probability_path_params": {},
                "prior_params": {"rate": 2.0},
                "base_dist_params": {},
                "drift": {
                    "architecture": "resmlp",
                    "hidden_dim": [32, 32],
                    "mlp_params": {}
                }
            }
        },
        "flow_x": {
            "conditional": True,
            "probability_path": "ot2",
            "prior": "power",
            "base_dist": "gaussian",
            "params": {
                "probability_path_params": {},
                "prior_params": {"rate": 2.0},
                "base_dist_params": {},
                "drift": {
                    "architecture": "resmlp",
                    "hidden_dim": [32, 32],
                    "mlp_params": {}
                }
            }
        },
        "npe": "fmpe"
    }

    # Test 1: Train with z_score rescaling
    print("\n" + "="*80)
    print("TEST 1: Training with rescale='z_score'")
    print("="*80)

    training_params_zscore = {
        "lr": 1e-3,
        "epochs": 2,  # Just 2 epochs for quick test
        "batch_size": 32,
        "max_patience": 10,
        "train_size": 0.8,
        "rescale": "z_score"  # ✅ This should be used!
    }

    try:
        # We'll patch the function to capture what rescale value is actually used
        import baselines.trainers as trainers_module
        original_FlowMatching = trainers_module.FlowMatching

        captured_rescale_theta = None
        captured_rescale_x = None

        class FlowMatchingCapture(original_FlowMatching):
            def set_scales(self, data, cond, rescale_mode):
                nonlocal captured_rescale_theta, captured_rescale_x
                # Capture what rescale parameter is actually passed
                if captured_rescale_theta is None:
                    captured_rescale_theta = rescale_mode
                else:
                    captured_rescale_x = rescale_mode
                return super().set_scales(data, cond, rescale_mode)

        # Temporarily replace
        trainers_module.FlowMatching = FlowMatchingCapture

        # Don't actually train, just initialize to capture rescale parameter
        # We'll create a minimal timer
        from utils.timing import HierarchicalTimer
        timer = HierarchicalTimer()

        # Create model (this will call set_scales with the rescale parameter)
        try:
            posterior_zscore = train_fm_post_transform(
                (theta[:10], x[:10], y[:10]),  # Use tiny dataset
                config,
                training_params_zscore,
                device,
                logname="test_zscore",
                model_path=model_path,
                load=False,
                save=False,
                timer=timer
            )
        except Exception as e:
            # We might get errors during training, but we should have captured rescale by now
            print(f"  (Training failed as expected with minimal setup: {e})")

        # Restore original
        trainers_module.FlowMatching = original_FlowMatching

        print(f"\n  Captured rescale for flow_theta: '{captured_rescale_theta}'")
        print(f"  Captured rescale for flow_x: '{captured_rescale_x}'")

        if captured_rescale_theta == "z_score":
            print("  ✅ PASS: flow_theta correctly used 'z_score' rescaling")
        else:
            print(f"  ❌ FAIL: flow_theta used '{captured_rescale_theta}' instead of 'z_score'")
            return False

        if captured_rescale_x == "z_score":
            print("  ✅ PASS: flow_x correctly used 'z_score' rescaling")
        else:
            print(f"  ❌ FAIL: flow_x used '{captured_rescale_x}' instead of 'z_score'")
            return False

    except Exception as e:
        print(f"  ❌ FAIL: Exception during z_score test: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Train with none rescaling
    print("\n" + "="*80)
    print("TEST 2: Training with rescale='none'")
    print("="*80)

    training_params_none = {
        "lr": 1e-3,
        "epochs": 2,
        "batch_size": 32,
        "max_patience": 10,
        "train_size": 0.8,
        "rescale": "none"  # ✅ This should be used!
    }

    try:
        # Patch again
        trainers_module.FlowMatching = FlowMatchingCapture

        captured_rescale_theta = None
        captured_rescale_x = None

        class FlowMatchingCapture(original_FlowMatching):
            def set_scales(self, data, cond, rescale_mode):
                nonlocal captured_rescale_theta, captured_rescale_x
                if captured_rescale_theta is None:
                    captured_rescale_theta = rescale_mode
                else:
                    captured_rescale_x = rescale_mode
                return super().set_scales(data, cond, rescale_mode)

        trainers_module.FlowMatching = FlowMatchingCapture

        from utils.timing import HierarchicalTimer
        timer = HierarchicalTimer()

        try:
            posterior_none = train_fm_post_transform(
                (theta[:10], x[:10], y[:10]),
                config,
                training_params_none,
                device,
                logname="test_none",
                model_path=model_path,
                load=False,
                save=False,
                timer=timer
            )
        except Exception as e:
            print(f"  (Training failed as expected with minimal setup: {e})")

        trainers_module.FlowMatching = original_FlowMatching

        print(f"\n  Captured rescale for flow_theta: '{captured_rescale_theta}'")
        print(f"  Captured rescale for flow_x: '{captured_rescale_x}'")

        if captured_rescale_theta == "none":
            print("  ✅ PASS: flow_theta correctly used 'none' rescaling")
        else:
            print(f"  ❌ FAIL: flow_theta used '{captured_rescale_theta}' instead of 'none'")
            return False

        if captured_rescale_x == "none":
            print("  ✅ PASS: flow_x correctly used 'none' rescaling")
        else:
            print(f"  ❌ FAIL: flow_x used '{captured_rescale_x}' instead of 'none'")
            return False

    except Exception as e:
        print(f"  ❌ FAIL: Exception during none test: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nThe fix is working correctly:")
    print("  - FM training functions now extract 'rescale' from training_params")
    print("  - Different rescale values ('z_score' vs 'none') are correctly applied")
    print("  - The bug that caused identical metrics has been fixed!")

    return True

if __name__ == "__main__":
    success = test_rescaling_parameter()
    sys.exit(0 if success else 1)
