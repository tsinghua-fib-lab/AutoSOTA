#!/usr/bin/env python3
"""
Quick test script to verify OCMB installation

This script performs basic import tests and runs a minimal example.
"""

import sys
import os

# Add OCMB to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test basic imports"""
    print("Testing imports...")

    try:
        import numpy as np
        print("✓ NumPy")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False

    try:
        import pandas as pd
        print("✓ Pandas")
    except ImportError as e:
        print(f"✗ Pandas: {e}")
        return False

    try:
        import sklearn
        print("✓ Scikit-learn")
    except ImportError as e:
        print(f"✗ Scikit-learn: {e}")
        return False

    try:
        import networkx as nx
        print("✓ NetworkX")
    except ImportError as e:
        print(f"✗ NetworkX: {e}")
        return False

    try:
        import torch
        print("✓ PyTorch")
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA not available")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False

    try:
        from OCMB import (
            OCMB_Base,
            OCMB_CaPS,
            OCMB_Random,
            run_ocmb,
            calculate_metrics
        )
        print("✓ OCMB core")
    except ImportError as e:
        print(f"✗ OCMB core: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_minimal_run():
    """Test minimal OCMB run"""
    print("\nTesting minimal OCMB run...")

    try:
        import numpy as np
        from OCMB import OCMB_Random, calculate_metrics

        # Generate minimal synthetic data
        np.random.seed(42)
        n_samples = 100
        n_nodes = 3

        X = np.zeros((n_samples, n_nodes))
        X[:, 0] = np.random.randn(n_samples)
        X[:, 1] = 0.8 * X[:, 0] + np.random.randn(n_samples) * 0.3
        X[:, 2] = 0.6 * X[:, 1] + np.random.randn(n_samples) * 0.3

        true_graph = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])

        # Run OCMB with Random backbone (simplest)
        ocmb = OCMB_Random(
            seed=42,
            max_parents=3,
            k_mb=3,
            alpha_mb=0.01,
            verbose=False
        )

        ocmb.fit(X, true_adj=true_graph)
        graph = ocmb.get_adjacency_matrix()

        metrics = calculate_metrics(true_graph, graph)

        print(f"✓ OCMB run completed")
        print(f"  Edges found: {int(np.sum(graph))}/{int(np.sum(true_graph))}")
        print(f"  F1 score: {metrics['F1']:.3f}")
        print(f"  Time: {ocmb.get_timings()['total']:.2f}s")

        return True

    except Exception as e:
        print(f"✗ OCMB run failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("OCMB Installation Test")
    print("=" * 60)

    # Test imports
    if not test_imports():
        print("\n✗ Import test failed!")
        print("Please install missing dependencies: pip install -r requirements.txt")
        return 1

    # Test minimal run
    if not test_minimal_run():
        print("\n✗ Minimal run test failed!")
        return 1

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nYou can now run the full examples:")
    print("  python example.py")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
