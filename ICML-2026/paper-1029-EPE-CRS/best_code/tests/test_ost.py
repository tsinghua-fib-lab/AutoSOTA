"""
Test script for the Optimal Sparse Survival Tree (OSST) baseline.

Usage (from repo root):
    python tests/test_ost.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
from data.gen import synth_nonlinear
from utils.subgroup import bounding_box, in_region, get_best_boxes, epe_metric
from algs.ost import ost_job, osst_tree_to_bounding_boxes


def test_ost_job():
    """Test ost_job on small synthetic data."""
    print("=" * 60)
    print("Test 1: ost_job on synthetic nonlinear data")
    print("=" * 60)

    d = 2
    n = 300
    B = np.ones((2, d))
    B[0, :] *= -1

    R = np.ones((2, d))
    R[0] *= -(1/6) ** (1/d)
    R[1] *= (1/6) ** (1/d)

    beta_in = 10 * np.ones(d)
    beta_out = 0.5 * np.ones(d)

    X, Y = synth_nonlinear(n, B, R, beta_in, beta_out, seed=42, censor_param=0, censor_type='none')

    X_adjust = X
    X_subgp = X
    B_subgp = bounding_box(X_subgp)

    regularization = 0.01
    minimum_captured_points = 7
    depth_budget = 5
    num_subgroups = 1

    print(f"  n={n}, d={d}, regularization={regularization}, "
          f"minimum_captured_points={minimum_captured_points}, depth_budget={depth_budget}")
    results = ost_job(X_adjust, X_subgp, Y, B_subgp,
                      num_subgroups=num_subgroups,
                      regularization=regularization,
                      minimum_captured_points=minimum_captured_points,
                      depth_budget=depth_budget)

    # One result per num_subgroups (no depth iteration in OSST)
    assert len(results) == num_subgroups, \
        f"Expected {num_subgroups} results, got {len(results)}"
    print(f"  Result count: {len(results)} (expected {num_subgroups}) -- OK")

    for i, res in enumerate(results):
        assert 'subgroup_id' in res, f"Result {i} missing 'subgroup_id'"
        assert 'R' in res, f"Result {i} missing 'R'"
        assert 'beta' in res, f"Result {i} missing 'beta'"
        assert 'regularization' in res, f"Result {i} missing 'regularization'"
        assert 'minimum_captured_points' in res, f"Result {i} missing 'minimum_captured_points'"
        assert 'depth_budget' in res, f"Result {i} missing 'depth_budget'"

        R_box = res['R']
        assert R_box.shape == (2, d), \
            f"Result {i}: R shape {R_box.shape}, expected (2, {d})"
        assert np.all(R_box[0] <= R_box[1]), \
            f"Result {i}: R lower bounds exceed upper bounds: {R_box}"

    print("  All result dicts have correct structure -- OK")
    for res in results:
        print(f"  R={np.round(res['R'], 3)}, beta={np.round(res['beta'], 3)}")

    print("  Test 1 PASSED\n")


def test_bounding_boxes():
    """Test osst_tree_to_bounding_boxes with a hand-crafted tree dict."""
    print("=" * 60)
    print("Test 2: osst_tree_to_bounding_boxes with mock tree")
    print("=" * 60)

    d = 2
    B = np.array([[-1.0, -1.0], [1.0, 1.0]])

    # Build a simple tree:
    #   root: feature 0 >= 0.0
    #     true:  feature 1 >= 0.5  →  leaf A (x0>=0, x1>=0.5)
    #            false: leaf B (x0>=0, x1<0.5)
    #     false: leaf C (x0<0)
    leaf_A = {"prediction": 1}
    leaf_B = {"prediction": 2}
    leaf_C = {"prediction": 3}
    inner = {"feature": 1, "reference": 0.5, "relation": ">=", "true": leaf_A, "false": leaf_B}
    root = {"feature": 0, "reference": 0.0, "relation": ">=", "true": inner, "false": leaf_C}

    boxes = osst_tree_to_bounding_boxes(root, B)
    assert len(boxes) == 3, f"Expected 3 boxes, got {len(boxes)}"

    # Leaf A: x0 >= 0, x1 >= 0.5
    # Leaf B: x0 >= 0, x1 < 0.5
    # Leaf C: x0 < 0
    # Order: A, B, C (DFS true-first)
    box_A, box_B, box_C = boxes

    assert box_A[0, 0] == 0.0 and box_A[0, 1] == 0.5, f"Leaf A lower bounds wrong: {box_A[0]}"
    assert box_A[1, 0] == 1.0 and box_A[1, 1] == 1.0, f"Leaf A upper bounds wrong: {box_A[1]}"

    assert box_B[0, 0] == 0.0 and box_B[0, 1] == -1.0, f"Leaf B lower bounds wrong: {box_B[0]}"
    assert box_B[1, 0] == 1.0 and box_B[1, 1] == 0.5, f"Leaf B upper bounds wrong: {box_B[1]}"

    assert box_C[0, 0] == -1.0, f"Leaf C lower bound wrong: {box_C[0]}"
    assert box_C[1, 0] == 0.0, f"Leaf C upper bound wrong: {box_C[1]}"

    # All boxes should be within or equal to B
    for j, box in enumerate(boxes):
        assert np.all(box[0] >= B[0] - 1e-10), f"Box {j} lower bound below B: {box[0]}"
        assert np.all(box[1] <= B[1] + 1e-10), f"Box {j} upper bound above B: {box[1]}"

    print("  3 boxes extracted correctly -- OK")
    print("  All bounds verified -- OK")
    print("  Test 2 PASSED\n")


def test_univariate():
    """Test ost_job with 1D subgroup features."""
    print("=" * 60)
    print("Test 3: ost_job with univariate subgroup features")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n = 200
    X_full = rng.standard_normal((n, 3))

    T = rng.exponential(np.exp(-2 * X_full[:, 0]))
    Y = np.empty(n, dtype=np.dtype([('failure', '?'), ('time', '<f8')]))
    for i in range(n):
        Y[i] = (True, T[i])

    sort_idx = np.argsort(Y['time'])
    X_full = X_full[sort_idx]
    Y = Y[sort_idx]

    X_subgp = X_full[:, [0]]
    X_adjust = X_full[:, [1]]
    B_subgp = bounding_box(X_subgp)

    results = ost_job(X_adjust, X_subgp, Y, B_subgp,
                      num_subgroups=1, regularization=0.01,
                      minimum_captured_points=7, depth_budget=5)

    assert len(results) == 1
    res = results[0]
    assert res['R'].shape == (2, 1), f"R shape {res['R'].shape}, expected (2, 1)"
    assert res['beta'].shape == (1,), f"beta shape {res['beta'].shape}, expected (1,)"
    print(f"  R={np.round(res['R'].flatten(), 3)}, beta={np.round(res['beta'], 3)}")

    print("  Test 3 PASSED\n")


if __name__ == '__main__':
    test_ost_job()
    test_bounding_boxes()
    test_univariate()
    print("All tests passed!")
