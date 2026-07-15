#!/usr/bin/env python3
"""
Test script for RegOT-CUDA sinkhorn_splr() function
"""

import numpy as np
import sys

try:
    import curegot
    sinkhorn_bcd = curegot.numpy.sinkhorn_bcd
    sinkhorn_splr = curegot.numpy.sinkhorn_splr
    print("✓ Successfully imported curegot (CUDA) module")
except ImportError as e:
    print(f"✗ Failed to import curegot: {e}")
    sys.exit(1)

def print_header(header):
    nchar_left = (78 - len(header)) // 2
    nchar_right = 78 - nchar_left - len(header)
    print(f"{'=' * nchar_left} {header} {'=' * nchar_right}")

def generate_data(n, m, reg):
    # Random cost matrix M [n x m]
    M = np.random.rand(n, m)

    # Probability vectors a [n] and b [m]
    a = np.random.rand(n)
    # Normalize to sum to 1
    a = a / np.sum(a)

    b = np.random.rand(m)
    # Normalize to sum to 1
    b = b / np.sum(b)

    print(f"Test data:")
    print(f"  M shape: {M.shape}")
    print(f"  a shape: {a.shape}, sum: {np.sum(a):.6f}")
    print(f"  b shape: {b.shape}, sum: {np.sum(b):.6f}")
    print(f"  reg: {reg}")
    print()

    return M, a, b

def test_splr_vs_bcd(n, m, reg=0.01):
    """Test sinkhorn_splr() against sinkhorn_bcd() reference"""
    # Create test data
    M, a, b = generate_data(n, m, reg)
    tol = 1e-6
    max_iter = 5000

    try:
        # Call sinkhorn_bcd()
        print("Running sinkhorn_bcd() reference implementation...")
        bcd_result = sinkhorn_bcd(M, a, b, reg, tol=tol, max_iter=max_iter, verbose=0)
        bcd_plan = bcd_result["plan"]
        bcd_niter = bcd_result["niter"]
        print("✓ sinkhorn_bcd() reference implementation completed")
        print()

        # Call sinkhorn_splr()
        print("Running sinkhorn_splr() implementation...")
        splr_result = sinkhorn_splr(M, a, b, reg, tol=tol, max_iter=max_iter, verbose=0)
        splr_plan = splr_result["plan"]
        splr_niter = splr_result["niter"]
        print("✓ sinkhorn_splr() implementation completed")
        print()

        # Compare results
        print("Comparison results:")
        print(f"  BCD plan shape : {bcd_plan.shape}")
        print(f"  SPLR plan shape: {splr_plan.shape}")
        print(f"  BCD plan sum   : {np.sum(bcd_plan):.6f}")
        print(f"  SPLR plan sum  : {np.sum(splr_plan):.6f}")
        print(f"  BCD niter      : {bcd_niter}")
        print(f"  SPLR niter     : {splr_niter}")

        # Check if plans are close
        relative_error = np.linalg.norm(splr_plan - bcd_plan) / np.linalg.norm(bcd_plan)
        print(f"  Relative Frobenius norm error: {relative_error:.2e}")

        if relative_error < 1e-3:
            print("✓ SPLR implementation matches BCD (error < 1e-3)")
        elif relative_error < 1e-2:
            print("⚠ SPLR implementation close to BCD (error < 1e-2)")
        else:
            print("✗ SPLR implementation differs significantly from BCD")
            return False

        # Check marginals (row and column sums)
        splr_row_sums = np.sum(splr_plan, axis=1)
        splr_col_sums = np.sum(splr_plan, axis=0)
        row_error = np.linalg.norm(splr_row_sums - a)
        col_error = np.linalg.norm(splr_col_sums - b)
        grad_norm = np.linalg.norm(np.concatenate((
            splr_row_sums - a,
            splr_col_sums[:-1] - b[:-1]
        )))

        print(f"  Row marginal error: {row_error:.2e}")
        print(f"  Column marginal error: {col_error:.2e}")
        print(f"  Gradient norm: {grad_norm:.2e}")

        if row_error < 1e-4 and col_error < 1e-4:
            print("✓ Marginals match `a` and `b`")
        else:
            print("⚠ Marginals differ from `a` and `b`")

        print()
        print("Sample results (first 3x3):")
        print("BCD plan:")
        print(bcd_plan[:3, :3])
        print("SPLR plan:")
        print(splr_plan[:3, :3])

        return True

    except Exception as e:
        print(f"✗ Error during comparison: {e}")
        return False

def main():
    """Run all tests"""
    np.random.seed(123)
    n, m, reg = 80, 60, 0.01
    print_header(f"n = {n}, m = {m}, reg = {reg}")
    print()
    success1 = test_splr_vs_bcd(n, m, reg)
    print()

    n, m, reg = 60, 80, 0.001
    print_header(f"n = {n}, m = {m}, reg = {reg}")
    print()
    success2 = test_splr_vs_bcd(n, m, reg)
    print()

    n, m, reg = 800, 600, 0.01
    print_header(f"n = {n}, m = {m}, reg = {reg}")
    print()
    success3 = test_splr_vs_bcd(n, m, reg)
    print()

    n, m, reg = 600, 800, 0.001
    print_header(f"n = {n}, m = {m}, reg = {reg}")
    print()
    success4 = test_splr_vs_bcd(n, m, reg)
    print()

    if success1 and success2 and success3 and success4:
        print("=" * 80)
        print("✓ All tests completed successfully!")
        return 0
    else:
        print("=" * 80)
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
