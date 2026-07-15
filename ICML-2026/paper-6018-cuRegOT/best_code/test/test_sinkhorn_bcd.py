#!/usr/bin/env python3
"""
Test script for RegOT-CUDA sinkhorn_bcd() function
"""

import numpy as np
import sys

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

def test_cuda_vs_reference():
    """Test CUDA implementation against RegOT-Python reference"""
    try:
        import curegot
        sinkhorn_bcd = curegot.numpy.sinkhorn_bcd
        print("✓ Successfully imported curegot (CUDA) module")
    except ImportError as e:
        print(f"✗ Failed to import curegot: {e}")
        return False

    try:
        import regot
        print("✓ Successfully imported regot (reference) module")
    except ImportError as e:
        print(f"✗ Failed to import regot (reference): {e}")
        print("Please install RegOT-Python with: pip install regot")
        return False

    # Create test data
    np.random.seed(123)
    # Problem size
    n, m = 80, 60
    # Regularization parameter
    reg = 0.01
    M, a, b = generate_data(n, m, reg)

    try:
        # Call CUDA implementation
        print("Running CUDA implementation...")
        cuda_result = sinkhorn_bcd(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=1)
        cuda_plan = cuda_result["plan"]
        print("✓ CUDA implementation completed")
        print()

        # Call RegOT-Python reference implementation
        print("Running RegOT-Python reference implementation...")
        ref_result = regot.sinkhorn_bcd(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=0)
        ref_plan = ref_result.plan
        print("✓ Reference implementation completed")
        print()

        # Compare results
        print("Comparison results:")
        print(f"  CUDA plan shape: {cuda_plan.shape}")
        print(f"  Reference plan shape: {ref_plan.shape}")
        print(f"  CUDA plan sum: {np.sum(cuda_plan):.6f}")
        print(f"  Reference plan sum: {np.sum(ref_plan):.6f}")

        # Check if plans are close
        relative_error = np.linalg.norm(cuda_plan - ref_plan) / np.linalg.norm(ref_plan)
        print(f"  Relative Frobenius norm error of plan: {relative_error:.2e}")

        if relative_error < 1e-3:
            print("✓ CUDA implementation matches reference (error < 1e-3)")
        elif relative_error < 1e-2:
            print("⚠ CUDA implementation close to reference (error < 1e-2)")
        else:
            print("✗ CUDA implementation differs significantly from reference")
            return False

        # Check marginals (row and column sums)
        cuda_row_sums = np.sum(cuda_plan, axis=1)
        cuda_col_sums = np.sum(cuda_plan, axis=0)
        row_error = np.linalg.norm(cuda_row_sums - a)
        col_error = np.linalg.norm(cuda_col_sums - b)

        print(f"  Row marginal error: {row_error:.2e}")
        print(f"  Column marginal error: {col_error:.2e}")

        if row_error < 1e-4 and col_error < 1e-4:
            print("✓ Marginals match `a` and `b`")
        else:
            print("⚠ Marginals differ from `a` and `b`")

        print()
        print("Sample results (first 3x3):")
        print("CUDA plan:")
        print(cuda_plan[:3, :3])
        print("Reference plan:")
        print(ref_plan[:3, :3])

        return True

    except Exception as e:
        print(f"✗ Error during comparison: {e}")
        return False

def test_convergence_mechanism():
    """Test the convergence mechanism and tolerance parameter"""
    try:
        import curegot
        sinkhorn_bcd = curegot.numpy.sinkhorn_bcd
        print("✓ Successfully imported curegot (CUDA) module")
    except ImportError as e:
        print(f"✗ Failed to import curegot: {e}")
        return False

    # Create test data
    np.random.seed(456)
    n, m = 50, 40
    reg = 0.01
    M, a, b = generate_data(n, m, reg)

    try:
        # Test with different tolerance values
        tolerances = [1e-2, 1e-4, 1e-6, 1e-8]
        max_iter = 1000

        print(f"Testing with different tolerance values:")
        for tol in tolerances:
            result = sinkhorn_bcd(M, a, b, reg, tol=tol, max_iter=max_iter, verbose=0)

            if "niter" not in result:
                print(f"  ✗ Result missing 'niter' key")
                return False

            niter = result["niter"]
            print(f"  tol={tol:e}: niter={niter}")

            # Higher tolerance should generally converge faster (fewer iterations)
            # but this is not strictly guaranteed due to problem structure
            if niter > max_iter:
                print(f"  ⚠ Warning: max iterations reached for tol={tol}")

        # Test convergence: stricter tolerance should take more iterations
        print()
        print(f"Testing convergence behavior:")
        strict_result = sinkhorn_bcd(M, a, b, reg, tol=1e-8, max_iter=2000, verbose=0)
        loose_result = sinkhorn_bcd(M, a, b, reg, tol=1e-2, max_iter=2000, verbose=0)

        print(f"  Strict tolerance (1e-8): {strict_result['niter']} iterations")
        print(f"  Loose tolerance (1e-2): {loose_result['niter']} iterations")

        # Test with very low max_iter to see truncation behavior
        print()
        print(f"Testing max_iter truncation:")
        truncated_result = sinkhorn_bcd(M, a, b, reg, tol=1e-10, max_iter=10, verbose=0)
        print(f"  max_iter=10: niter={truncated_result['niter']}")

        if truncated_result["niter"] != 10:
            print(f"  ✗ Expected niter=10 when truncated, got {truncated_result['niter']}")
            return False

        # Test convergence validation: manually compute marginal errors
        print()
        print(f"Testing convergence validation:")
        test_tolerances = [1e-4, 1e-6]

        for tol in test_tolerances:
            result = sinkhorn_bcd(M, a, b, reg, tol=tol, max_iter=1000, verbose=0)
            plan = result["plan"]

            # Compute marginals from the final transport plan
            row_marginals = np.sum(plan, axis=1)  # P * 1_m
            col_marginals = np.sum(plan, axis=0)  # P^T * 1_n

            # Compute marginal errors
            row_error = np.linalg.norm(row_marginals - a)
            col_error = np.linalg.norm(col_marginals - b)
            total_marginal_error = np.hypot(row_error, col_error)

            print(f"  tol={tol:e}: marginal_error={total_marginal_error:e}, niter={result['niter']}")

            # The marginal error should be less than tol (or very close)
            # We allow some tolerance for numerical precision
            if total_marginal_error > tol * 10:  # Allow 10x tolerance for numerical issues
                print(f"⚠ Warning: marginal_error ({total_marginal_error:e}) > tol*10 ({tol*10:e})")
            else:
                print(f"✓ Marginal error validation passed for tol={tol:e}")

        # Test that convergence actually happened by comparing plans
        print()
        print(f"Testing actual convergence:")
        converged_result = sinkhorn_bcd(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=0)
        unconverged_result = sinkhorn_bcd(M, a, b, reg, tol=1e-6, max_iter=5, verbose=0)

        # The converged result should have smaller marginal errors
        plan_converged = converged_result["plan"]
        plan_unconverged = unconverged_result["plan"]

        row_err_conv = np.linalg.norm(np.sum(plan_converged, axis=1) - a)
        col_err_conv = np.linalg.norm(np.sum(plan_converged, axis=0) - b)
        total_err_conv = np.hypot(row_err_conv, col_err_conv)

        row_err_unconv = np.linalg.norm(np.sum(plan_unconverged, axis=1) - a)
        col_err_unconv = np.linalg.norm(np.sum(plan_unconverged, axis=0) - b)
        total_err_unconv = np.hypot(row_err_unconv, col_err_unconv)

        print(f"  Converged ({converged_result['niter']} iter): error={total_err_conv:e}")
        print(f"  Unconverged ({unconverged_result['niter']} iter): error={total_err_unconv:e}")

        if total_err_conv < total_err_unconv:
            print(f"✓ Convergence improves marginal errors")
        else:
            print(f"⚠ Warning: Convergence may not be improving marginal errors as expected")

        print("✓ Convergence mechanism tests passed")
        return True

    except Exception as e:
        print(f"✗ Error in convergence test: {e}")
        return False

def test_warm_start():
    """Test the warm-start mechanism"""
    try:
        import curegot
        sinkhorn_bcd = curegot.numpy.sinkhorn_bcd
        print("✓ Successfully imported curegot (CUDA) module")
    except ImportError as e:
        print(f"✗ Failed to import curegot: {e}")
        return False

    # Create test data
    np.random.seed(789)
    n, m = 50, 40
    reg = 0.01
    M, a, b = generate_data(n, m, reg)

    tol = 1e-8
    max_iter = 2000

    # First run: using zero initial values
    print("First run: using zero initial values...")
    result1 = sinkhorn_bcd(M, a, b, reg, tol, max_iter, verbose=0)

    niter1 = result1["niter"]
    dual1 = result1["dual"]
    plan1 = result1["plan"]

    print(f"  niter: {niter1}")
    print(f"  dual variable shape: {dual1.shape}")
    print(f"  plan shape: {plan1.shape}")

    # Get alpha and beta dual variables
    alpha1 = dual1[:n]
    beta1 = dual1[n:]
    print(f"  alpha shape: {alpha1.shape}, beta shape: {beta1.shape}")
    print()

    # Second run: use the dual variables in the first run as initial values
    print("Second run: use the dual variables in the first run as initial values...")
    # Use x0 argument
    result2 = sinkhorn_bcd(
        M, a, b, reg, tol, max_iter, verbose=0, x0=dual1
    )

    niter2 = result2["niter"]
    print(f"  niter: {niter2}")
    print()

    # Third run: use perturbed initial values
    print("Third run: use perturbed initial values...")
    # Add some noise
    noisy_dual = dual1 + 0.01 * np.random.randn(n + m)
    result3 = sinkhorn_bcd(
        M, a, b, reg, tol, max_iter, verbose=0, x0=noisy_dual
    )

    niter3 = result3["niter"]
    print(f"  niter: {niter3}")
    print()

    # Compare results
    print("Result comparison:")
    print(f"  niter using zero initial values: {niter1}")
    print(f"  niter using dual values: {niter2}")
    print(f"  niter using perturbed dual: {niter3}")
    print()

    # Check quality
    a_tilde_1 = plan1.sum(axis=1)
    b_tilde_1 = plan1.sum(axis=0)
    marginal_error_a_1 = np.linalg.norm(a_tilde_1 - a)
    marginal_error_b_1 = np.linalg.norm(b_tilde_1 - b)

    # Compare transport plan
    plan_diff_12 = np.linalg.norm(plan1 - result2["plan"], ord="fro")
    plan_diff_13 = np.linalg.norm(plan1 - result3['plan'], ord="fro")
    print("Check quality:")
    print(f"  || a_tilde - a || = {marginal_error_a_1:.2e}")
    print(f"  || b_tilde - b || = {marginal_error_b_1:.2e}")
    print(f"  || plan1 - plan2 ||_F: {plan_diff_12:.2e}")
    print(f"  || plan1 - plan3 ||_F: {plan_diff_13:.2e}")
    print()

    success = True

    if niter2 > niter1:
        print("✗ niter should not increase with better initial values")
        success = False
    else:
        print(f"✓ Initial values decrease niter ({niter1} -> {niter2})")

    if plan_diff_12 > 1e-8:
        print(f"✗ Difference of transport plans is too large ({plan_diff_12:.2e})")
        success = False
    else:
        print(f"✓ Transport plans are consistent (difference: {plan_diff_12:.2e})")

    if marginal_error_a_1 > 1e-6 or marginal_error_b_1 > 1e-6:
        print(f"✗ Marginal errors are too large (a: {marginal_error_a_1:.2e}, b: {marginal_error_b_1:.2e})")
        success = False
    else:
        print(f"✓ Marginal errors are acceptable (a: {marginal_error_a_1:.2e}, b: {marginal_error_b_1:.2e})")

    return success

def main():
    """Run all tests"""
    print_header("RegOT-CUDA Tests")
    print()

    # Test against reference implementation
    print_header("1. Comparison with RegOT-Python reference")
    comparison_success = test_cuda_vs_reference()
    print()

    if not comparison_success:
        print("✗ Basic functionality test failed")
        return 1

    # Test convergence mechanism
    print_header("2. Testing convergence mechanism")
    convergence_success = test_convergence_mechanism()
    print()

    # Test warm-start mechanism
    print_header("3. Testing warm-start mechanism")
    warmstart_success = test_warm_start()
    print()

    if convergence_success and comparison_success and warmstart_success:
        print("=" * 80)
        print("All tests completed successfully!")
        print("✓ CUDA version consistent with RegOT-Python")
        print("✓ Tolerance parameter working")
        print("✓ Warm-start working")
        return 0
    else:
        print("=" * 80)
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
