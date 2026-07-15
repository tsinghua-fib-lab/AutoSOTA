#!/usr/bin/env python3
"""
Test script for CSR conversion functionality
"""

import numpy as np
from scipy.sparse import csr_matrix

np.set_printoptions(linewidth=100)

def print_header(header):
    nchar_left = (78 - len(header)) // 2
    nchar_right = 78 - nchar_left - len(header)
    print(f"{'=' * nchar_left} {header} {'=' * nchar_right}")

def test_csr_conversion():
    """Test the CSR conversion functionality"""
    try:
        import curegot
        print("✓ Successfully imported curegot (CUDA) module")
    except ImportError as e:
        print(f"✗ Failed to import curegot: {e}")
        return

    # Simple test case
    n, m, K = 6, 3, 10
    reg = 0.1
    shift = 0.01

    # Create test data
    alpha = np.random.normal(scale=0.1, size=n)
    beta = np.random.normal(scale=0.1, size=m)
    M = np.abs(np.random.normal(scale=0.1, size=(n, m)))
    a = np.random.uniform(size=n)
    b = np.random.uniform(size=m)
    a = a / np.sum(a)
    b = b / np.sum(b)
    T = np.exp((alpha.reshape(n, 1) + beta.reshape(1, m) - M) / reg)

    print_header("Input Data")
    print(f"Input shape: {n} x {m}")
    print(f"Regularization parameter: {reg}")
    print(f"K (top elements requested): {K}")
    print(f"alpha:\n{alpha}")
    print(f"beta:\n{beta}")
    print(f"M matrix:\n{M}")
    print()

    print(f"T matrix:\n{T}")
    Tsum = np.sum(T)
    Trowsums = np.sum(T, axis=1)
    Tcolsums = np.sum(T, axis=0)
    print(f"T row sums: {Trowsums}")
    print(f"T column sums: {Tcolsums}")
    print()
    objfn = reg * Tsum - alpha.dot(a) - beta.dot(b)
    grad = np.concatenate((Trowsums - a, Tcolsums[:-1] - b[:-1]))
    gnorm = np.linalg.norm(grad)
    print(f"Objfn = {objfn:.6f}")
    print(f"Grad = {grad}")
    print(f"GradNorm = {gnorm}")
    print()

    # Exclude last column
    T[:, -1] = 0
    topk = np.argpartition(-T.flatten(), K)[:K]
    Tsp = np.zeros(n * m)
    Tsp[topk] = T.flatten()[topk]
    Tsp = Tsp.reshape(n, m)
    Hsl = np.diag(np.concat((Trowsums, Tcolsums[:-1])) + shift)
    Hsl[n:, :n] = Tsp[:, :-1].transpose()
    Hsl_d = Hsl
    Hsl = csr_matrix(Hsl_d)
    print("Hsl sparsified:")
    print(Hsl_d)
    print(f"Data: {Hsl.data}")
    print(f"Col index: {Hsl.indices}")
    print(f"Row pointer: {Hsl.indptr}")
    print()

    try:
        # Call the function
        result = curegot.tests.test_T_computation_sparsify(
            alpha, beta, M, a, b, reg, shift, K, nrun=1
        )

        print_header("T Computation Results")
        print(f"Row sums: {result['Trowsums']}")
        print(f"Column sums: {result['Tcolsums']}")
        print()

        print(f"Objfn: {result['objfn']:.6f}")
        print(f"Grad: {result['grad']}")
        print(f"GradNorm: {result['gnorm']}")
        print()

        csr_val = result["csr_val"]
        csr_colind = result["csr_colind"]
        csr_rowptr = result["csr_rowptr"]
        print(f"CSR data: {csr_val}")
        print(f"CSR col index: {csr_colind}")
        print(f"CSR row pointer: {csr_rowptr}")
        print()

        print_header("Verification")
        Trowsums_diff = np.linalg.norm(Trowsums - result["Trowsums"])
        if Trowsums_diff < 1e-8:
            print(f"✓ Trowsums matches, difference = {Trowsums_diff}")
        else:
            print(f"✗ Trowsums does not match, difference = {Trowsums_diff}")

        Tcolsums_diff = np.linalg.norm(Tcolsums - result["Tcolsums"])
        if Tcolsums_diff < 1e-8:
            print(f"✓ Tcolsums matches, difference = {Tcolsums_diff}")
        else:
            print(f"✗ Tcolsums does not match, difference = {Tcolsums_diff}")
        
        objfn_diff = np.abs(objfn - result["objfn"])
        if objfn_diff < 1e-8:
            print(f"✓ Objfn matches, difference = {objfn_diff}")
        else:
            print(f"✗ Objfn does not match, difference = {objfn_diff}")
        
        grad_diff = np.linalg.norm(grad - result["grad"])
        if grad_diff < 1e-8:
            print(f"✓ Grad matches, difference = {grad_diff}")
        else:
            print(f"✗ Grad does not match, difference = {grad_diff}")
        
        gnorm_diff = np.abs(gnorm - result["gnorm"])
        if gnorm_diff < 1e-8:
            print(f"✓ GradNorm matches, difference = {gnorm_diff}")
        else:
            print(f"✗ GradNorm does not match, difference = {gnorm_diff}")

        val_diff = np.linalg.norm(Hsl.data - csr_val)
        if val_diff < 1e-8:
            print(f"✓ CSR data matches, difference = {val_diff}")
        else:
            print(f"✗ CSR data does not match, difference = {val_diff}")

        colind_diff = np.linalg.norm(Hsl.indices - csr_colind)
        if colind_diff < 1e-8:
            print(f"✓ CSR col index matches, difference = {colind_diff}")
        else:
            print(f"✗ CSR col index does not match, difference = {colind_diff}")

        rowptr_diff = np.linalg.norm(Hsl.indptr - csr_rowptr)
        if rowptr_diff < 1e-8:
            print(f"✓ CSR row pointer, difference = {rowptr_diff}")
        else:
            print(f"✗ CSR row pointer does not match, difference = {rowptr_diff}")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    np.random.seed(123)
    test_csr_conversion()
