#!/usr/bin/env python3
"""
Test script for sparse Cholesky solver functionality
"""

import sys
import numpy as np
import scipy.sparse as sparse

try:
    import curegot
    print("✓ Successfully imported curegot (CUDA) module")
except ImportError as e:
    print(f"✗ Failed to import curegot: {e}")
    sys.exit(1)

# np.set_printoptions(linewidth=100)

def print_header(header):
    nchar_left = (78 - len(header)) // 2
    nchar_right = 78 - nchar_left - len(header)
    print(f"{'=' * nchar_left} {header} {'=' * nchar_right}")

def test_sparse_cholesky_random(n, density=0.01):
    """Test the sparse Cholesky solver with random data"""

    # Create a random sparse square matrix
    spmat = sparse.random(n, n, density=density, format="coo")
    # print(spmat.toarray())

    # Get the lower part
    L = sparse.tril(spmat)

    # Get sparse positive definite matrix
    M = L.dot(L.T) + 0.1 * sparse.eye(n)
    M = sparse.csr_array(M)

    # Only the lower triangular part
    Mlower = sparse.tril(M)
    Mlower = sparse.csr_array(Mlower)

    # Get random rhs
    rhs = np.random.normal(size=n)

    # Get solution using Numpy/SciPy
    x = sparse.linalg.spsolve(M, rhs)

    try:
        # Solve the system
        xtest = curegot.tests.test_sparse_cholesky_solve(
            M.data, M.indices, M.indptr, rhs)
        err1 = np.linalg.norm(xtest - x)
        print(f"Test sparse Cholesky solver:          error = {err1}")
        
        # Using only the lower triangular part
        xtest = curegot.tests.test_sparse_cholesky_solve(
            Mlower.data, Mlower.indices, Mlower.indptr, rhs)
        err2 = np.linalg.norm(xtest - x)
        print(f"Using only the lower triangular part: error = {err2}")

        if err1 < 1e-8 and err2 < 1e-8:
            print("✓ Test PASSED!")
        else:
            print("✗ Test FAILED!")

    except Exception as e:
        print(f"✗ Test failed: {e}")

if __name__ == "__main__":
    np.random.seed(123)
    n, density = 10, 0.1
    print_header(f"n = {n}, density = {density}")
    test_sparse_cholesky_random(10, density=0.1)
    print()

    n, density = 100, 0.01
    print_header(f"n = {n}, density = {density}")
    test_sparse_cholesky_random(100, density=0.01)
    print()

    n, density = 1000, 0.001
    print_header(f"n = {n}, density = {density}")
    test_sparse_cholesky_random(1000, density=0.001)
    print()

    n, density = 10000, 0.0001
    print_header(f"n = {n}, density = {density}")
    test_sparse_cholesky_random(10000, density=0.0001)
