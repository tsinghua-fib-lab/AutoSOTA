from sympy import IndexedBase, MatrixSymbol, Sum, symbols, Piecewise
from sympy.functions.special.tensor_functions import KroneckerDelta
from ntkunlimited.recursions.symbolic_inverse_operations import InverseMatrixSymbol, SymmetricMatrixSymbol, inverse_contract_simp
from ntkunlimited.recursions.recursion_symbols import (
    K,
    Kinv,
    V,
    a1, a2,
    b1, b2, b3, b4,
    g1, g2, g3, g4,
)


def test_identity():
    n = symbols("n", integer=True)
    A = MatrixSymbol("A", n, n)
    A_inv = InverseMatrixSymbol(A)

    i, j, k = symbols("i j k", integer=True)

    expr = Sum(A_inv[i, j] * A[j, k], (j, 0, n - 1))
    expr_simplified, _ = inverse_contract_simp(expr)
    assert expr_simplified == KroneckerDelta(i, k), "Identity simplification failed"


def test_no_symmetric_matrix():
    n = symbols("n", integer=True)
    A = MatrixSymbol("A", n, n)
    A_inv = InverseMatrixSymbol(A)

    i, j, k = symbols("i j k", integer=True)

    expr = Sum(A_inv[i, j] * A[k, j], (j, 0, n - 1))
    expr_simplified, _ = inverse_contract_simp(expr)
    assert expr_simplified == Sum(A_inv[i, j] * A[k, j], (j, 0, n - 1)), "Invalid inverse: Cannot assume a symmetric matrix"


def test_symmetric_matrix():
    n = symbols("n", integer=True)
    A = SymmetricMatrixSymbol("A", n)
    A_inv = InverseMatrixSymbol(A)

    i, j, k = symbols("i j k", integer=True)

    expr = Sum(A_inv[i, j] * A[k, j], (j, 0, n - 1))
    expr_simplified, _ = inverse_contract_simp(expr)
    assert expr_simplified == KroneckerDelta(i, k), "Invalid inverse: Cannot assume a symmetric matrix"


def test_identity_no_neighbors():
    n = symbols("n", integer=True)
    A = MatrixSymbol("A", n, n)
    A_inv = InverseMatrixSymbol(A)

    i, j, k = symbols("i j k", integer=True)
    c = symbols("c", real=True)

    expr = Sum(A_inv[i, j] * c * A[j, k], (j, 0, n - 1))
    expr_simplified, _ = inverse_contract_simp(expr)
    assert expr_simplified == c * KroneckerDelta(i, k), "Identity simplification failed"


def test_inverse_contraction_in_summand():
    n = symbols("n", integer=True)
    A = MatrixSymbol("A", n, n)
    A_inv = InverseMatrixSymbol(A)

    i, j, k = symbols("i j k", integer=True)
    c = symbols("c", real=True)

    expr = Sum(A[i, j] * A_inv[j, k] + c, (j, 0, n - 1))
    expr_simplified, _ = inverse_contract_simp(expr)
    assert (
        expr_simplified.doit() == KroneckerDelta(i, k) + c * n
    ), "Inverse contraction in summand failed"


def test_full_contraction():
    n = symbols("n", integer=True)
    A = MatrixSymbol("A", n, n)
    A_inv = InverseMatrixSymbol(A)

    i, j = symbols("i j", integer=True)

    expr = Sum(A[i, j] * A_inv[j, i], (j, 0, n - 1), (i, 0, n - 1))
    expr_simplified, _ = inverse_contract_simp(expr)
    assert expr_simplified.doit() == n, "Full contraction simplification failed"


def test_2_contractions():
    n = symbols("n", integer=True)
    A = MatrixSymbol("A", n, n)
    A_inv = InverseMatrixSymbol(A)

    B = MatrixSymbol("B", n, n)
    B_inv = InverseMatrixSymbol(B)

    i, j, k, l, m = symbols("i j k l m", integer=True)

    expr = Sum(
        A[i, j] * B[i, l] * B_inv[l, m] * A_inv[j, k], (j, 0, n - 1), (l, 0, n - 1)
    )
    expr_simplified, _ = inverse_contract_simp(expr)
    assert expr_simplified.doit() == KroneckerDelta(i, k) * KroneckerDelta(
        i, m
    ), "Two contractions simplification failed"


def test_no_contraction_dependent_factors():
    n = symbols("n", integer=True)
    A = MatrixSymbol("A", n, n)
    A_inv = InverseMatrixSymbol(A)

    c = IndexedBase("c", (n))

    i, j, k, l, m = symbols("i j k l m", integer=True)

    expr = Sum(A[i, j] * A_inv[j, k] * c[j], (j, 0, n - 1), (l, 0, n - 1))
    expr_simplified, _ = inverse_contract_simp(expr)
    assert (
        expr_simplified.doit() == expr
    ), "If other factors depend on the summation index, no "
    "contraction should be performed"


def test_nested_sums():
    n = symbols("n", integer=True)
    A = MatrixSymbol("A", n, n)
    A_inv = InverseMatrixSymbol(A)
    c = symbols("c")

    i, j, k = symbols("i j k", integer=True)

    inner_sum = Sum(A_inv[i, j] * A[j, k], (k, 0, n - 1))
    expr = Sum(inner_sum + c, (j, 0, n - 1))
    expr_simplified, _ = inverse_contract_simp(expr)
    expr_simplified = expr_simplified.doit()
    assert expr_simplified == Piecewise((1, (i >= 0) & (i <= n - 1)), (0, True)) + n * c, "Nested sums simplification failed"


def test_multiplied_sum():
    n = symbols("n", integer=True)
    A = MatrixSymbol("A", n, n)
    A_inv = InverseMatrixSymbol(A)

    i, j, k = symbols("i j k", integer=True)
    c = symbols("c")

    expr = c * Sum(A_inv[i, j] * A[j, k], (j, 0, n - 1))
    expr_simplified, _ = inverse_contract_simp(expr)
    expr_simplified = expr_simplified.doit()
    assert expr_simplified == c * KroneckerDelta(i, k), "Multiplied sum simplification failed"


def test_double_contraction_with_additional_matrix():
    dim = 4

    A = MatrixSymbol("A", dim, dim)

    expr = Kinv[b1, b3] * Kinv[b2, b4] * A[b3, b4] * K[b1, a1] * K[b2, a2]
    expr = Sum(expr, (b1, 0, dim - 1), (b2, 0, dim - 1), (b3, 0, dim - 1), (b4, 0, dim - 1))

    expr_simplified, _ = inverse_contract_simp(expr)
    expr_simplified = expr_simplified.doit()
    assert expr_simplified == A[a1, a2], "Double contraction with additional matrix failed"


def test_4d_tensor_2d_trace():
    dim = 4
    expr = V[g1, g3, g2, g4] * Kinv[g1, b1] * Kinv[g2, b2] * Kinv[g3, b3] * Kinv[g4, b4] \
        * K[b3, b4] * K[b1, a1] * K[b2, a2]
    expr = Sum(
        expr,
        (b1, 0, dim - 1),
        (b2, 0, dim - 1),
        (b3, 0, dim - 1),
        (b4, 0, dim - 1),
        (g1, 0, dim - 1),
        (g2, 0, dim - 1),
        (g3, 0, dim - 1),
        (g4, 0, dim - 1)
    )
    expr_simplified, _ = inverse_contract_simp(expr)
    # expr_simplified = expr_simplified.doit()
    expected = Sum(
        KroneckerDelta(a1, g1) * KroneckerDelta(a2, g2) * KroneckerDelta(b4, g3)
        * V[g1, g3, g2, g4] * Kinv[g4, b4],
        (b4, 0, 3), (g1, 0, 3), (g2, 0, 3), (g3, 0, 3), (g4, 0, 3)
    )
    assert expr_simplified == expected, "4D tensor 2D trace simplification failed"


def test_4d_tensor_full_trace():
    dim = 4
    expr = V[g1, g3, g2, g4] * Kinv[g1, b1] * Kinv[g2, b2] * Kinv[g3, b3] * Kinv[g4, b4] \
        * K[b1, b2] * K[b3, b4]
    expr = Sum(
        expr,
        (b1, 0, dim - 1),
        (b2, 0, dim - 1),
        (b3, 0, dim - 1),
        (b4, 0, dim - 1),
        (g1, 0, dim - 1),
        (g2, 0, dim - 1),
        (g3, 0, dim - 1),
        (g4, 0, dim - 1)
    )
    expr_simplified, _ = inverse_contract_simp(expr)
    # expr_simplified = expr_simplified.doit()
    expected = Sum(
        KroneckerDelta(b2, g1) * KroneckerDelta(b4, g3) * V[g1, g3, g2, g4]
        * Kinv[g2, b2] * Kinv[g4, b4],
        (b2, 0, 3), (b4, 0, 3), (g1, 0, 3), (g2, 0, 3), (g3, 0, 3), (g4, 0, 3)
    )
    assert expr_simplified == expected, "4D tensor full trace simplification failed"


if __name__ == "__main__":
    # test_identity()
    # test_symmetric_matrix()
    # test_inverse_contraction_in_summand()
    # test_identity_no_neighbors()
    # test_full_contraction()
    # test_2_contractions()
    # test_nested_sums()
    # test_multiplied_sum()
    # test_double_contraction_with_additional_matrix()
    # test_4d_tensor_2d_trace()
    test_4d_tensor_full_trace()
    print("All tests passed!")
