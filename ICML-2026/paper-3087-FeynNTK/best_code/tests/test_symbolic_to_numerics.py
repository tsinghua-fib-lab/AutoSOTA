import numpy as np
import inspect

# import pytest
from sympy import symbols, diff
from ntkunlimited.recursions.symbolic_gaussian_expectation import GaussExpec
from ntkunlimited.recursions.symbolic_to_numerics import (
    _check_integrand_symmetry,
    GaussExpecNumeric,
    make_numeric,
    custom_gaussexpec,
    IndexStandardizer,
    ConstIdx,
    hash_expr,
)
from ntkunlimited.recursions.numerical_integration import CubatureConf
from ntkunlimited.recursions.recursion_symbols import z, K, sig, a1, a2, a3, a4, Gelu


def test_full_integrand_symmetry_detection():
    integrand = sig(z[a1]) * sig(z[a2]) * sig(z[a3]) * sig(z[a4])
    syms, ids, n, dim = _check_integrand_symmetry(integrand, z)
    assert len(syms) == 24, "Symmetry detection failed for full symmetry"
    assert n == 4, "Incorrect tensor order detected"
    assert dim == 4, "Incorrect data dimension detected"
    full_sym = [
        (0, 1, 2, 3),
        (0, 1, 3, 2),
        (0, 2, 1, 3),
        (0, 2, 3, 1),
        (0, 3, 1, 2),
        (0, 3, 2, 1),
        (1, 0, 2, 3),
        (1, 0, 3, 2),
        (1, 2, 0, 3),
        (1, 2, 3, 0),
        (1, 3, 0, 2),
        (1, 3, 2, 0),
        (2, 0, 1, 3),
        (2, 0, 3, 1),
        (2, 1, 0, 3),
        (2, 1, 3, 0),
        (2, 3, 0, 1),
        (2, 3, 1, 0),
        (3, 0, 1, 2),
        (3, 0, 2, 1),
        (3, 1, 0, 2),
        (3, 1, 2, 0),
        (3, 2, 0, 1),
        (3, 2, 1, 0),
    ]
    assert syms == full_sym, "Full symmetry detection failed"


def test_pair_symmetry_derivatives():
    integrand = (
        sig(z[a1]) * sig(z[a2]) * diff(sig(z[a3]), z[a3]) * diff(sig(z[a4]), z[a4])
    )
    syms, ids, n, dim = _check_integrand_symmetry(integrand, z)
    pair_sym = [(0, 1, 2, 3), (0, 1, 3, 2), (1, 0, 2, 3), (1, 0, 3, 2)]
    assert syms == pair_sym, "Pairwise symmetry detection failed"


def test_symmetry_index_duplicates():
    integrand = (
        sig(z[a1]) * sig(z[a1]) * diff(sig(z[a2]), z[a2]) * diff(sig(z[a2]), z[a2])
    )
    syms, ids, n, dim = _check_integrand_symmetry(integrand, z)
    expected_syms = [(0, 1)]
    assert syms == expected_syms, "Symmetry detection with duplicate indices failed"


def test_symmetric_integration():
    integral = GaussExpec(sig(z[a1]) * sig(z[a2]) * sig(z[a3]) * sig(z[a4]))
    integral = integral.subs(sig, Gelu)

    dim = 4
    rng = np.random.default_rng(3423451436)
    A = rng.random((dim, dim))
    K_np = A @ A.T

    cub_conf = CubatureConf(1e-2, 10_000)
    gaussexpec_numeric = GaussExpecNumeric(cub_conf)

    custom_modules = custom_gaussexpec | {"gaussexpec_numeric": gaussexpec_numeric}

    f = make_numeric(integral, [K, a1, a2, a3, a4], custom_modules)

    sig_0000 = f(K_np, 0, 0, 0, 0)
    sig_1100 = f(K_np, 1, 1, 0, 0)
    sig_0101 = f(K_np, 0, 1, 0, 1)
    sig_1000 = f(K_np, 1, 0, 0, 0)
    assert np.isclose(sig_0000, 7.782893476831424, rtol=cub_conf.rtol)
    assert np.isclose(sig_1100, 7.6196445621780295, rtol=cub_conf.rtol)
    assert np.isclose(sig_0101, 7.6196445621780295, rtol=cub_conf.rtol)
    assert np.isclose(sig_1000, 7.622749716871479, rtol=cub_conf.rtol)


def test_integration_with_derivative():
    integral = GaussExpec(
        sig(z[a1]) * sig(z[a2]) * diff(sig(z[a3]), z[a3]) * diff(sig(z[a4]), z[a4])
    )
    integral = integral.subs(sig, Gelu)
    integral = integral.doit()
    # integral = integral.expand()
    print(integral)

    dim = 4
    rng = np.random.default_rng(3423451436)
    A = rng.random((dim, dim))
    K_np = A @ A.T
    print("K_np:", K_np, "\n")

    cub_conf = CubatureConf(1e-2, 10_000)
    gaussexpec_numeric = GaussExpecNumeric(cub_conf)

    custom_modules = custom_gaussexpec | {"gaussexpec_numeric": gaussexpec_numeric}

    f = make_numeric(integral, [K, a1, a2, a3, a4], custom_modules)
    print(inspect.getsource(f))

    sig_0000 = f(K_np, 0, 0, 0, 0)
    sig_1100 = f(K_np, 1, 1, 0, 0)
    sig_0101 = f(K_np, 0, 1, 0, 1)
    sig_1000 = f(K_np, 1, 0, 0, 0)
    sig_1010 = f(K_np, 1, 0, 1, 0)
    assert np.isclose(sig_0000, 1.1861226202549202, rtol=cub_conf.rtol)
    assert np.isclose(sig_1100, 1.2047531199235133, rtol=cub_conf.rtol)
    assert np.isclose(sig_0101, 1.1608887867750186, rtol=cub_conf.rtol)
    assert np.isclose(sig_1000, 1.162406889178339, rtol=cub_conf.rtol)
    assert np.isclose(sig_1010, 1.1608887867750186, rtol=cub_conf.rtol)


def test_index_standardizer():
    idx_standardizer = IndexStandardizer()

    c = symbols("c")
    # expr = sig(z[a1]) * sig(z[a3]) * diff(sig(z[a2]), z[a2]) * diff(sig(z[a4]), z[a4])
    expr = sig(z[a3]) ** 2 * diff(sig(z[a4]), z[a4]) * (sig(z[a1]) + c)
    expr_std, std_mapping = idx_standardizer(expr)
    std_idxs = list(expr_std.atoms(ConstIdx))
    std_idxs = sorted(std_idxs, key=lambda x: x.name)
    expected_map = {a1: std_idxs[0], a3: std_idxs[1], a4: std_idxs[2]}
    expected_expr = expr.subs(expected_map)
    assert expr_std == expected_expr, "Index standardization failed"


def test_hash_expr():
    expr1 = sig(z[a1]) ** 2 * diff(sig(z[a2]), z[a2]) * diff(sig(z[a4]), z[a4])
    expr2 = sig(z[a1]) ** 2 * diff(sig(z[a4]), z[a4]) * diff(sig(z[a2]), z[a2])
    expr3 = sig(z[a2]) ** 2 * diff(sig(z[a1]), z[a1]) * diff(sig(z[a4]), z[a4])
    hash1 = hash_expr(expr1)
    hash2 = hash_expr(expr2)
    hash3 = hash_expr(expr3)
    assert hash1 == hash2, "Hashing failed for equivalent expressions"
    assert hash1 != hash3, "Hashing failed to distinguish different expressions"


if __name__ == "__main__":
    # test_full_integrand_symmetry_detection()
    test_symmetry_index_duplicates()
    # test_pair_symmetry_derivatives()
    # test_symmetric_integration()
    # test_integration_with_derivative()
    # test_index_standardizer()
    # test_hash_expr()
    # print("All tests passed!")
