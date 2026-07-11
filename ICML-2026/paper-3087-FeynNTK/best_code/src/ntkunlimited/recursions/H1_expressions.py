from ntkunlimited.recursions.symbolic_gaussian_expectation import GaussExpec
from ntkunlimited.recursions.symbolic_to_numerics import (
    hash_expr,
    make_efficient_numeric,
    inverse_contract_simp,
)
from sympy import Sum, diff
from ntkunlimited.recursions.recursion_symbols import (
    z,
    sig,
    K,
    G1,
    H,
    H1,
    V,
    D,
    F,
    Omega,
    a1,
    a2,
    b1,
    b2,
    b3,
    b4,
    n_l,
    n_lm1,
    C_W,
)
from ntkunlimited.recursions.config import cache_dir


def create_recursion_expr():
    dim = 4

    d2_dsig_dsig = diff(diff(sig(z[a1]), z[a1]) * diff(sig(z[a2]), z[a2]), z[b1], z[b2])

    H1_term = (
        C_W
        / (n_lm1)
        * H1[a1, a2]
        * GaussExpec(diff(sig(z[a1]), z[a1]) * diff(sig(z[a2]), z[a2]))
    )

    G1_term = (
        1 / (2 * n_lm1) * G1[b1, b2] * GaussExpec(diff(Omega(a1, a2), z[b1], z[b2]))
    )
    G1_term = Sum(G1_term, (b1, 0, dim - 1), (b2, 0, dim - 1))

    V_term = (
        1
        / (8 * n_lm1)
        * V[b1, b2, b3, b4]
        * GaussExpec(diff(Omega(a1, a2), z[b1], z[b2], z[b3], z[b4]))
    )
    V_term = Sum(
        V_term, (b1, 0, dim - 1), (b2, 0, dim - 1), (b3, 0, dim - 1), (b4, 0, dim - 1)
    )

    D_term = C_W / (2 * n_lm1) * D[b1, b2, a1, a2] * GaussExpec(d2_dsig_dsig)
    D_term = Sum(D_term, (b1, 0, dim - 1), (b2, 0, dim - 1))

    F_term = C_W / n_lm1 * F[b1, a1, b2, a2] * GaussExpec(d2_dsig_dsig)
    F_term = Sum(F_term, (b1, 0, dim - 1), (b2, 0, dim - 1))

    recursion_expr = H1_term + G1_term + V_term + D_term + F_term
    recursion_expr = n_l * recursion_expr
    return recursion_expr


def args_hasher(args):
    return hash_expr(args[0]), hash_expr(args[1])


def create_recursion_numeric_fn(gaussexpec_numeric, act_fn):
    """Creates a numeric function for the recursion expression with caching. If an expression has
    been seen before, the numerical function is loaded from cache.
    """
    expr = create_recursion_expr()
    f_cache_name = cache_dir / "H1_recursion_numeric_fn.pkl"

    # dim = 4
    # Kinv_placeholder = MatrixSymbol("Kinv", dim, dim)
    # subs = {Kinv: Kinv_placeholder}

    f_args = [H1, K, H, G1, V, D, F, a1, a2, n_l, n_lm1, C_W]

    f = make_efficient_numeric(
        expr, act_fn, f_args, f_cache_name, args_hasher, gaussexpec_numeric
    )

    return f


if __name__ == "__main__":
    expr = create_recursion_expr()
    expr = expr.expand(ibp=True)
    expr, _ = inverse_contract_simp(expr)
    expr = expr.doit()
    print("D recursion expression:")
    print(expr)
