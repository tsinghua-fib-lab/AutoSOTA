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
    D,
    K,
    H,
    Kinv,
    V,
    Omega,
    a1,
    a2,
    a3,
    a4,
    b1,
    b2,
    b3,
    b4,
    g1,
    g2,
    g3,
    g4,
    n_l,
    n_lm1,
    C_W,
)
from ntkunlimited.recursions.config import cache_dir


def create_recursion_expr():
    dim = 4

    first_term = C_W * (
        GaussExpec(sig(z[a1]) * sig(z[a2]) * Omega(a3, a4))
        - GaussExpec(sig(z[a1]) * sig(z[a2])) * GaussExpec(Omega(a3, a4))
    )

    V_term = (
        C_W
        * n_l
        / (4 * n_lm1)
        * V[b1, b2, b3, b4]
        * Kinv[b1, g1]
        * Kinv[b2, g2]
        * Kinv[b3, g3]
        * Kinv[b4, g4]
        * GaussExpec((z[g1] * z[g2] - K[g1, g2]) * sig(z[a1]) * sig(z[a2]))
        * GaussExpec((z[g3] * z[g4] - K[g3, g4]) * Omega(a3, a4))
    )

    # Sums must be at the same level, otherwise the contraction routine doesn't work
    V_term = Sum(
        V_term,
        (g1, 0, dim - 1),
        (g2, 0, dim - 1),
        (g3, 0, dim - 1),
        (g4, 0, dim - 1),
        (b1, 0, dim - 1),
        (b2, 0, dim - 1),
        (b3, 0, dim - 1),
        (b4, 0, dim - 1),
    )

    D_term = (
        C_W**2
        * n_l
        / (2 * n_lm1)
        * D[g1, g2, a3, a4]
        * GaussExpec((z[b1] * z[b2] - K[b1, b2]) * sig(z[a1]) * sig(z[a2]))
        * Kinv[b1, g1]
        * Kinv[b2, g2]
        * GaussExpec(diff(sig(z[a3]), z[a3]) * diff(sig(z[a4]), z[a4]))
    )
    D_term = Sum(
        D_term,
        (b1, 0, dim - 1),
        (b2, 0, dim - 1),
        (g1, 0, dim - 1),
        (g2, 0, dim - 1),
    )

    recursion_expr = first_term + V_term + D_term
    return recursion_expr


def args_hasher(args):
    return hash_expr(args[0]), hash_expr(args[1])


def create_recursion_numeric_fn(gaussexpec_numeric, act_fn):
    """Creates a numeric function for the recursion expression with caching. If an expression has
    been seen before, the numerical function is loaded from cache.
    """
    expr = create_recursion_expr()
    f_cache_name = cache_dir / "D_recursion_numeric_fn.pkl"

    # dim = 4
    # Kinv_placeholder = MatrixSymbol("Kinv", dim, dim)
    # subs = {Kinv: Kinv_placeholder}

    f_args = [D, K, H, V, a1, a2, a3, a4, n_l, n_lm1, C_W]

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
