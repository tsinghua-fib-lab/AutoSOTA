from ntkunlimited.recursions.symbolic_gaussian_expectation import GaussExpec
from ntkunlimited.recursions.symbolic_to_numerics import (
    hash_expr,
    make_efficient_numeric,
)
from sympy import Sum
from ntkunlimited.recursions.recursion_symbols import (
    z,
    sig,
    V,
    K,
    Kinv,
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
    sig4pt_conn_expr = C_W**2 * (
        GaussExpec(sig(z[a1]) * sig(z[a2]) * sig(z[a3]) * sig(z[a4]))
        - GaussExpec(sig(z[a1]) * sig(z[a2])) * GaussExpec(sig(z[a3]) * sig(z[a4]))
    )
    v_term = (
        C_W**2
        * n_l
        / (4 * n_lm1)
        * (
            V[g1, g2, g3, g4]
            * Kinv[g1, b1]
            * Kinv[g2, b2]
            * Kinv[g3, b3]
            * Kinv[g4, b4]
            * GaussExpec(sig(z[a1]) * sig(z[a2]) * (z[b1] * z[b2] - K[b1, b2]))
            * GaussExpec(sig(z[a3]) * sig(z[a4]) * (z[b3] * z[b4] - K[b3, b4]))
        )
    )
    dim = 4
    v_term = Sum(
        v_term,
        (b1, 0, dim - 1),
        (b2, 0, dim - 1),
        (b3, 0, dim - 1),
        (b4, 0, dim - 1),
        (g1, 0, dim - 1),
        (g2, 0, dim - 1),
        (g3, 0, dim - 1),
        (g4, 0, dim - 1),
    )

    recursion_expr = sig4pt_conn_expr + v_term
    return recursion_expr


def args_hasher(args):
    return hash_expr(args[0]), hash_expr(args[1])


def create_recursion_numeric_fn(gaussexpec_numeric, act_fn):
    """Creates a numeric function for the recursion expression with caching. If an expression has
    been seen before, the numerical function is loaded from cache.
    """
    expr = create_recursion_expr()
    f_cache_name = cache_dir / "V4_recursion_numeric_fn.pkl"

    f_args = [V, K, a1, a2, a3, a4, n_l, n_lm1, C_W]

    f = make_efficient_numeric(
        expr, act_fn, f_args, f_cache_name, args_hasher, gaussexpec_numeric
    )

    return f
