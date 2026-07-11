from ntkunlimited.recursions.symbolic_gaussian_expectation import GaussExpec
from ntkunlimited.recursions.symbolic_to_numerics import hash_expr, make_efficient_numeric
from sympy import Sum, diff
from ntkunlimited.recursions.recursion_symbols import (
    z,
    sig,
    F,
    K,
    H,
    Kinv,
    a1,
    a2,
    a3,
    a4,
    b1,
    b2,
    g1,
    g2,
    n_l,
    n_lm1,
    C_W,
)
from ntkunlimited.recursions.config import cache_dir


def create_recursion_expr():
    dim = 4

    ntk_term = (
        C_W**2
        * GaussExpec(
            sig(z[a1]) * sig(z[a3]) * diff(sig(z[a2]), z[a2]) * diff(sig(z[a4]), z[a4])
        )
        * H[a2, a4]
    )

    f_prev_term = (
        C_W**2
        * n_l
        / n_lm1
        * GaussExpec(sig(z[a1]) * diff(sig(z[a2]), z[a2]) * z[b1])
        * GaussExpec(sig(z[a3]) * diff(sig(z[a4]), z[a4]) * z[b2])
        * Kinv[b1, g1]
        * Kinv[b2, g2]
        * F[g1, a2, g2, a4]
    )
    f_prev_term = Sum(
        f_prev_term,
        (b1, 0, dim - 1),
        (b2, 0, dim - 1),
        (g1, 0, dim - 1),
        (g2, 0, dim - 1),
    )

    recursion_expr = ntk_term + f_prev_term
    # recursion_expr = f_prev_term
    return recursion_expr


def args_hasher(args):
    return hash_expr(args[0]), hash_expr(args[1])


def create_recursion_numeric_fn(gaussexpec_numeric, act_fn):
    """Creates a numeric function for the recursion expression with caching. If an expression has
    been seen before, the numerical function is loaded from cache.
    """
    expr = create_recursion_expr()
    f_cache_name = cache_dir / "F_recursion_numeric_fn.pkl"

    # dim = 4
    # Kinv_placeholder = MatrixSymbol("Kinv", dim, dim)
    # subs = {Kinv: Kinv_placeholder}

    f_args = [F, K, Kinv, H, a1, a2, a3, a4, n_l, n_lm1, C_W]

    f = make_efficient_numeric(
        expr, act_fn, f_args, f_cache_name, args_hasher, gaussexpec_numeric
    )

    return f


if __name__ == "__main__":
    expr = create_recursion_expr()
    print("F recursion expression:")
    print(expr)
