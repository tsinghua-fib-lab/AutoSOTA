from ntkunlimited.recursions.symbolic_gaussian_expectation import GaussExpec
from ntkunlimited.recursions.symbolic_to_numerics import (
    hash_expr,
    make_efficient_numeric,
    inverse_contract_simp,
)
from sympy import diff
from ntkunlimited.recursions.recursion_symbols import (
    z,
    sig,
    B,
    K,
    H,
    a1,
    a2,
    a3,
    a4,
    n_l,
    n_lm1,
    C_W,
)
from ntkunlimited.recursions.config import cache_dir


def create_recursion_expr():

    first_term = (
        GaussExpec(
            diff(sig(z[a1]), z[a1])
            * diff(sig(z[a2]), z[a2])
            * diff(sig(z[a3]), z[a3])
            * diff(sig(z[a4]), z[a4])
        )
        * H[a1, a3]
        * H[a2, a4]
    )

    B_term = (
        n_l
        / n_lm1
        * GaussExpec(diff(sig(z[a1]), z[a1]) * diff(sig(z[a2]), z[a2]))
        * GaussExpec(diff(sig(z[a3]), z[a3]) * diff(sig(z[a4]), z[a4]))
        * B[a1, a2, a3, a4]
    )
    recursion_expr = C_W**2 * (first_term + B_term)
    return recursion_expr


def args_hasher(args):
    return hash_expr(args[0]), hash_expr(args[1])


def create_recursion_numeric_fn(gaussexpec_numeric, act_fn):
    """Creates a numeric function for the recursion expression with caching. If an expression has
    been seen before, the numerical function is loaded from cache.
    """
    expr = create_recursion_expr()
    f_cache_name = cache_dir / "B_recursion_numeric_fn.pkl"

    f_args = [B, K, H, a1, a2, a3, a4, n_l, n_lm1, C_W]

    f = make_efficient_numeric(
        expr, act_fn, f_args, f_cache_name, args_hasher, gaussexpec_numeric
    )

    return f


if __name__ == "__main__":
    expr = create_recursion_expr()
    expr = expr.expand(ibp=True)
    expr, _ = inverse_contract_simp(expr)
    expr = expr.doit()
    print("B recursion expression:")
    print(expr)
