from ntkunlimited.recursions.symbolic_gaussian_expectation import GaussExpec
from ntkunlimited.recursions.symbolic_to_numerics import hash_expr, make_efficient_numeric
from sympy import Sum, MatrixSymbol
from ntkunlimited.recursions.recursion_symbols import (
    z,
    sig,
    V,
    K,
    G1,
    Kinv,
    a1,
    a2,
    b1,
    b2,
    b3,
    b4,
    g1,
    g2,
    g3,
    g4,
    h1,
    h2,
    h3,
    h4,
    n_l,
    n_lm1,
    C_W,
)
from ntkunlimited.recursions.config import cache_dir


def create_recursion_expr():
    dim = 4
    # dim = 2

    first_term = (
        1
        / 2
        * Kinv[b1, b3]
        * Kinv[b2, b4]
        * G1[b3, b4]
        * GaussExpec(sig(z[a1]) * sig(z[a2]) * (z[b1] * z[b2] - K[b1, b2]))
    )

    second_term = (
        1
        / 8
        * V[g1, g2, g3, g4]
        * Kinv[g1, b1]
        * Kinv[g2, b2]
        * Kinv[g3, b3]
        * Kinv[g4, b4]
        * GaussExpec(
            sig(z[a1])
            * sig(z[a2])
            * (z[b1] * z[b2] - K[b1, b2])
            * (z[b3] * z[b4] - K[b3, b4])
        )
    )
    second_term = Sum(
        second_term,
        (g1, 0, dim - 1),
        (g2, 0, dim - 1),
        (g3, 0, dim - 1),
        (g4, 0, dim - 1),
    )

    third_term = (
        1
        / 4
        * V[h1, h3, h2, h4]
        * Kinv[h1, b1]
        * Kinv[h2, b2]
        * Kinv[h3, b3]
        * Kinv[h4, b4]
        * K[b3, b4]
        * GaussExpec(sig(z[a1]) * sig(z[a2]) * (-2 * z[b1] * z[b2] + K[b1, b2]))
    )
    # third_term = - 2 * 1 / 4 * V[h1, h3, h2, h4] * Kinv[h1, b1] * Kinv[h2, b2] * Kinv[h3, b3] * Kinv[h4, b4] * K[b3, b4] \
    #     * GaussExpec(sig(z[a1]) * sig(z[a2]) * z[b1] * z[b2]) \
    #     + 1 / 4 * V[h1, h3, h2, h4] * Kinv[h1, b1] * Kinv[h2, b2] * Kinv[h3, b3] * Kinv[h4, b4] * K[b3, b4] \
    #     * GaussExpec(sig(z[a1]) * sig(z[a2])) * K[b1, b2]

    third_term = Sum(
        third_term,
        (h1, 0, dim - 1),
        (h2, 0, dim - 1),
        (h3, 0, dim - 1),
        (h4, 0, dim - 1),
    )

    recursion_expr = first_term + second_term + third_term
    # recursion_expr = second_term

    recursion_expr = Sum(
        recursion_expr,
        (b1, 0, dim - 1),
        (b2, 0, dim - 1),
        (b3, 0, dim - 1),
        (b4, 0, dim - 1),
    )
    # recursion_expr = Sum(first_term + second_term + third_term, (b1, 0, 1), (b2, 0, 1), (b3, 0, 1), (b4, 0, 1))
    recursion_expr = C_W * n_l / n_lm1 * recursion_expr
    return recursion_expr


def args_hasher(args):
    return hash_expr(args[0]), hash_expr(args[1])


def create_recursion_numeric_fn(gaussexpec_numeric, act_fn):
    """Creates a numeric function for the recursion expression with caching. If an expression has
    been seen before, the numerical function is loaded from cache.
    """
    expr = create_recursion_expr()
    f_cache_name = cache_dir / "G1_recursion_numeric_fn.pkl"

    dim = 4
    Kinv_placeholder = MatrixSymbol("Kinv", dim, dim)
    subs = {Kinv: Kinv_placeholder}

    f_args = [G1, V, K, Kinv_placeholder, a1, a2, n_l, n_lm1, C_W]

    f = make_efficient_numeric(
        expr, act_fn, f_args, f_cache_name, args_hasher, gaussexpec_numeric, subs
    )

    return f


if __name__ == "__main__":
    dim = 4
    second_term = (
        V[g1, g2, g3, g4]
        * Kinv[g1, b1]
        * Kinv[g2, b2]
        * Kinv[g3, b3]
        * Kinv[g4, b4]
        * GaussExpec(
            sig(z[a1])
            * sig(z[a2])
            * (z[b1] * z[b2] - K[b1, b2])
            * (z[b3] * z[b4] - K[b3, b4])
        )
    )
    second_term = Sum(
        second_term,
        (g1, 0, dim - 1),
        (g2, 0, dim - 1),
        (g3, 0, dim - 1),
        (g4, 0, dim - 1),
    )
    reduced_expr = second_term.expand(ibp=True)
    print(reduced_expr)
