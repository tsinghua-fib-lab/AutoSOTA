from sympy import diff
from ntkunlimited.recursions.symbolic_gaussian_expectation import GaussExpec
from ntkunlimited.recursions.recursion_symbols import (
    z,
    sig,
    K,
    a1, a2,
    b1, b2
)


def test_two_ibps():
    expr = GaussExpec(sig(z[a1]) * sig(z[a2]) * (z[b1] * z[b2] - K[b1, b2]))
    expr = expr.expand(ibp=True)
    expr = expr.doit()
    expected = (K[b1, a1] * K[b2, a2] + K[b1, a2] * K[b2, a1]) \
        * GaussExpec(diff(sig(z[a1]), z[a1]) * diff(sig(z[a2]), z[a2])) \
        + K[b1, a1] * K[b2, a1] * GaussExpec(diff(sig(z[a1]), (z[a1], 2)) * sig(z[a2])) \
        + K[b1, a2] * K[b2, a2] * GaussExpec(sig(z[a1]) * diff(sig(z[a2]), (z[a2], 2)))
    assert expr.equals(expected)


if __name__ == "__main__":
    test_two_ibps()
