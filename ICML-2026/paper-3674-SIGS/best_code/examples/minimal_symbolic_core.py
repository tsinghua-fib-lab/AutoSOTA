"""Minimal data-free SIGS example.

This example checks the symbolic core without loading model checkpoints or latent
cluster files. It is useful as a first sanity check after installation.
"""

import sympy as sp

from sigs.grammar import GCFG, S, get_mask
from sigs.utils import ExpressionUtils


def main() -> None:
    productions = GCFG.productions()
    start_mask = get_mask(S, GCFG)

    print(f"Number of grammar productions: {len(productions)}")
    print(f"Admissible start-symbol productions: {sum(bool(v) for v in start_mask)}")

    flags = ExpressionUtils.parse_expression("x*t")
    print(f"Expression class for x*t: {flags.math_class.name}")

    x, t = sp.symbols("x t")
    u = sp.sin(sp.pi * x) * sp.exp(-(sp.pi**2) * t)
    residual = sp.simplify(sp.diff(u, t) - sp.diff(u, x, 2))
    print(f"Heat-equation residual: {residual}")


if __name__ == "__main__":
    main()
