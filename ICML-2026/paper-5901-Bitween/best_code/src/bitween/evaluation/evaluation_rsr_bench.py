import math
from time import time

import numpy as np
import sympy

from bitween._settings import InitialMethod
from bitween.analyzer import verify
from bitween.config import Config, MILPSolver
from bitween.main import infer_property
from bitween.miscs import getLogger
from bitween.sampler import Distribution, Domain

config = Config()
log = getLogger(__name__, config.logger_level)


def test_sin_glibc():
    iteration = 40
    sample_error = {}

    def F(x):
        return math.sin(x)

    for i in range(5, iteration, 5):
        props, error, sample = infer_property(
            Domain.Real,
            Distribution(np.random.uniform, low=-5, high=5),
            ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
            ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
            [F],
            max_degree=2,
            n=i,
            epsilon=0.1,
        )

        def f(x):
            return sympy.sin(x)

        for eq in props:
            verify(eq, [f])

        if props:
            print("Properties found:")
            e_sum = 0
            s_sample = 0
            for loc, props in props.items():
                print(f"Loc: {loc}")
                for prop in props:
                    print(f" {prop}")
                if not props:
                    continue
                e = round(error[loc], 5)
                print(f"Error: {e}")
                s = sample[loc]
                print(f"Sample: {s}")
                (e_sum, s_sample) = (e_sum + e, s_sample + s)
            sample_error[i] = (e_sum / len(props), s_sample / len(props))
        else:
            print("No properties found")

    # create a panda dataset for figure and order by sample
    import pandas as pd

    df = pd.DataFrame(sample_error).T
    df.columns = ["Error", "Sample"]
    df = df.sort_values(by="Sample")
    print(df)


def test_sin():
    def F(x, terms=40):
        """sinTaylor"""
        result = 0.0

        for n in range(terms):
            numerator = (-1) ** n
            denominator = 1

            for i in range(1, 2 * n + 2):
                denominator *= i

            term = numerator * (x ** (2 * n + 1)) / denominator
            result += term

        return result

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=15,
    )

    def f(x):
        return sympy.sin(x)

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f])[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x)")
                )
            )


def test_cos():
    def F(x):
        return math.cos(x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=100,
    )

    def f(x):
        return sympy.cos(x)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_tan():
    def F(x):
        return math.tan(x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=3,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return sympy.tan(x)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_tanh():
    def F(x):
        return math.tanh(x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=3,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=80,
    )

    def f(x):
        return sympy.tanh(x)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_identity():
    _c = 5

    def F(x, c=_c):
        return c * x

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=1,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=10,
    )

    def f(x, c=_c):
        return c * x

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f], {"c": _c})

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_inverse():
    def F(x):
        return 1 / x

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=2,
        n=150,
    )

    def f(x):
        return 1 / x

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_inverse_add():
    def F(x):
        return 1 / (x + 1)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=2,
        n=150,
    )

    def f(x):
        return 1 / (x + 1)

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f])[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_exp():
    def F(x):
        return math.exp(x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=10,
    )

    def f(x):
        return sympy.exp(x)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


# double exp1x(double x) { return (exp(x) - 1.0) / x; }
def test_exp_div_by_x():
    def F(x):
        return math.exp(x) / x

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-2, high=2),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=5,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=200,
    )

    def f(x):
        return sympy.exp(x) / x

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_exp_div_by_x_composite():
    def F(x):
        return math.exp(x) / x

    def H(x):
        return math.exp(x)

    def P(x, y):
        return x + y

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-2, high=2),
        ["F(x+y)", "H(x)", "H(y)", "P(x,y)"],
        ["f(x+y)", "h(x)", "h(y)", "p(x,y)"],
        [F, H, P],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return sympy.exp(x) / x

    def h(x):
        return sympy.exp(x)

    def p(x, y):
        return x + y

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f, h, p])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


# double exp1x_log(double x) { return (exp(x) - 1.0) / log(exp(x)); }
def test_exp_div_by_log():
    def F(x):
        return (math.exp(x) - 1.0) / math.log(math.exp(x))

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        [F],
        max_degree=3,
        n=150,
    )

    def f(x):
        return (sympy.exp(x) - 1.0) / sympy.log(sympy.exp(x))

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


# double floudas(double x1, double x2) { return x1 + x2; }
def test_floudas():
    def F(x1, x2):
        return x1 + x2

    def pre_F(x, y):
        return 0 <= x <= 2 and 0 <= y <= 3 and x + y <= 4

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=0, high=2),
        [
            "F(x1+x2,y1+y2)",
            "F(x2+x3,y2+y3)",
            "F(x1+x3,y1+y3)",
            "F(x1,y1)",
            "F(x2,y2)",
        ],
        [
            "f(x1+x2,y1+y2)",
            "f(x2+x3,y2+y3)",
            "f(x1+x3,y1+y3)",
            "f(x1,y1)",
            "f(x2,y2)",
        ],
        [F],
        max_degree=1,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
        preconditions={"F": pre_F},
    )

    def f(x1, x2):
        return x1 + x2

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_mean():
    def F(x, y, z):
        return 1 / 3 * (x + y + z)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        # fmt: off
        ["F(x1+x2+x3, y1+y2+y3, z1+z2+z3)", "F(x1+x2,y1+y2,z1+z2)", "F(x2+x3,y2+y3,z2+z3)", "F(x1+x3,y1+y3,z1+z3)", "F(x1,y1,z1)", "F(x2,y2,z2)", "F(x3,y3,z3)"],
        ["f(x1+x2+x3, y1+y2+y3, z1+z2+z3)", "f(x1+x2,y1+y2,z1+z2)", "f(x2+x3,y2+y3,z2+z3)", "f(x1+x3,y1+y3,z1+z3)", "f(x1,y1,z1)", "f(x2,y2,z2)", "f(x3,y3,z3)"],
        # fmt: on
        [F],
        max_degree=1,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x, y, z):
        return 1 / 3 * (x + y + z)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_inverse_square():

    def F(x):
        return 1 / (x**2)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x)", "F(y)", "F(x-y)"],
        ["f(x+y)", "f(x)", "f(y)", "f(x-y)"],
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return 1 / (x**2)

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f])[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_diff_x2_y2():
    def F(x1, x2):
        return x1**2 - x2**2

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        # fmt: off
        ["F(x-a,y-b)", "F(x+a,y-b)", "F(x,y)", "F(a,b)", "F(x-a, y)", "F(x+a, y)", "F(x, y-a)", "F(x, y+a)"],
        ["f(x-a,y-b)", "f(x+a,y-b)", "f(x,y)", "f(a,b)", "f(x-a,y)", "f(x+a,y)", "f(x,y-a)", "f(x,y+a)"],
        # fmt: on
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=100,
    )

    def f(x1, x2):
        return x1**2 - x2**2

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_exp_minus_one():
    def F(x):
        return math.exp(x) - 1

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=10,
    )

    def f(x):
        return sympy.exp(x) - 1

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_cot():
    def F(x):
        return 1 / math.tan(x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=3,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return 1 / sympy.tan(x)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_inverse_cot_plus_one():

    def F(x):
        return 1 / (1 / math.tan(x) + 1)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=3,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
        # milp=MILPSolver.GUROBI,
        # var_bound=4,
    )

    def f(x):
        return 1 / (1 / sympy.tan(x) + 1)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_inverse_tan_plus_one():

    def F(x):
        return 1 / (math.tan(x) + 1)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=3,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return 1 / (sympy.tan(x) + 1)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_x_over_one_minus_x():
    def F(x):
        return x / (1 - x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=1000,
        epsilon=0.001,
    )

    def f(x):
        return x / (1 - x)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_minus_x_over_one_minus_x():
    def F(x):
        return -x / (1 - x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return -x / (1 - x)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


# f(x)=\frac{\sin cx}{\sin(cx+a)
def test_sin_over_sin():
    _a = math.radians(90)
    _c = 1

    def F(x, a=_a, c=_c):
        return math.sin(c * x) / math.sin(c * x + a)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        [F],
        max_degree=3,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=300,
    )

    def f(x):
        return sympy.sin(x) / sympy.sin(x + 1)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f], {"a": _a, "c": _c})

    print(f"Sample complexity: {equations[2]['vtrace1']}")


# f(x)=\frac{\sinh cx}{\sinh(cx+a)
def test_sinh_over_sinh():
    _a = math.radians(90)
    _c = 1

    def F(x, a=_a, c=_c):
        return math.sinh(c * x) / math.sinh(c * x + a)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=3,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x, a=_a, c=_c):
        return sympy.sinh(c * x) / sympy.sinh(c * x + a)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f], {"a": _a, "c": _c})

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_cosh():
    def F(x):
        return math.cosh(x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return sympy.cosh(x)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_squared():
    def F(x):
        return x**2

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        [F],
        max_degree=1,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=10,
    )

    def f(x):
        return x**2

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f])[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )
            print("Isolating f(x):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x)")
                )
            )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_cube():

    def F(x):
        return x**3

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return x**3

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f])[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_sinh():
    def F(x):
        return math.sinh(x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return sympy.sinh(x)

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f])[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_sigmoid():
    def F(x):
        return 1 / (1 + math.exp(-x))

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        [F],
        max_degree=3,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=100,
    )

    def f(x):
        return 1 / (1 + sympy.exp(-x))

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_sigmoid_extra():

    def F(x):
        return 1 / (1 + math.exp(-x))

    def f(x):
        return 1 / (1 + sympy.exp(-x))

    import statistics

    max_input = 300
    max_iter = 3
    sample_error = {}

    for i in range(10, max_input, 10):
        e_sum = 0
        s_sample = 0
        error_j = 0
        sample_j = 0
        prop_j = []
        count = 0
        for j in range(max_iter):

            props, error, sample = infer_property(
                Domain.Real,
                Distribution(np.random.uniform, low=-5, high=5),
                ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],
                ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
                [F],
                max_degree=3,
                n=i,
                milp=None,
                epsilon=0.2,
                method=InitialMethod.MULTIPLE_REGRESSION,
            )

            if props:
                verified = 0
                print("Properties found:")
                for loc, props in props.items():
                    print(f"Loc: {loc}")
                    for prop in props:
                        print(f" {prop}")
                        if (
                            isinstance(prop, sympy.core.relational.Equality)
                            and verify(prop, [f])[0]
                        ):
                            verified += 1
                    if not props:
                        continue
                    e = round(error[loc], 5)
                    print(f"Error: {e}")
                    s = sample[loc]
                    print(f"Sample: {s}")
                    (e_sum, s_sample) = (e_sum + e, s_sample + s)
                if len(props) > 0:
                    error_j += e_sum / len(props)
                    sample_j += s_sample / len(props)
                    prop_j.append(verified)
                    count += 1
            else:
                print("No properties found")
        if count > 0:
            sample_error[i] = (
                error_j / count,
                sample_j / count,
                statistics.median(prop_j),
            )
        else:
            sample_error[i] = (0, i, 0)

    # create a panda dataset for figure and order by sample
    import pandas as pd

    df = pd.DataFrame(sample_error).T
    df.columns = ["Error", "Sample", "Properties"]
    df = df.sort_values(by="Sample")
    # save the dataset to a csv file
    df.to_csv("sigmoid_extra.csv")
    print(df)


def test_sigmoid_derivative():
    def F(x):
        return 1 / (1 + math.exp(-x))

    def dF(x):
        return F(x) * (1 - F(x))

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["dF(x+y)", "dF(x-y)", "dF(x)", "dF(y)"],
        ["df(x+y)", "df(x-y)", "df(x)", "df(y)"],
        [dF],
        max_degree=5,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=100,
    )

    def f(x):
        return 2 / (1 + sympy.exp(-x))

    def df(x):
        return f(x) * (1 - f(x))

    for eq in equations[0]["vtrace1"]:
        verify(eq, [df])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_logistic(L=1, k=2, x0=0):

    def F(x):
        return L / (1 + math.exp(-k * (x - x0)))

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return L / (1 + sympy.exp(-k * (x - x0)))

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f], {"L": L, "k": k, "x0": x0})[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_logistic1(L=3, k=2, x0=1):

    def F(x):
        return L / (1 + math.exp(-k * (x - x0)))

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return L / (1 + sympy.exp(-k * (x - x0)))

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f], {"L": L, "k": k, "x0": x0})[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_softmax2_1():
    def F(x, y):
        return math.exp(x) / (math.exp(x) + math.exp(y))

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+r, y+r)", "F(x, y)", "F(x, r)", "F(y, r)"],
        ["f(x+r, y+r)", "f(x, y)", "f(x, r)", "f(y, r)"],
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x, y):
        return sympy.exp(x) / (sympy.exp(x) + sympy.exp(y))

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_softmax2_2():
    def F(x, y):
        return math.exp(y) / (math.exp(x) + math.exp(y))

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y, y+z)", "F(x,y)", "F(y,z)", "F(x,z)"],
        ["f(x+y, y+z)", "f(x,y)", "f(y,z)", "f(x,z)"],
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x, y):
        return sympy.exp(y) / (sympy.exp(x) + sympy.exp(y))

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_softmax2_alt_1():
    def F(x, y):
        return math.exp(x) / (math.exp(x) + math.exp(y))

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x1+x2, y1+y2)", "F(x1, y1)", "F(x2, y1)", "F(x1, y2)", "F(x2, y2)"],
        ["f(x1+x2, y1+y2)", "f(x1, y1)", "f(x2, y1)", "f(x1, y2)", "f(x2, y2)"],
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x, y):
        return sympy.exp(x) / (sympy.exp(x) + sympy.exp(y))

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_arctan():
    def F(x):
        return math.atan(x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-2, high=2),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        [F],
        max_degree=3,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=200,
        milp=MILPSolver.GLPK,
    )

    def f(x):
        return sympy.atan(x)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_mod():
    _R = 101

    def F(x, R=_R):
        return x % R

    equations = infer_property(
        Domain.Positive_Integer,
        Distribution(np.random.randint, low=1, high=10),
        ["F(x+y)", "F(x)", "F(y)"],
        ["f(x+y)", "f(x)", "f(y)"],
        [F],
        max_degree=1,
        n=100,
    )

    def f(x, R=_R):
        return sympy.Mod(x, R)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f], Domain.Positive_Integer, {"R": _R})
        print("Isolating f(x+y):")
        print(
            sympy.solve(
                sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
            )
        )
        print("Isolating f(x-y):")
        print(
            sympy.solve(
                sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
            )
        )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_mod_mult():
    _R = 100001

    def F(x, y, R=_R):
        return (x * y) % R

    equations = infer_property(
        Domain.Positive_Integer,
        Distribution(np.random.randint, low=1, high=10),
        ["F(x1+x2, y1+y2)", "F(x1, y1)", "F(x2, y1)", "F(x1, y2)", "F(x2, y2)"],
        ["f(x1+x2, y1+y2)", "f(x1, y1)", "f(x2, y1)", "f(x1, y2)", "f(x2, y2)"],
        [F],
        max_degree=1,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=100,
    )

    def f(x, y, R=_R):
        return (x * y) % R

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f], Domain.Positive_Integer, {"R": _R})

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_int_mult():
    def F(x, y):
        return x * y

    equations = infer_property(
        Domain.Integer,
        Distribution(np.random.randint, low=1, high=10),
        ["F(x1+x2, y1+y2)", "F(x1, y1)", "F(x2, y1)", "F(x1, y2)", "F(x2, y2)"],
        ["f(x1+x2, y1+y2)", "f(x1, y1)", "f(x2, y1)", "f(x1, y2)", "f(x2, y2)"],
        [F],
        max_degree=1,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=10,
    )

    def f(x, y):
        return x * y

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f], Domain.Integer)

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_int_mult_3():
    def F(x, y):
        return x * y

    equations = infer_property(
        Domain.Integer,
        Distribution(np.random.randint, low=1, high=10),
        [
            "F(x1+x2+x3, y1+y2+y3)",
            "F(x1, y1)",
            "F(x2, y1)",
            "F(x3, y1)",
            "F(x1, y2)",
            "F(x2, y2)",
            "F(x3, y2)",
            "F(x1, y3)",
            "F(x2, y3)",
            "F(x3, y3)",
        ],
        [
            "f(x1+x2+x3, y1+y2+y3)",
            "f(x1, y1)",
            "f(x2, y1)",
            "f(x3, y1)",
            "f(x1, y2)",
            "f(x2, y2)",
            "f(x3, y2)",
            "f(x1, y3)",
            "f(x2, y3)",
            "f(x3, y3)",
        ],
        [F],
        max_degree=1,
        n=20,
    )

    def f(x, y):
        return x * y

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f], Domain.Integer)

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_int_add_3():
    def F(x, y):
        return x + y

    equations = infer_property(
        Domain.Integer,
        Distribution(np.random.randint, low=1, high=10),
        [
            "F(x1+x2+x3, y1+y2+y3)",
            "F(x1, y1)",
            "F(x2, y1)",
            "F(x3, y1)",
            "F(x1, y2)",
            "F(x2, y2)",
            "F(x3, y2)",
            "F(x1, y3)",
            "F(x2, y3)",
            "F(x3, y3)",
        ],
        [
            "f(x1+x2+x3, y1+y2+y3)",
            "f(x1, y1)",
            "f(x2, y1)",
            "f(x3, y1)",
            "f(x1, y2)",
            "f(x2, y2)",
            "f(x3, y2)",
            "f(x1, y3)",
            "f(x2, y3)",
            "f(x3, y3)",
        ],
        [F],
        max_degree=1,
        n=10,
    )

    def f(x, y):
        return x + y

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f], Domain.Integer)

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_sinc_composite():
    def F(x):
        return math.sin(x) / x

    def RSR_SIN(x, y):
        return (math.sin(x) ** 2 - math.sin(y) ** 2) / math.sin(x - y)

    def RSR_X(x, y):
        return x + y

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "RSR_SIN(x,y)", "RSR_X(x,y)"],
        ["f(x+y)", "rsr_sin(x,y)", "rsr_x(x,y)"],
        [F],
        RSR_SIN,
        RSR_X,
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=50,
    )

    def f(x):
        return sympy.sin(x) / x

    def rsr_sin(x, y):
        return (sympy.sin(x) ** 2 - sympy.sin(y) ** 2) / sympy.sin(x - y)

    def rsr_x(x, y):
        return x + y

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f, rsr_sin, rsr_x])[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_sinc():
    def F(x):
        return math.sin(x) / x

    def P(x, y):
        return x + y

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "sin(x)", "sin(y)", "sin(x-y)", "P(x,y)"],
        ["f(x+y)", "sin(x)", "sin(y)", "sin(x-y)", "p(x,y)"],
        [F, P],
        sum,
        max_degree=3,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return sympy.sin(x) / x

    def p(x, y):
        return x + y

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f, p])[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_log():
    def F(x):
        return math.log(x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x*y)", "F(x)", "F(y)"],
        ["f(x*y)", "f(x)", "f(y)"],
        [F],
        max_degree=1,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=50,
    )

    def f(x):
        return sympy.log(x)

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f])[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_sec():
    def F(x):
        return 1 / math.cos(x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=3,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return 1 / sympy.cos(x)

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f])[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_csc():
    def F(x):
        return 1 / math.sin(x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],  # function basis aka the template
        # ["F(x+y)", "F(x)", "F(y)"],  # how to call functions
        # ["f(x+y)", "f(x)", "f(y)"],  # function basis aka the template
        [F],
        max_degree=4,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=200,
    )

    def f(x):
        return 1 / sympy.sin(x)

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f])[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_square_loss():
    def F(v):
        return (1 - v) ** 2

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        [F],
        max_degree=2,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return (1 - x) ** 2

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f])[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_savage_loss():
    def F(v):
        return 1 / (1 + math.exp(v)) ** 2

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        [F],
        max_degree=4,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=100,
    )

    def f(x):
        return 1 / (1 + sympy.exp(x)) ** 2

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_savage_loss_library():
    def F(x):
        return 1 / (1 + math.exp(x)) ** 2

    def G(x, y):
        return math.exp(x) * math.exp(y)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "G(x, y)", "F(x)", "F(y)"],
        ["f(x+y)", "g(x, y)", "f(x)", "f(y)"],
        [F, G],
        max_degree=3,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=100,
    )

    def f(x):
        return 1 / (1 + sympy.exp(x)) ** 2

    def g(x, y):
        return sympy.exp(x) * sympy.exp(y)

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f, g])[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )

    # f(x+y)*g(x)*g(y)**2 + 2*f(x+y)*g(x)*g(y) + f(x+y) - 1 = 0

    # f(x+y) = \frac{1}{g(x) \cdot g(y)^2 + 2 \cdot g(x) \cdot g(y) + 1}

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_savage_loss_basis():
    def F(x):
        return 1 / (1 + math.exp(x)) ** 2

    def G(x):
        return math.exp(x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "G(x)", "G(y)"],
        ["f(x+y)", "g(x)", "g(y)"],
        [F, G],
        max_degree=5,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=100,
    )

    def f(x):
        return 1 / (1 + sympy.exp(x)) ** 2

    def g(x):
        return sympy.exp(x)

    for eq in equations[0]["vtrace1"]:
        if verify(eq, [f, g])[0]:
            print("Isolating f(x+y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x+y)")
                )
            )
            print("Isolating f(x-y):")
            print(
                sympy.solve(
                    sympy.Eq(sympy.sympify(str(eq.lhs)), 0), sympy.sympify("f(x-y)")
                )
            )

    print(f"Sample complexity: {equations[2]['vtrace1']}")


def test_relu():
    def F(x):
        return max(0, x)

    equations = infer_property(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)"],
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        [F],
        max_degree=3,
        # milp=MILPSolver.GUROBI,
        # method=InitialMethod.EAGER_MILP,
        n=150,
    )

    def f(x):
        return sympy.Max(0, x)

    for eq in equations[0]["vtrace1"]:
        verify(eq, [f])

    print(f"Sample complexity: {equations[2]['vtrace1']}")


if __name__ == "__main__":
    st = time()

    # test_identity()  # 1
    # test_exp()  # 2
    # test_exp_minus_one()  # 3
    # test_exp_div_by_x()  # no solution # 4
    # test_exp_div_by_x_composite()  # 5
    # test_floudas()  # 6
    # test_mean()  # 7
    # test_tan()  # 8
    # test_cot()  # 9
    # test_diff_x2_y2()  # 10
    # test_inverse_square()  # 11
    # test_inverse()  # 12
    # test_inverse_add()  # 13
    # test_inverse_cot_plus_one()  # 14
    # test_inverse_tan_plus_one()  # 15
    # test_x_over_one_minaus_x()  # 16
    # test_minus_x_over_one_minus_x()  # 17
    # test_sin_over_sin()  # TODO fails
    # test_sinh_over_sinh()  # TODO fails
    # test_cos()  # 18
    # test_cosh()  # 19
    test_squared()  # 20
    # test_sin()  # 21
    # test_sin_glibc()
    # test_sinh()  # 22
    # test_cube()  # 23
    # test_log()  # 24
    # test_sec()  # 25
    # test_csc()  # 26
    # https://en.wikipedia.org/wiki/Sinc_function
    # test_sinc()  # 27
    # test_sinc_composite()  # 28
    # test_arctan()  # No solution
    # test_mod()  # 29
    # test_mod_mult()  # 30
    # test_int_mult()  # 31
    # test_int_mult_3()  # 31
    # test_int_add_3()  # 31
    #### ACTIVATION FUNCTIONS ####
    # test_tanh()  # 32
    # test_sigmoid()  # 33
    # test_sigmoid_extra()
    # test_sigmoid_derivative()
    # https://en.wikipedia.org/wiki/Softmax_function
    # test_softmax2_1()  # 34
    # test_softmax2_2()  # 35
    # test_softmax2_alt_1()  # no solution

    # https://en.wikipedia.org/wiki/Logistic_function
    # test_logistic()  # 36
    # test_logistic1()  # 37

    # loss functions:https://en.wikipedia.org/wiki/Loss_functions_for_classification
    # test_square_loss()  # 38
    # test_savage_loss()  # no solution
    # test_savage_loss_library()  # 39 library
    # test_savage_loss_basis()  # 40

    # test_relu()

    log.debug(f"Total Time: {time() - st:.2f}s")
