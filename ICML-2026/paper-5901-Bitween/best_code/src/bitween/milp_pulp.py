import numpy as np
import pulp
from pulp import (
    LpContinuous,
    LpInteger,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
)
from sympy import Rational, Symbol, simplify

from bitween.config import Config, MILPSolver
from bitween.miscs import getLogger
from bitween.utilities import pp

config = Config()
log = getLogger(__name__, config.logger_level)

# Constants are the same
ERROR_BOUND = 0.1
ROUND = None
OPTIMAL = 1  # In PuLP, the status code for optimal is 1
MSG = 0


def milp_synthesis(
    data: np.ndarray,
    terms: list[str],
    pivot: str,
    blocked: str = None,
    bound=config.bound,
    timeout=config.milp_timeout,
    scale=1,
    solver=MILPSolver.GLPK,
) -> tuple[int, str | None, float | None, list[str]]:

    global message
    message = ""

    def _print(e):
        global message
        message += str(e) + "\n"

    if MSG:
        _print("------------------------------------------")
    m = LpProblem("synthesis_of_random_self_reducible_properties", LpMinimize)

    _print("------------------------------------------")
    _print(f"Pivot: {pivot} | Blocked: {blocked} | Solver: {solver}")

    term_vars = [
        LpVariable(f"term_{i}", -bound, bound, LpInteger) for i in range(len(terms) + 1)
    ]  # +1 for the constant term

    # Set pivot and blocked constraints
    pivot_index = terms.index(pivot)
    term_vars[pivot_index].setInitialValue(1)
    term_vars[pivot_index].fixValue()

    if blocked is not None:
        blocked_index = terms.index(blocked)
        term_vars[blocked_index].setInitialValue(0)
        term_vars[blocked_index].fixValue()

    error_vars = [
        LpVariable(f"error_{i}", -ERROR_BOUND, ERROR_BOUND, LpContinuous)
        for i in range(len(data))
    ]

    # Define constraints
    for i, data_point in enumerate(data):
        expr = (
            lpSum(
                [data_point[j] * scale * term_vars[j] for j in range(len(data_point))]
            )
            + term_vars[-1]
        )
        m += (expr == error_vars[i], f"error_{i}_ub")

    # Objective function
    m += lpSum(error_vars)

    # m.writeLP("synthesis_pulp.lp")

    # NOTE: Solver Selection and solve the problem
    if solver == MILPSolver.PULP:
        m.solve(pulp.PULP_CBC_CMD(msg=MSG, timeLimit=timeout))
    elif solver == MILPSolver.GLPK:
        m.solve(pulp.GLPK_CMD(msg=MSG, timeLimit=timeout))
    elif solver == MILPSolver.GUROBI:
        m.solve(pulp.GUROBI_CMD(msg=MSG, timeLimit=timeout))
    else:
        raise ValueError(f"Unknown solver: {solver}")

    status = LpStatus[m.status]

    if status != "Optimal":
        _print("------------------------------------------")
        _print(f"Optimization ended with status {status}: ({pivot})")
        _print("------------------------------------------")
        print(message)
        return status, None, None, None, None

    _print("------------------------------------------")
    # Post-processing to generate expression and term costs
    expr = 0
    terms.append("1")  # NOTE: add the last constant as a term

    term_costs = {}
    term_coefs: dict[str, float] = {}

    for v in m.variables():
        if v.name.startswith("term_"):
            max_len = max([len(x) for x in terms])
            max_val = max([len(str(pp(round(v.varValue, 5)))) for v in m.variables()])
            cost = ""
            value = v.varValue
            if value != 0:
                coeff = value
                term = terms[int(v.name.split("_")[1])]
                term_ = Rational(coeff) * (1 if term == "1" else Symbol(term))
                # TODO: fix this later. There are multiplicative terms such as f(x)*f(x)
                # cost = sympify(f"{coeff}*{term}").count_ops()
                cost = term_.count_ops()
                term_costs[term] = cost
                term_coefs[term] = value
                cost = f"cost: {cost}"
                expr += term_
            _print(
                f"%8s | %{max_len}s = %{max_val}s"
                % (v.name, terms[int(v.name[5:])], pp(value))
                + f" | {cost}"
            )

    expr = simplify(expr)  # NOTE is simplifying the expression necessary?
    obj_val = m.objective.value()

    _print("------------------------------------------")
    if status == "Optimal":
        _print(f"Optimal objective for `{pivot}`: {obj_val}")
    else:
        _print(f"Optimization ended with status {status}")

    print(message)
    return (
        m.status,
        expr,
        obj_val,
        term_coefs,
        sorted(term_costs.keys(), key=lambda x: term_costs[x]),
    )


if __name__ == "__main__":  # noqa E123
    """
    Test cases
    """
    import csv

    import pulp
    from sympy import Function, Symbol, sin

    from bitween.analyzer import property_test, verify
    from bitween.sampler import Distribution, Domain, sample
    from bitween.terms import get_values_terms

    solver_list = pulp.listSolvers(onlyAvailable=True)
    print(solver_list)

    domain = Domain.Real
    distribution = Distribution(np.random.uniform, low=-5, high=5)
    degree = 2
    bound = 2
    blocked = None
    pivot = "f(x)*f(x)"

    f = Function("f")
    x = Symbol("x")

    def f(x):
        return sin(x)

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

    print("------------------------------------------")

    # clear the file
    with open("data.csv", mode="w", newline="") as file:
        file.write("")  # Write an empty string to clear the file

    with open("data.csv", mode="a", newline="\n") as file:
        writer = csv.writer(file)

        important = None

        i = 30
        while i > 0:
            x, y = tuple(sample(distribution, ["x", "y"]).values())

            (values, terms) = get_values_terms(
                degree,
                [F(x + y), F(x - y), F(x), F(y), 1],
                ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],
                "1",
            )

            # putting important term to the end
            important = terms.index(pivot)
            values = values[:important] + values[important + 1 :] + [values[important]]
            terms = terms[:important] + terms[important + 1 :] + [terms[important]]

            # removing constant
            constant = terms.index("1")
            values = values[:constant] + values[constant + 1 :]
            terms = terms[:constant] + terms[constant + 1 :]

            i -= 1

            writer.writerow(values)

        print(terms)

    # Load the CSV file into a numpy array
    data = np.genfromtxt("data.csv", delimiter=",")

    status, expr, obj, _, terms = milp_synthesis(data, terms, pivot, blocked, bound)
    print("-costs----------------")
    if terms:
        for term in terms:
            print(f"{term}")
    print("---------------------")
    assert status == OPTIMAL
    assert abs(obj) <= 1e-11
    expr_s = str(expr)
    assert expr_s == "f(x)*f(x) - f(x-y)*f(x+y) - f(y)*f(y)"
    assert verify(expr_s, f)
    print("---------------------")
    assert property_test(expr_s, F)

    print("------------------------------------------")
