import gurobipy as gp
import numpy as np
from gurobipy import GRB
from sympy import sympify

from bitween.utilities import pp

INTEGRALITY_FOCUS = 1
# MIPFOCUS = 2
NUMERIC_FOCUS = 3
# OPTIMALITY_TOL = 1e-9
# SOS1 = False
ERROR_BOUND = 0.1  # 0.05
ROUND = None

OPTIMAL = GRB.OPTIMAL

__version__ = "1.0.0"


def milp_synthesis(  # noqa: C901
    data: np.ndarray,
    terms: list[str],
    pivot: str,  # pivot term
    blocked: str = None,  # blocked term
    bound=2,
    timeout=10,  # 10 seconds
    scale=1,
    sr_vals: list[float] = [],
    sr_func: str | None = None,
) -> tuple[int, str | None, float | None, list[str]]:
    """
    Parameters
    ----------
    data : np.ndarray
        list of datapoints
    terms : list[str]
        list of term names
    pivot : str
        pivot term
    blocked : str, optional
        blocked term, by default None
    bound : int, optional
        bound, by default 2
    scale : int, optional
        scale, by default 1

    Returns
    -------
    status: the status of the optimization (e.g. GRB.OPTIMAL)
    expr: the synthesized property
    obj_val: the objective value
    """
    print("------------------------------------------")
    m = gp.Model("synthesis of random self-reducible properties")
    m.setParam("TimeLimit", timeout)
    m.setParam("IntegralityFocus", INTEGRALITY_FOCUS)
    # m.setParam("MIPFocus", MIPFOCUS)
    # m.setParam("OptimalityTol", OPTIMALITY_TOL)
    m.setParam("NumericFocus", NUMERIC_FOCUS)

    # For each term, create an integer decision variable and
    for i in range(len(terms)):
        m.addVar(vtype=GRB.INTEGER, name=f"term_{i}", lb=-bound, ub=bound)
        m.update()
        if terms[i] == pivot:  # e.g. "f(x+y)"
            var = m.getVarByName(f"term_{i}")
            m.addConstr(var == 1, name=f"term_{i}_ub")
            # m.getVarByName(f"term_{i}").setAttr("BranchPriority", 1000)
        if blocked is not None and terms[i] == blocked:  # e.g. "f(x-y)*f(x+y)"
            var = m.getVarByName(f"term_{i}")
            m.addConstr(var == 0, name=f"term_{i}_ub")

    # NOTE: add the last constant as a term
    m.addVar(vtype=GRB.INTEGER, name=f"term_{len(terms)}", lb=-bound, ub=bound)
    m.update()

    # m.addConstr(gp.quicksum(m.getVars()) >= -20, name="sum_terms")
    # m.addConstr(gp.quicksum(m.getVars()) <= -15, name="sum_terms")

    abs_vars_name = {}
    abs_vars_expr = {}
    for i in range(len(data)):
        # create a continous error variable.
        var_error = m.addVar(
            vtype=GRB.CONTINUOUS, name=f"error_{i}", lb=-ERROR_BOUND, ub=ERROR_BOUND
        )
        abs_vars_name[f"error_{i}"] = var_error
        vals = []
        for j in range(len(data[i])):
            # TODO: round the value of X[i][j] to ? decimal places
            if ROUND is None:
                vals.append(data[i][j] * scale * m.getVarByName(f"term_{j}"))
            else:
                vals.append(
                    round(data[i][j], ROUND) * scale * m.getVarByName(f"term_{j}")
                )

        # add the last constant as a term
        vals.append(m.getVarByName(f"term_{len(terms)}"))

        # create an expression for the sum of the datapoint times the decision variable.
        sum_terms = gp.quicksum(vals)
        abs_vars_expr[f"error_{i}"] = sum_terms
        try:
            if len(sr_vals) > 0:
                m.addConstr(sr_vals[i] == sum_terms + var_error, name=f"error_{i}_ub")
            else:
                m.addConstr(sum_terms == var_error, name=f"error_{i}_ub")
        except gp.GurobiError as e:
            print("Error code " + str(e.errno) + ": " + str(e))
            return GRB.ERROR_NUMERIC, None, None, None

        # m.addConstr(sum_terms <= ERROR_BOUND, name=f"error_{i}_ub")
        # m.addConstr(sum_terms >= -ERROR_BOUND, name=f"error_{i}_lb")

    # convexify the absolute value function in the objective function
    # NOTE: set objective function
    for name in abs_vars_name.keys():
        var = abs_vars_name[name]
        expr = abs_vars_expr[name]
        # m.addConstr(expr <= var, name=f"sum_{name}_ub")
        # m.addConstr(-expr <= var, name=f"sum_{name}_lb")

    # m.addConstr(gp.quicksum(abs_vars_name.values()) <= 1e4, name="obj_ub")

    m.setObjective(gp.quicksum(abs_vars_name.values()), GRB.MINIMIZE)

    # m.write("synthesis.lp")

    try:
        print("------------------------------------------")
        m.optimize()
    except gp.GurobiError as e:
        print("Error code " + str(e.errno) + ": " + str(e))
        exit()

    print("------------------------------------------")
    if m.Status == GRB.OPTIMAL:
        print("Optimal objective: %g" % m.ObjVal)
    elif m.Status == GRB.INF_OR_UNBD:
        print("Model is infeasible or unbounded")
        print("------------------------------------------")
        print(f"scale: {scale:e}")
        print("------------------------------------------")
        return m.Status, None, None, None
    elif m.Status == GRB.INFEASIBLE:
        print("Model is infeasible")
        print("------------------------------------------")
        print(f"scale: {scale:e}")
        print("------------------------------------------")
        return m.Status, None, None, None
    elif m.Status == GRB.UNBOUNDED:
        print("Model is unbounded")
        return m.Status, None, None, None
    else:
        print("Optimization ended with status %d" % m.Status)
        return m.Status, None, None, None

    expr = ""
    first_term = True
    terms.append("1")  # NOTE: add the last constant as a term

    term_costs = {}
    for v in m.getVars():
        # if v.X != 0:
        #     print('%s %g' % (v.VarName, v.X))
        if v.VarName.startswith("term_"):  # NOTE
            # find the length of the longest term interm_names
            max_len = max([len(x) for x in terms])
            cost = ""
            if v.X != 0:
                coeff = ""
                if v.X > 0 and not first_term:
                    coeff = " + "
                if v.X > 0 and first_term:
                    coeff = ""
                if v.X < 0:
                    coeff = " - "
                if v.X != 1 and v.X != -1:
                    coeff += f"{pp(abs(v.X))}*"
                term = terms[int(v.VarName[5:])]
                term_ = f"{coeff}{terms[int(v.VarName[5:])]}"
                cost = sympify(term_).count_ops()
                term_costs[term] = cost
                cost = f"cost: {cost}"
                expr += f"{coeff}{terms[int(v.VarName[5:])]}"
                first_term = False
            print(
                f"%8s | %{max_len}s = %3s"
                % (v.VarName, terms[int(v.VarName[5:])], pp(v.X))
                + f" | {cost}"
            )

    # for v in m.getVars():
    #     if v.VarName.startswith("error_"):  # NOTE
    #         print('%s %g' % (v.VarName, v.X))

    expr = expr.strip()
    print("------------------------------------------")
    print(f"Time: {round(m.runTime, 2)}s")
    print("Optimal objective: %g" % m.ObjVal)
    print("------------------------------------------", end=" ")
    m.printStats()
    print("------------------------------------------", end=" ")
    m.printQuality()
    print("------------------------------------------")
    if sr_func is not None:
        print(
            f"milp property: {sr_func} = {expr} ({m.ObjVal}) (block: {blocked}) (bound: {bound})"
        )
    else:
        print(
            f"milp property: {expr} = 0 ({m.ObjVal}) (block: {blocked}) (bound: {bound})"
        )
    print("------------------------------------------")
    print(f"scale: {scale:e}")

    return (
        m.Status,
        expr,
        m.ObjVal,
        sorted(term_costs.keys(), key=lambda x: term_costs[x]),
    )


if __name__ == "__main__":  # noqa E123
    """
    Test cases
    """
    import csv

    from sympy import Function, Symbol, sin

    from bitween.analyzer import property_test, verify
    from bitween.sampler import Distribution, Domain, sample
    from bitween.terms import get_values_terms

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

    status, expr, obj, terms = milp_synthesis(data, terms, pivot, blocked, bound)
    print("-costs----------------")
    for term in terms:
        print(f"{term}")
    print("---------------------")
    assert status == GRB.OPTIMAL
    assert abs(obj) <= 1e-12
    assert expr == "- f(x-y)*f(x+y) - f(y)*f(y) + f(x)*f(x)"
    assert verify(expr, f)
    print("---------------------")
    assert property_test(expr, F)

    print("------------------------------------------")
