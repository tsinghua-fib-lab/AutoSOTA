import gurobipy as gp
import numpy as np
from gurobipy import GRB
from sympy import Rational, Symbol, simplify

from bitween.config import Config, MILPSolver
from bitween.miscs import getLogger
from bitween.utilities import pp

config = Config()
log = getLogger(__name__, config.logger_level)

INTEGRALITY_FOCUS = 1
# MIPFOCUS = 2
NUMERIC_FOCUS = 3
# OPTIMALITY_TOL = 1e-9
# SOS1 = False
ERROR_BOUND = 0.1  # 0.05
ROUND = None

OPTIMAL = GRB.OPTIMAL


def milp_synthesis(  # noqa: C901
    data: np.ndarray,
    terms: list[str],
    pivot: str,  # pivot term
    blocked: str = None,  # blocked term
    bound=config.bound,
    timeout=config.milp_timeout,
    scale=1,
    solver=MILPSolver.GUROBI,
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
    term_coefs: the coefficients of each term
    term_costs: the cost of each term
    """
    global str_
    str_ = ""

    def p(s):
        global str_
        str_ += s + "\n"

    if solver != MILPSolver.GUROBI:
        log.error("Gurobi is the only supported MILP solver")
        raise NotImplementedError
    p("------------------------------------------")
    p(f"Pivot: {pivot} | Blocked: {blocked} | Solver: {solver}")

    with gp.Env(empty=True) as env:
        if not config.milp_warnings:
            env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(f"{pivot}", env=env)
        m.setParam("TimeLimit", timeout)
        m.setParam("IntegralityFocus", INTEGRALITY_FOCUS)
        # m.setParam("MIPFocus", MIPFOCUS)
        # m.setParam("OptimalityTol", OPTIMALITY_TOL)
        m.setParam("NumericFocus", NUMERIC_FOCUS)

        # find the min and max values of the data, and find the matrix range, which is max/min
        # min_coef = np.min(data)
        # if min_coef == 0:
        #     min_coef = 1
        # max_coef = np.max(data)
        # matrix_range = max_coef / min_coef
        # # assign scale half of the matrix range until it is less than 10e9
        # if abs(matrix_range) > 10e9:
        #     log.warning(
        #         f"Matrix range is too large(max: {max_coef}; min: {min_coef}), scaling the data for pivot: {pivot}"
        #     )
        #     exponent = math.pow(10, int("{:.1e}".format(matrix_range).split("e")[1]))
        #     while abs(exponent) > 10e9:
        #         exponent = exponent / 10**4
        #     scale = 1 / matrix_range

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
                # TODO round the value of X[i][j] to ? decimal places
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
                m.addConstr(sum_terms == var_error, name=f"error_{i}_ub")
            except gp.GurobiError as e:
                p("------------------------------------------")
                p("Error code " + str(e.errno) + ": " + str(e))
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
            m.optimize()

        except gp.GurobiError as e:
            p("------------------------------------------")
            p("Error code " + str(e.errno) + ": " + str(e))
            exit()

        if m.Status == GRB.OPTIMAL:
            p("------------------------------------------")
            p("Optimal objective: %g" % m.ObjVal)
        elif m.Status == GRB.INF_OR_UNBD:
            p("------------------------------------------")
            p("Model is infeasible or unbounded")
            p("------------------------------------------")
            p(f"scale: {scale:e}")
            p("------------------------------------------")
            return m.Status, None, None, None, None
        elif m.Status == GRB.INFEASIBLE:
            p("------------------------------------------")
            p("Model is infeasible")
            p("------------------------------------------")
            p(f"scale: {scale:e}")
            p("------------------------------------------")
            return m.Status, None, None, None, None
        elif m.Status == GRB.UNBOUNDED:
            p("------------------------------------------")
            p("Model is unbounded")
            return m.Status, None, None, None, None
        else:
            p("------------------------------------------")
            p("Optimization ended with status %d" % m.Status)
            return m.Status, None, None, None, None

        expr = 0
        terms.append("1")  # NOTE: add the last constant as a term

        term_costs = {}
        term_coefs: dict[str, float] = {}
        for v in m.getVars():
            # if v.X != 0:
            #     p('%s %g' % (v.VarName, v.X))
            if v.VarName.startswith("term_"):  # NOTE
                # find the length of the longest term interm_names
                max_len = max([len(x) for x in terms])
                cost = ""
                if v.X != 0:
                    coeff = v.X
                    term = terms[int(v.VarName[5:])]
                    term_ = Rational(coeff) * (1 if term == "1" else Symbol(term))
                    # TODO: fix this later. There are multiplicative terms such as f(x)*f(x)
                    # cost = sympify(f"{coeff}*{term}").count_ops()
                    cost = term_.count_ops()
                    term_costs[term] = cost
                    term_coefs[term] = v.X
                    cost = f"cost: {cost}"
                    expr += term_
                p(
                    f"%8s | %{max_len}s = %3s"
                    % (v.VarName, terms[int(v.VarName[5:])], pp(v.X))
                    + f" | {cost}"
                )

        # for v in m.getVars():
        #     if v.VarName.startswith("error_"):
        #         p('%s %g' % (v.VarName, v.X))

        expr = simplify(expr)  # NOTE is simplifying the expression necessary?
        p("------------------------------------------")
        p(f"Time: {round(m.runTime, 3)}s")
        p("Optimal objective: %g" % m.ObjVal)
        m.printStats()
        m.printQuality()
        p(f"milp property: {expr} = 0 ({m.ObjVal}) (block: {blocked}) (bound: {bound})")
        p("------------------------------------------")
        p(f"scale: {scale:e}")

        print(str_)

        return (
            m.Status,
            expr,
            m.ObjVal,
            term_coefs,
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

    status, expr, obj, _, terms = milp_synthesis(data, terms, pivot, blocked, bound)
    print("-costs----------------")
    for term in terms:
        print(f"{term}")
    print("---------------------")
    assert status == GRB.OPTIMAL
    assert abs(obj) <= 1e-12
    expr_s = str(expr)
    assert expr_s == "f(x)*f(x) - f(x-y)*f(x+y) - f(y)*f(y)"
    assert verify(expr_s, f)
    print("---------------------")
    assert property_test(expr_s, F)

    print("------------------------------------------")
