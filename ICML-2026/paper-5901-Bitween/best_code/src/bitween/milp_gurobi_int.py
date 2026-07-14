# import math
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from sympy import Rational, Symbol, simplify, sympify

from bitween.config import Config
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
    bound=2,
    timeout=10,  # 10 seconds
    scale=1,
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

    p("------------------------------------------")
    p(f"Pivot: {pivot} | Blocked: {blocked}")

    with gp.Env(empty=True) as env:
        if not config.milp_warnings:
            env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(f"{pivot}: {terms}", env=env)
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

        for i in range(len(data)):
            # create a continous error variable.
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
            try:
                m.addConstr(gp.quicksum(vals) == 0, name=f"eq_{i}_ub")
            except gp.GurobiError as e:
                p("------------------------------------------")
                p("Error code " + str(e.errno) + ": " + str(e))
                return GRB.ERROR_NUMERIC, None, None, None

        # m.setObjective(gp.quicksum(0), GRB.MINIMIZE)

        m.write("synthesis.lp")

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
                    cost = sympify(f"{coeff}*{term}").count_ops()
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
        p(f"Time: {round(m.runTime, 2)}s")
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


if __name__ == "__main__":
    pass
