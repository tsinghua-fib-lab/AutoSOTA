import argparse
import math
import os
from time import time

import numpy as np
import sympy

from bitween.analyzer import log as anlz_log
from bitween.analyzer import verify_with_timeout
from bitween.config import Config, Method, MILPSolver
from bitween.main import infer_property_with_timeout
from bitween.main import log as main_log
from bitween.miscs import addFileHandler, getLogger, removeFileHandler
from bitween.reducer import log as rdc_log
from bitween.sampler import Distribution, Domain

config = Config()
log = getLogger(__name__, config.logger_level, empty_format=True)


def evaluate(
    domain: Domain,
    distribution: Distribution,
    exprs: list[str],
    infer_funcs: list[str],
    sympy_funcs: list[callable],
    test_id: str,
    res_dir: str,
    method: Method,
    timeout_sec: float,
    milp: MILPSolver = None,
    max_degree: int = 2,
    n: int = 30,
    epsilon: float = 0.001,
    preconditions: dict[str, callable] = None,
    constants: dict = None,
    var_bound: int = None,
    isolate_terms: list[str] = None,
) -> list[any]:
    trace_file = os.path.join(res_dir, f"{test_id}_trace.csv")
    out_file = os.path.join(res_dir, f"{test_id}.txt")

    if os.path.exists(out_file):
        os.remove(out_file)

    log_list = [log, main_log, anlz_log, rdc_log]
    fch_list = [addFileHandler(_log, out_file, empty_format=True) for _log in log_list]

    log.info(f"Starting {test_id}")

    try:
        st = time()
        eqs_dct, error_dct, sample_comp_dct, error = infer_property_with_timeout(
            domain=domain,
            distribution=distribution,
            exprs=exprs,
            template=exprs,
            functions=infer_funcs,
            max_degree=max_degree,
            n=n,
            epsilon=epsilon,
            preconditions=preconditions,
            milp=milp,
            var_bound=var_bound,
            method=method,
            trace_file=trace_file,
            timeout_sec=timeout_sec,
        )

        if error:
            log.error(f"Error found evaluating {test_id}: {error}")
            return []

        found_eqs = eqs_dct["vtrace1"]
        mean_error = error_dct["vtrace1"]
        sample_complexity = sample_comp_dct["vtrace1"]

        log.info(
            f"\nEquations found: {len(found_eqs)}, "
            f"Mean Error: {mean_error}, "
            f"Sample Complexity: {sample_complexity}\n"
        )

        categories = ["verified", "unverified", "faulty", "unknown"]
        equations = {key: [] for key in categories}

        for eq in found_eqs:
            ok, error_msg = verify_with_timeout(eq, sympy_funcs, domain, constants)

            if error_msg:
                is_faulty = error_msg.startswith("Exception")
                key = "faulty" if is_faulty else "unverified"
            else:
                key = "verified" if ok else "unknown"

            equations[key].append((eq, error_msg))

        took_time = time() - st

        def fmt_pair(pair):
            eq, msg = pair
            return f"{eq} | {msg}" if msg else f"{eq}"

        for key in categories:
            eqs = equations[key]
            eqs_len = len(eqs)
            if eqs_len > 0:
                eqs_str = "\n".join(map(fmt_pair, eqs))
                log.info(f"\n{key.capitalize()} ({eqs_len}):\n{eqs_str}\n")

        log.info(f"Took time: {took_time:.2f}s")

        if not isolate_terms:
            return

        for veq, _ in equations["verified"]:
            for iso_term in isolate_terms:
                sol_list = sympy.solve(
                    sympy.Eq(sympy.sympify(str(veq.lhs)), 0),
                    sympy.sympify(iso_term),
                )
                log.info(f"Isolating {iso_term}:\n{sol_list}\n")

    except Exception as e:
        log.error(f"Exception found evaluating {test_id}", exc_info=e)

    finally:
        log.info(f"Ending {test_id}")

        if os.path.exists(trace_file):
            os.remove(trace_file)

        for _log, _fch in zip(log_list, fch_list):
            removeFileHandler(_log, _fch)


def test_identity(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    _c = 5

    def f(x, c=_c):
        return c * x

    def _sp_f(x, c=_c):
        return c * x

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="01_identity",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=1,
        n=10,
        var_bound=20,
        constants={"c": _c},
        timeout_sec=timeout_sec,
    )


def test_exp(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return math.exp(x)

    def _sp_f(x):
        return sympy.exp(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="02_exp",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=10,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_exp_minus_one(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return math.exp(x) - 1

    def _sp_f(x):
        return sympy.exp(x) - 1

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="03_exp_minus_one",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=20,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_exp_div_by_x(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return math.exp(x) / x

    def _sp_f(x):
        return sympy.exp(x) / x

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-2, high=2),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="04_exp_div_by_x",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        var_bound=20,
        n=200,
        timeout_sec=timeout_sec,
    )


def test_exp_div_by_x_composite(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return math.exp(x) / x

    def h(x):
        return math.exp(x)

    def p(x, y):
        return x + y

    def _sp_f(x):
        return sympy.exp(x) / x

    def _sp_h(x):
        return sympy.exp(x)

    def _sp_p(x, y):
        return x + y

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-2, high=2),
        exprs=["f(x+y)", "h(x)", "h(y)", "p(x,y)"],
        infer_funcs=[f, h, p],
        sympy_funcs=[_sp_f, _sp_h, _sp_p],
        test_id="05_exp_div_by_x_composite",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=10,
        timeout_sec=timeout_sec,
        # var_bound=20,
    )


def test_floudas(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x, y):
        return x + y

    def _sp_f(x, y):
        return x + y

    def pre_f(x, y):
        return 0 <= x <= 2 and 0 <= y <= 3 and x + y <= 4

    evaluate(
        domain=Domain.Positive_Real,
        distribution=Distribution(np.random.uniform, low=0, high=2),
        exprs=[
            "f(x1+x2,y1+y2)",
            "f(x2+x3,y2+y3)",
            "f(x1+x3,y1+y3)",
            "f(x1,y1)",
            "f(x2,y2)",
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="06_floudas",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=1,
        n=15,
        var_bound=20,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_mean(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x, y, z):
        return 1 / 3 * (x + y + z)

    def _sp_f(x, y, z):
        return 1 / 3 * (x + y + z)

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=[
            "f(x1+x2+x3, y1+y2+y3, z1+z2+z3)",
            "f(x1+x2,y1+y2,z1+z2)",
            "f(x2+x3,y2+y3,z2+z3)",
            "f(x1+x3,y1+y3,z1+z3)",
            "f(x1,y1,z1)",
            "f(x2,y2,z2)",
            "f(x3,y3,z3)",
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="07_mean",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=1,
        n=15,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_tan(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return math.tan(x)

    def _sp_f(x):
        return sympy.tan(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="08_tan",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=30,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_cot(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 1 / math.tan(x)

    def _sp_f(x):
        return 1 / sympy.tan(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="09_cot",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=15,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_diff_squares(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x, y):
        return x**2 - y**2

    def _sp_f(x, y):
        return x**2 - y**2

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=[
            "f(x-a,y-b)",
            "f(x+a,y-b)",
            "f(x,y)",
            "f(a,b)",
            "f(x-a,y)",
            "f(x+a,y)",
            "f(x,y-a)",
            "f(x,y+a)",
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="10_diff_squares",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_inverse_square(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 1 / (x**2)

    def _sp_f(x):
        return 1 / (x**2)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="11_inverse_square",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=20,
        isolate_terms=["f(x+y)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )


def test_inverse(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 1 / x

    def _sp_f(x):
        return 1 / x

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="12_inverse",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_inverse_add(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 1 / (x + 1)

    def _sp_f(x):
        return 1 / (x + 1)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="13_inverse_add",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=20,
        isolate_terms=["f(x+y)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )


def test_inverse_cot_plus_one(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 1 / (1 / math.tan(x) + 1)

    def _sp_f(x):
        return 1 / (1 / sympy.tan(x) + 1)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="14_inverse_cot_plus_one",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=15,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_inverse_tan_plus_one(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 1 / (math.tan(x) + 1)

    def _sp_f(x):
        return 1 / (sympy.tan(x) + 1)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="15_inverse_tan_plus_one",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=40,
        var_bound=20,
        isolate_terms=["f(x+y)", "f(x)"],
        timeout_sec=timeout_sec,
    )


def test_x_over_one_minus_x(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return x / (1 - x)

    def _sp_f(x):
        return x / (1 - x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="16_x_over_one_minus_x",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_minus_x_over_one_minus_x(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return -x / (1 - x)

    def _sp_f(x):
        return -x / (1 - x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="17_minus_x_over_one_minus_x",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_cos(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return math.cos(x)

    def _sp_f(x):
        return sympy.cos(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="18_cos",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_cosh(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return math.cosh(x)

    def _sp_f(x):
        return sympy.cosh(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="19_cosh",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_squared(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return x**2

    def _sp_f(x):
        return x**2

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="20_squared",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_sin(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x, terms=20):
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

    def _sp_f(x):
        return sympy.sin(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="21_sin",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=20,
        isolate_terms=["f(x+y)", "f(x)"],
        timeout_sec=timeout_sec,
    )


def test_sinh(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return math.sinh(x)

    def _sp_f(x):
        return sympy.sinh(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="22_sinh",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=20,
        isolate_terms=["f(x+y)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )


def test_cube(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return x**3

    def _sp_f(x):
        return x**3

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="23_cube",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=20,
        var_bound=20,
        isolate_terms=["f(x+y)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )


def test_log(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return math.log(x)

    def _sp_f(x):
        return sympy.log(x)

    def pre_f(x):
        return x > 0

    evaluate(
        domain=Domain.Positive_Real,
        distribution=Distribution(np.random.uniform, low=0, high=5),
        exprs=["f(x*y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="24_log",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=1,
        n=10,
        var_bound=10,
        preconditions={"f": pre_f},
        isolate_terms=["f(x+y)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )


def test_sec(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 1 / math.cos(x)

    def _sp_f(x):
        return 1 / sympy.cos(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="25_sec",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=50,
        var_bound=10,
        # isolate_terms=["f(x+y)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )


def test_csc(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 1 / math.sin(x)

    def _sp_f(x):
        return 1 / sympy.sin(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="26_csc",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=4,
        n=200,
        var_bound=10,
        timeout_sec=timeout_sec,
    )


def test_sinc(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return math.sin(x) / x

    def p(x, y):
        return x + y

    def _sp_f(x):
        return sympy.sin(x) / x

    def _sp_p(x, y):
        return x + y

    def pre_f(x):
        return x != 0

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "sin(x)", "sin(y)", "sin(x-y)", "p(x,y)"],
        infer_funcs=[f, p, sum],
        sympy_funcs=[_sp_f, _sp_p],
        test_id="27_sinc",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=50,
        var_bound=20,
        preconditions={"f": pre_f},
        isolate_terms=["f(x+y)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )


def test_sinc_composite(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return math.sin(x) / x

    def rsr_sin(x, y):
        return (math.sin(x) ** 2 - math.sin(y) ** 2) / math.sin(x - y)

    def rsr_x(x, y):
        return x + y

    def _sp_f(x):
        return sympy.sin(x) / x

    def _sp_rsr_sin(x, y):
        return (sympy.sin(x) ** 2 - sympy.sin(y) ** 2) / sympy.sin(x - y)

    def _sp_rsr_x(x, y):
        return x + y

    def pre_f(x):
        return x != 0

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "rsr_sin(x,y)", "rsr_x(x,y)"],
        infer_funcs=[f, rsr_sin, rsr_x],
        sympy_funcs=[_sp_f, _sp_rsr_sin, _sp_rsr_x],
        test_id="28_sinc_composite",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=20,
        preconditions={"f": pre_f},
        isolate_terms=["f(x+y)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )


def test_mod(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    _R = 101

    def f(x, R=_R):
        return x % R

    def _sp_f(x, R=_R):
        return sympy.Mod(x, R)

    evaluate(
        domain=Domain.Positive_Integer,
        distribution=Distribution(np.random.randint, low=1, high=10),
        exprs=["f(x+y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="29_mod",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=1,
        n=10,
        var_bound=10,
        constants={"R": _R},
        isolate_terms=["f(x+y)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )


def test_mod_mult(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    _R = 100001

    def f(x, y, R=_R):
        return (x * y) % R

    def _sp_f(x, y, R=_R):
        return (x * y) % R

    evaluate(
        domain=Domain.Positive_Integer,
        distribution=Distribution(np.random.randint, low=1, high=10),
        exprs=[
            "f(x1+x2, y1+y2)",
            "f(x1, y1)",
            "f(x2, y1)",
            "f(x1, y2)",
            "f(x2, y2)",
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="30_mod_mult",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=1,
        n=10,
        var_bound=10,
        constants={"R": _R},
        timeout_sec=timeout_sec,
    )


def test_int_mult(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x, y):
        return x * y

    def _sp_f(x, y):
        return x * y

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=1, high=10),
        exprs=[
            "f(x1+x2, y1+y2)",
            "f(x1, y1)",
            "f(x2, y1)",
            "f(x1, y2)",
            "f(x2, y2)",
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="31_int_mult",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=1,
        n=15,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_tanh(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return math.tanh(x)

    def _sp_f(x):
        return sympy.tanh(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="32_tanh",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=300,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_logistic(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
    L=1,
    k=2,
    x0=0,
):
    def f(x):
        return L / (1 + math.exp(-k * (x - x0)))

    def _sp_f(x):
        return L / (1 + sympy.exp(-k * (x - x0)))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="36_logistic",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=50,
        var_bound=20,
        constants={"L": L, "k": k, "x0": x0},
        isolate_terms=["f(x+y)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )


def test_logistic_scaled(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
    L=3,
    k=2,
    x0=0,
):
    def f(x):
        return L / (1 + math.exp(-k * (x - x0)))

    def _sp_f(x):
        return L / (1 + sympy.exp(-k * (x - x0)))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="37_logistic_scaled",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=30,
        # epsilon=0.2,
        # var_bound=20,
        constants={"L": L, "k": k, "x0": x0},
        isolate_terms=["f(x+y)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )


def test_square_loss(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return (1 - x) ** 2

    def _sp_f(x):
        return (1 - x) ** 2

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="38_square_loss",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=10,
        isolate_terms=["f(x+y)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )


def test_savage_loss_library(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 1 / (1 + math.exp(x)) ** 2

    def g(x, y):
        return math.exp(x) * math.exp(y)

    def _sp_f(x):
        return 1 / (1 + sympy.exp(x)) ** 2

    def _sp_g(x, y):
        return sympy.exp(x) * sympy.exp(y)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "g(x, y)", "f(x)", "f(y)"],
        infer_funcs=[f, g],
        sympy_funcs=[_sp_f, _sp_g],
        test_id="39_savage_loss_library",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=60,
        var_bound=20,
        isolate_terms=["f(x+y)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )

    # f(x+y)*g(x)*g(y)**2 + 2*f(x+y)*g(x)*g(y) + f(x+y) - 1 = 0
    # f(x+y) = \frac{1}{g(x) \cdot g(y)^2 + 2 \cdot g(x) \cdot g(y) + 1}


def test_savage_loss_basis(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 1 / (1 + math.exp(x)) ** 2

    def g(x):
        return math.exp(x)

    def _sp_f(x):
        return 1 / (1 + sympy.exp(x)) ** 2

    def _sp_g(x):
        return sympy.exp(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "g(x)", "g(y)"],
        infer_funcs=[f, g],
        sympy_funcs=[_sp_f, _sp_g],
        test_id="40_savage_loss_basis",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=5,
        n=100,
        var_bound=20,
        isolate_terms=["f(x+y)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )


def test_exp_div_by_log(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return (math.exp(x) - 1.0) / math.log(math.exp(x))

    def _sp_f(x):
        return (sympy.exp(x) - 1.0) / sympy.log(sympy.exp(x))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="xx_exp_div_by_log",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=150,
        timeout_sec=timeout_sec,
    )


def test_sin_over_sin(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    _a = math.radians(90)
    _c = 1

    def f(x, a=_a, c=_c):
        return math.sin(c * x) / math.sin(c * x + a)

    def _sp_f(x, a=_a, c=_c):
        return sympy.sin(_c * x) / sympy.sin(_c * x + _a)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="xx_sin_over_sin",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=300,
        var_bound=20,
        constants={"a": _a, "c": _c},
        timeout_sec=timeout_sec,
    )


def test_sinh_over_sinh(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    _a = math.radians(90)
    _c = 1

    def f(x, a=_a, c=_c):
        return math.sinh(c * x) / math.sinh(c * x + a)

    def _sp_f(x, a=_a, c=_c):
        return sympy.sinh(_c * x) / sympy.sinh(_c * x + _a)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="xx_sinh_over_sinh",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=150,
        var_bound=20,
        constants={"a": _a, "c": _c},
        timeout_sec=timeout_sec,
    )


def test_sigmoid(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 1 / (1 + math.exp(-x))

    def _sp_f(x):
        return 1 / (1 + sympy.exp(-x))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="33_sigmoid",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=30,
        # var_bound=20,
        isolate_terms=["f(x)", "f(x-y)"],
        timeout_sec=timeout_sec,
    )

    # eq = (
    #    "((f(x - y) * (f(y) - 1)) "
    #    "/ (2 * f(x - y) * f(y) - f(x - y) - f(y)))"
    #    "- f(x)"
    # )


def test_softmax2_1(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x, y):
        return math.exp(x) / (math.exp(x) + math.exp(y))

    def _sp_f(x, y):
        return sympy.exp(x) / (sympy.exp(x) + sympy.exp(y))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+r, y+r)", "f(x, y)", "f(x, r)", "f(y, r)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="34_softmax2_1",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_softmax2_2(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x, y):
        return math.exp(y) / (math.exp(x) + math.exp(y))

    def _sp_f(x, y):
        return sympy.exp(y) / (sympy.exp(x) + sympy.exp(y))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y, y+z)", "f(x,y)", "f(y,z)", "f(x,z)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="35_softmax2_2",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=15,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_sigmoid_4(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 1 / (1 + math.exp(-x))

    def _sp_f(x):
        return 1 / (1 + sympy.exp(-x))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=[
            "f(x+r+y+s)",
            "f(x)",
            "f(r)",
            "f(y)",
            "f(s)",
            "f(x+r)",
            "f(x+s)",
            "f(x+y)",
            "f(r+s)",
            "f(r+y)",
            "f(s+y)",
            "f(x-r)",
            "f(x-s)",
            "f(x-y)",
            "f(r-s)",
            "f(r-y)",
            "f(s-y)",
            "f(x+r+y)",
            "f(x+r+s)",
            "f(x+y+s)",
            "f(r+y+s)",
            "f(x+r-y)",
            "f(x+r-s)",
            "f(x+y-s)",
            "f(r+y-s)",
            "f(x-r-y)",
            "f(x-r-s)",
            "f(x-y-s)",
            "f(r-y-s)",
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="xx_sigmoid_4",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=30,
        # var_bound=20,
        isolate_terms=["f(x)"],
        timeout_sec=timeout_sec,
    )


def test_sigmoid_derivative(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 1 / (1 + math.exp(-x))

    def df(x):
        return f(x) * (1 - f(x))

    def _sp_f(x):
        return 2 / (1 + sympy.exp(-x))

    def _sp_df(x):
        return f(x) * (1 - f(x))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["df(x+y)", "df(x-y)", "df(x)", "df(y)"],
        infer_funcs=[df],
        sympy_funcs=[_sp_df],
        test_id="xx_sigmoid_derivative",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=5,
        n=100,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_gelu(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))

    def _sp_f(x):
        return 0.5 * x * (1 + sympy.erf(x / sympy.sqrt(2)))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="xx_gelu",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=5,
        n=50,
        # var_bound=20,
        isolate_terms=["f(x)"],
        timeout_sec=timeout_sec,
    )


def test_softmax2_alt_1(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x, y):
        return math.exp(x) / (math.exp(x) + math.exp(y))

    def _sp_f(x, y):
        return sympy.exp(x) / (sympy.exp(x) + sympy.exp(y))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=[
            "f(x1+x2, y1+y2)",
            "f(x1, y1)",
            "f(x2, y1)",
            "f(x1, y2)",
            "f(x2, y2)",
        ],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="xx_softmax2_alt_1",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=2,
        n=150,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_arctan(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return math.atan(x)

    def _sp_f(x):
        return sympy.atan(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-2, high=2),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="xx_arctan",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=200,
        var_bound=20,
        timeout_sec=timeout_sec,
    )


def test_savage_loss(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(v):
        return 1 / (1 + math.exp(v)) ** 2

    def _sp_f(x):
        return 1 / (1 + sympy.exp(x)) ** 2

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="xx_savage_loss",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=4,
        n=100,
        var_bound=10,
        timeout_sec=timeout_sec,
    )


def test_relu(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return max(0, x)

    def _sp_f(x):
        return sympy.Max(0, x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="xx_relu",
        res_dir=res_dir,
        method=method,
        milp=milp,
        max_degree=3,
        n=150,
        var_bound=10,
        timeout_sec=timeout_sec,
    )


def test_sin_glibc(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    iteration = 40
    sample_error = {}

    def f(x):
        return math.sin(x)

    def _sp_f(x):
        return sympy.sin(x)

    for i in range(5, iteration, 5):
        props, error, sample, _ = infer_property_with_timeout(
            domain=Domain.Real,
            distribution=Distribution(np.random.uniform, low=-5, high=5),
            exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
            template=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
            functions=[f],
            max_degree=2,
            n=i,
            epsilon=0.1,
        )

        for eq in props:
            verify_with_timeout(eq, [_sp_f])

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


def test_sigmoid_extra(
    res_dir: str,
    method: Method,
    milp: MILPSolver,
    timeout_sec: float,
):
    def f(x):
        return 1 / (1 + math.exp(-x))

    def _sp_f(x):
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
            props, error, sample, _ = infer_property_with_timeout(
                domain=Domain.Real,
                distribution=Distribution(np.random.uniform, low=-5, high=5),
                exprs=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
                template=["f(x+y)", "f(x-y)", "f(x)", "f(y)"],
                functions=[f],
                max_degree=3,
                n=i,
                milp=None,
                epsilon=0.2,
                method=Method.MULTIPLE_REGRESSION,
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
                            and verify_with_timeout(prop, [_sp_f])[0]
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
    df.to_csv("sigmoid_extra.csv")
    print(df)


def get_parser():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--res_dir",
        type=str,
        default="./results",
        help="the results directory",
    )

    parser.add_argument(
        "--method",
        type=Method,
        default=Method.MULTIPLE_REGRESSION,
        choices=list(Method),
        help="the fitting method to be used",
    )

    parser.add_argument(
        "--milp",
        type=MILPSolver,
        default=None,
        choices=list(MILPSolver),
        help="the MILP solver to be used",
    )

    parser.add_argument(
        "--timeout_sec",
        type=float,
        default=1800.0,
        help="the end-to-end timeout (sec) for each test case",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    res_dir = args.res_dir
    os.makedirs(res_dir, exist_ok=True)

    method = args.method
    milp = args.milp
    timeout_sec = args.timeout_sec

    test_args = (res_dir, method, milp, timeout_sec)

    st = time()

    test_identity(*test_args)  # 1
    test_exp(*test_args)  # 2
    test_exp_minus_one(*test_args)  # 3
    test_exp_div_by_x(*test_args)  # 4 no solution
    test_exp_div_by_x_composite(*test_args)  # 5
    test_floudas(*test_args)  # 6
    test_mean(*test_args)  # 7
    test_tan(*test_args)  # 8
    test_cot(*test_args)  # 9
    test_diff_squares(*test_args)  # 10
    test_inverse_square(*test_args)  # 11
    test_inverse(*test_args)  # 12
    test_inverse_add(*test_args)  # 13
    test_inverse_cot_plus_one(*test_args)  # 14
    test_inverse_tan_plus_one(*test_args)  # 15
    test_x_over_one_minus_x(*test_args)  # 16
    test_minus_x_over_one_minus_x(*test_args)  # 17
    test_cos(*test_args)  # 18
    test_cosh(*test_args)  # 19
    test_squared(*test_args)  # 20
    test_sin(*test_args)  # 21
    test_sinh(*test_args)  # 22
    test_cube(*test_args)  # 23
    test_log(*test_args)  # 24
    test_sec(*test_args)  # 25
    test_csc(*test_args)  # 26
    # https://en.wikipedia.org/wiki/Sinc_function
    test_sinc(*test_args)  # 27
    test_sinc_composite(*test_args)  # 28
    test_mod(*test_args)  # 29
    test_mod_mult(*test_args)  # 30
    test_int_mult(*test_args)  # 31

    # Activation Functions
    test_tanh(*test_args)  # 32
    test_sigmoid(*test_args)  # 33
    # https://en.wikipedia.org/wiki/Softmax_function
    test_softmax2_1(*test_args)  # 34
    test_softmax2_2(*test_args)  # 35

    # https://en.wikipedia.org/wiki/Logistic_function
    test_logistic(*test_args)  # 36
    test_logistic_scaled(*test_args)  # 37

    # Loss Functions
    # https://en.wikipedia.org/wiki/Loss_functions_for_classification
    test_square_loss(*test_args)  # 38
    test_savage_loss_library(*test_args)  # 39 library
    test_savage_loss_basis(*test_args)  # 40

    # Not checked
    # test_sigmoid_4(*test_args)
    # test_sigmoid_extra(*test_args)
    # test_sigmoid_derivative(*test_args)
    # test_sin_glibc(*test_args)
    # test_arctan(*test_args)
    # test_sin_over_sin(*test_args)
    # test_sinh_over_sinh(*test_args)
    # test_softmax2_alt_1(*test_args)
    # test_savage_loss(*test_args)
    # test_relu(*test_args)
    # test_gelu(*test_args)

    log.info(f"Total Time: {time() - st:.2f}s")
