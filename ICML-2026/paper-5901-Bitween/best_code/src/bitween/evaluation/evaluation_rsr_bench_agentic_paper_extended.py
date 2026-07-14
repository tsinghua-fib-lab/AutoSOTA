import math
import os
from time import time

import numpy as np
import sympy

from bitween.agent import BaseAgent
from bitween.config import Config
from bitween.evaluation.evaluation_rsr_bench_agentic_paper import (
    evaluate,
    get_agent_from_args,
    get_parser,
)
from bitween.miscs import getLogger
from bitween.sampler import Distribution, Domain

config = Config()
config.logger_level = 5
log = getLogger(__name__, config.logger_level)


# Inverse trigonometric functions


def test_arcsin(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.asin(x)

    def _sp_f(x):
        return sympy.asin(x)

    def pre_f(x):
        return -1 <= x <= 1

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-1, high=1),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="41_arcsin",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_arccos(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.acos(x)

    def _sp_f(x):
        return sympy.acos(x)

    def pre_f(x):
        return -1 <= x <= 1

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-1, high=1),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="42_arccos",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_arctan(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.atan(x)

    def _sp_f(x):
        return sympy.atan(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="43_arctan",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# Inverse hyperbolic functions


def test_arcsinh(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.asinh(x)

    def _sp_f(x):
        return sympy.asinh(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="44_arcsinh",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_arccosh(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.acosh(x)

    def _sp_f(x):
        return sympy.acosh(x)

    def pre_f(x):
        return x > 1

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=1, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="45_arccosh",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_arctanh(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.atanh(x)

    def _sp_f(x):
        return sympy.atanh(x)

    def pre_f(x):
        return -1 < x < 1

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-1, high=1),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="46_arctanh",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


# ML activation functions


def test_relu(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return max(0, x)

    def _sp_f(x):
        return sympy.Max(0, x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="47_relu",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_leaky_relu(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    _alpha = 0.01

    def f(x, alpha=_alpha):
        return x if x > 0 else alpha * x

    def _sp_f(x, alpha=_alpha):
        return sympy.Piecewise((x, x > 0), (alpha * x, True))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="48_leaky_relu",
        agent=agent,
        res_dir=res_dir,
        constants={"alpha": _alpha},
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_swish(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return x / (1 + math.exp(-x))

    def _sp_f(x):
        return x / (1 + sympy.exp(-x))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="49_swish",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_gelu(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))

    def _sp_f(x):
        return 0.5 * x * (1 + sympy.erf(x / sympy.sqrt(2)))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-3, high=3),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="50_gelu",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# Logarithmic variants


def test_log1p(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.log1p(x)

    def _sp_f(x):
        return sympy.log(1 + x)

    def pre_f(x):
        return x > -1

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-1, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="51_log1p",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_logit(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.log(x / (1 - x))

    def _sp_f(x):
        return sympy.log(x / (1 - x))

    def pre_f(x):
        return 0 < x < 1

    evaluate(
        domain=Domain.Positive_Real,
        distribution=Distribution(np.random.uniform, low=0, high=1),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="52_logit",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_log2(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.log2(x)

    def _sp_f(x):
        return sympy.log(x, 2)

    def pre_f(x):
        return x > 0

    evaluate(
        domain=Domain.Positive_Real,
        distribution=Distribution(np.random.uniform, low=0, high=10),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="53_log2",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


# Power and root functions


def test_sqrt(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.sqrt(x)

    def _sp_f(x):
        return sympy.sqrt(x)

    def pre_f(x):
        return x >= 0

    evaluate(
        domain=Domain.Positive_Real,
        distribution=Distribution(np.random.uniform, low=0, high=10),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="54_sqrt",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_cbrt(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return x ** (1 / 3) if x >= 0 else -((-x) ** (1 / 3))

    def _sp_f(x):
        return sympy.cbrt(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-8, high=8),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="55_cbrt",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_x_to_x(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return x**x

    def _sp_f(x):
        return x**x

    def pre_f(x):
        return x > 0

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=0, high=4),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="56_x_to_x",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


# Number theory functions


def test_floor(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.floor(x)

    def _sp_f(x):
        return sympy.floor(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="57_floor",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_ceil(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.ceil(x)

    def _sp_f(x):
        return sympy.ceiling(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="58_ceil",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_frac(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return x - math.floor(x)

    def _sp_f(x):
        return x - sympy.floor(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="59_frac",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# Special functions


def test_erf(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.erf(x)

    def _sp_f(x):
        return sympy.erf(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-3, high=3),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="60_erf",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_gamma(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.gamma(x)

    def _sp_f(x):
        return sympy.gamma(x)

    def pre_f(x):
        return x > 0

    evaluate(
        domain=Domain.Positive_Real,
        distribution=Distribution(np.random.uniform, low=0, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="61_gamma",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


# Specialized compositions


def test_exp_sin(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.exp(math.sin(x))

    def _sp_f(x):
        return sympy.exp(sympy.sin(x))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="62_exp_sin",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_sin_exp(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.sin(math.exp(x))

    def _sp_f(x):
        return sympy.sin(sympy.exp(x))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-2, high=2),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="63_sin_exp",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_log_cos(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.log(abs(math.cos(x)))

    def _sp_f(x):
        return sympy.log(sympy.Abs(sympy.cos(x)))

    def pre_f(x):
        return math.cos(x) != 0

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-3, high=3),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="64_log_cos",
        agent=agent,
        res_dir=res_dir,
        preconditions={"f": pre_f},
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_sqrt_one_plus_x2(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.sqrt(1 + x**2)

    def _sp_f(x):
        return sympy.sqrt(1 + x**2)

    def pre_f(x):
        return 1 + x**2 > 0

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="65_sqrt_one_plus_x2",
        agent=agent,
        res_dir=res_dir,
        preconditions={"f": pre_f},
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_abs(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return abs(x)

    def _sp_f(x):
        return sympy.Abs(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="66_abs",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_sign(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return 1 if x > 0 else (-1 if x < 0 else 0)

    def _sp_f(x):
        return sympy.sign(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="67_sign",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_gudermannian(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.atan(math.sinh(x))

    def _sp_f(x):
        return sympy.atan(sympy.sinh(x))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-3, high=3),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="68_gudermannian",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_2_to_x(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return 2**x

    def _sp_f(x):
        return 2**x

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-3, high=3),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="69_2_to_x",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_10_to_x(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return 10**x

    def _sp_f(x):
        return 10**x

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-2, high=2),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="70_10_to_x",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# Rational functions


def test_pade_1_1(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """Padé approximant [1/1] for exp(x) around x=0"""

    def f(x):
        return (1 + x / 2) / (1 - x / 2)

    def _sp_f(x):
        return (1 + x / 2) / (1 - x / 2)

    def pre_f(x):
        return x != 2

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-2, high=2),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="71_pade_1_1",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_pade_2_2(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """Padé approximant [2/2] for exp(x) around x=0"""

    def f(x):
        return (1 + x / 2 + x**2 / 12) / (1 - x / 2 + x**2 / 12)

    def _sp_f(x):
        return (1 + x / 2 + x**2 / 12) / (1 - x / 2 + x**2 / 12)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-2, high=2),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="72_pade_2_2",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_continued_fraction_golden(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """Continued fraction representation approximating golden ratio"""

    def f(x):
        return 1 + 1 / (1 + 1 / (1 + 1 / (1 + x)))

    def _sp_f(x):
        return 1 + 1 / (1 + 1 / (1 + 1 / (1 + x)))

    def pre_f(x):
        return x != -1

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-1, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="73_continued_fraction_golden",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_continued_fraction_tan(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """Continued fraction representation for tan(x)"""

    def f(x):
        return x / (1 - x**2 / (3 - x**2 / 5))

    def _sp_f(x):
        return x / (1 - x**2 / (3 - x**2 / 5))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-1, high=1),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="74_continued_fraction_tan",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_mobius_simple(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """Simple Möbius transformation"""

    _a = 2
    _b = 1
    _c = 1
    _d = 3

    def f(x, a=_a, b=_b, c=_c, d=_d):
        return (a * x + b) / (c * x + d)

    def _sp_f(x, a=_a, b=_b, c=_c, d=_d):
        return (a * x + b) / (c * x + d)

    def pre_f(x):
        return x != -(_d / _c) if _c != 0 else True

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="75_mobius_simple",
        agent=agent,
        res_dir=res_dir,
        constants={"a": _a, "b": _b, "c": _c, "d": _d},
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_mobius_inversion(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """Möbius transformation representing inversion"""

    def f(x):
        return 1 / x

    def _sp_f(x):
        return 1 / x

    def pre_f(x):
        return x != 0

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="76_mobius_inversion",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_mobius_cayley(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """Cayley transform - maps upper half-plane to unit disk"""

    def f(x):
        return (x - 1) / (x + 1)

    def _sp_f(x):
        return (x - 1) / (x + 1)

    def pre_f(x):
        return x != -1

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-1, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="77_mobius_cayley",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


# Additional exponential variants


def test_exp_x2(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """exp(x^2) - Gaussian-related function"""

    def f(x):
        return math.exp(x**2)

    def _sp_f(x):
        return sympy.exp(x**2)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-1.5, high=1.5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="78_exp_x2",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_exp_cos(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    """exp(cos(x)) - Exponential of cosine composition"""

    def f(x):
        return math.exp(math.cos(x))

    def _sp_f(x):
        return sympy.exp(sympy.cos(x))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="79_exp_cos",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


# Polynomials


def test_fourth(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return x**4

    def _sp_f(x):
        return x**4

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="80_fourth",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    agent = get_agent_from_args(args)

    res_dir = args.res_dir
    os.makedirs(res_dir, exist_ok=True)

    custom_tools = args.custom_tools
    mcp_tools = args.mcp_tools
    timeout_sec = args.timeout_sec

    test_args = (agent, res_dir, custom_tools, mcp_tools, timeout_sec)

    st = time()

    # Inverse trigonometric functions
    test_arcsin(*test_args)  # 41
    test_arccos(*test_args)  # 42
    test_arctan(*test_args)  # 43

    # Inverse hyperbolic functions
    test_arcsinh(*test_args)  # 44
    test_arccosh(*test_args)  # 45
    test_arctanh(*test_args)  # 46

    # ML activation functions
    test_relu(*test_args)  # 47
    test_leaky_relu(*test_args)  # 48
    test_swish(*test_args)  # 49
    test_gelu(*test_args)  # 50

    # Logarithmic variants
    test_log1p(*test_args)  # 51
    test_logit(*test_args)  # 52
    test_log2(*test_args)  # 53

    # Power and root functions
    test_sqrt(*test_args)  # 54
    test_cbrt(*test_args)  # 55
    test_x_to_x(*test_args)  # 56

    # Number theory functions
    test_floor(*test_args)  # 57
    test_ceil(*test_args)  # 58
    test_frac(*test_args)  # 59

    # Special functions
    test_erf(*test_args)  # 60
    test_gamma(*test_args)  # 61

    # Specialized compositions
    test_exp_sin(*test_args)  # 62
    test_sin_exp(*test_args)  # 63
    test_log_cos(*test_args)  # 64
    test_sqrt_one_plus_x2(*test_args)  # 65
    test_abs(*test_args)  # 66
    test_sign(*test_args)  # 67
    test_gudermannian(*test_args)  # 68
    test_2_to_x(*test_args)  # 69
    test_10_to_x(*test_args)  # 70

    # Rational functions
    test_pade_1_1(*test_args)  # 71
    test_pade_2_2(*test_args)  # 72
    test_continued_fraction_golden(*test_args)  # 73
    test_continued_fraction_tan(*test_args)  # 74
    test_mobius_simple(*test_args)  # 75
    test_mobius_inversion(*test_args)  # 76
    test_mobius_cayley(*test_args)  # 77

    # Additional exponential variants
    test_exp_x2(*test_args)  # 78
    test_exp_cos(*test_args)  # 79

    # Some Polynomials
    test_fourth(*test_args)  # 80

    log.info(f"Total Time: {time() - st:.2f}s")
