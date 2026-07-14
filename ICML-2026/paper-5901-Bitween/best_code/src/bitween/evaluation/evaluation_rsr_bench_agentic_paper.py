import argparse
import math
import os
from contextlib import ExitStack
from time import time

import numpy as np
import sympy

from bitween.agent import (
    BaseAgent,
    create_infer_property_tool,
    create_symbolic_verify_tool,
)
from bitween.analyzer import verify_with_timeout
from bitween.bedrock_agent import BedrockAgent
from bitween.config import Config
from bitween.miscs import getLogger
from bitween.openai_agent import OpenAIAgent
from bitween.sampler import Distribution, Domain

config = Config()
config.logger_level = 5
log = getLogger(__name__, config.logger_level)


def evaluate(
    domain: Domain,
    distribution: Distribution,
    infer_funcs: list[callable],
    sympy_funcs: list[callable],
    test_id: str,
    agent: BaseAgent,
    res_dir: str,
    timeout_sec: float,
    preconditions: dict[str, callable] = None,
    constants: dict = None,
    custom_tools: list[str] = None,
    mcp_tools: list[str] = None,
):
    trace_file = os.path.join(res_dir, f"{test_id}_trace.csv")
    out_file = os.path.join(res_dir, f"{test_id}.txt")

    available_tools = {
        "infer_property_tool": create_infer_property_tool(
            domain=domain,
            distribution=distribution,
            functions=infer_funcs,
            trace_file=trace_file,
            preconditions=preconditions,
        ),
        "symbolic_verify_tool": create_symbolic_verify_tool(
            functions=sympy_funcs,
            domain=domain,
            constants=constants,
        ),
    }

    tools = []

    # filter custom tools
    custom_tool_names = []
    custom_tools = custom_tools or []
    for name in custom_tools:
        if name not in available_tools:
            log.warning(f"Provided unavailable custom tool: {name}")
        else:
            custom_tool_names.append(name)
            tools.append(available_tools[name])

    # filter mcp tools
    mcp_clients = []
    mcp_tools = mcp_tools or []
    for name in mcp_tools:
        if name not in agent.mcp_clients:
            log.warning(f"Provided unavailable mcp tool: {name}")
        else:
            mcp_clients.append(agent.mcp_clients[name])

    prompt = agent.create_prompt_from_functions(infer_funcs, custom_tool_names)
    log.info(f"Starting {test_id}")

    try:
        st = time()

        with ExitStack() as stack:
            for mcp_client in mcp_clients:
                stack.enter_context(mcp_client)
                mcp_tools = mcp_client.list_tools_sync()
                tools.extend(mcp_tools)

            response = agent.query(prompt, tools, timeout_sec)

        log.info(response.to_string(with_trace=False))

        categories = ["verified", "unverified", "faulty", "unknown"]
        equations = {key: [] for key in categories}

        for eq in response.answers:
            ok, error_msg = verify_with_timeout(eq, sympy_funcs, domain, constants)

            if error_msg:
                is_faulty = error_msg.startswith("Exception")
                key = "faulty" if is_faulty else "unverified"
            else:
                key = "verified" if ok else "unknown"

            equations[key].append((eq, error_msg))

        took_time = time() - st
        log.info(f"Took time: {took_time:.2f}s")

        def fmt_pair(pair):
            eq, msg = pair
            return f"{eq} | {msg}" if msg else f"{eq}"

        with open(out_file, "w") as fd:
            fd.write(f"{response.to_string()}\n\n")

            for key in categories:
                eqs = equations[key]
                eqs_len = len(eqs)
                if eqs_len > 0:
                    eqs_str = "\n".join(map(fmt_pair, eqs))
                    fd.write(f"{key.capitalize()} ({eqs_len}):\n{eqs_str}\n\n")

            fd.write(f"Took time: {took_time:.2f}s\n")

    except Exception as e:
        log.error(f"Exception found evaluating {test_id}", exc_info=e)

    finally:
        if os.path.exists(trace_file):
            os.remove(trace_file)

        log.info(f"Ending {test_id}")


def test_identity(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="01_identity",
        agent=agent,
        res_dir=res_dir,
        constants={"c": _c},
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_exp(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.exp(x)

    def _sp_f(x):
        return sympy.exp(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="02_exp",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_exp_minus_one(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.exp(x) - 1

    def _sp_f(x):
        return sympy.exp(x) - 1

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="03_exp_minus_one",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_exp_div_by_x(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.exp(x) / x

    def _sp_f(x):
        return sympy.exp(x) / x

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-2, high=2),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="04_exp_div_by_x",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_exp_div_by_x_composite(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f, h, p],
        sympy_funcs=[_sp_f, _sp_h, _sp_p],
        test_id="05_exp_div_by_x_composite",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_floudas(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="06_floudas",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_mean(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x, y, z):
        return 1 / 3 * (x + y + z)

    def _sp_f(x, y, z):
        return 1 / 3 * (x + y + z)

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="07_mean",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_tan(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.tan(x)

    def _sp_f(x):
        return sympy.tan(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="08_tan",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_cot(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return 1 / math.tan(x)

    def _sp_f(x):
        return 1 / sympy.tan(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="09_cot",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_diff_squares(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x, y):
        return x**2 - y**2

    def _sp_f(x, y):
        return x**2 - y**2

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="10_diff_squares",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_inverse_square(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return 1 / (x**2)

    def _sp_f(x):
        return 1 / (x**2)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="11_inverse_square",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_inverse(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return 1 / x

    def _sp_f(x):
        return 1 / x

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="12_inverse",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_inverse_add(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return 1 / (x + 1)

    def _sp_f(x):
        return 1 / (x + 1)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="13_inverse_add",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_inverse_cot_plus_one(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return 1 / (1 / math.tan(x) + 1)

    def _sp_f(x):
        return 1 / (1 / sympy.tan(x) + 1)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="14_inverse_cot_plus_one",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_inverse_tan_plus_one(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return 1 / (math.tan(x) + 1)

    def _sp_f(x):
        return 1 / (sympy.tan(x) + 1)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="15_inverse_tan_plus_one",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_x_over_one_minus_x(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return x / (1 - x)

    def _sp_f(x):
        return x / (1 - x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="16_x_over_one_minus_x",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_minus_x_over_one_minus_x(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return -x / (1 - x)

    def _sp_f(x):
        return -x / (1 - x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="17_minus_x_over_one_minus_x",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_cos(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.cos(x)

    def _sp_f(x):
        return sympy.cos(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="18_cos",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_cosh(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.cosh(x)

    def _sp_f(x):
        return sympy.cosh(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="19_cosh",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_squared(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return x**2

    def _sp_f(x):
        return x**2

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="20_squared",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_sin(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="21_sin",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_sinh(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.sinh(x)

    def _sp_f(x):
        return sympy.sinh(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="22_sinh",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_cube(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return x**3

    def _sp_f(x):
        return x**3

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="23_cube",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_log(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="24_log",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_sec(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return 1 / math.cos(x)

    def _sp_f(x):
        return 1 / sympy.cos(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="25_sec",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_csc(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return 1 / math.sin(x)

    def _sp_f(x):
        return 1 / sympy.sin(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="26_csc",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_sinc(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f, p],
        sympy_funcs=[_sp_f, _sp_p],
        test_id="27_sinc",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_sinc_composite(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f, rsr_sin, rsr_x],
        sympy_funcs=[_sp_f, _sp_rsr_sin, _sp_rsr_x],
        test_id="28_sinc_composite",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        preconditions={"f": pre_f},
        timeout_sec=timeout_sec,
    )


def test_mod(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="29_mod",
        agent=agent,
        res_dir=res_dir,
        constants={"R": _R},
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_mod_mult(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="30_mod_mult",
        agent=agent,
        res_dir=res_dir,
        constants={"R": _R},
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_int_mult(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x, y):
        return x * y

    def _sp_f(x, y):
        return x * y

    evaluate(
        domain=Domain.Integer,
        distribution=Distribution(np.random.randint, low=1, high=10),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="31_int_mult",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_tanh(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return math.tanh(x)

    def _sp_f(x):
        return sympy.tanh(x)

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="32_tanh",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_sigmoid(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return 1 / (1 + math.exp(-x))

    def _sp_f(x):
        return 1 / (1 + sympy.exp(-x))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="33_sigmoid",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_softmax2_1(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x, y):
        return math.exp(x) / (math.exp(x) + math.exp(y))

    def _sp_f(x, y):
        return sympy.exp(x) / (sympy.exp(x) + sympy.exp(y))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="34_softmax2_1",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_softmax2_2(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x, y):
        return math.exp(y) / (math.exp(x) + math.exp(y))

    def _sp_f(x, y):
        return sympy.exp(y) / (sympy.exp(x) + sympy.exp(y))

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="35_softmax2_2",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_logistic(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="36_logistic",
        agent=agent,
        res_dir=res_dir,
        constants={"L": L, "k": k, "x0": x0},
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_logistic_scaled(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="37_logistic_scaled",
        agent=agent,
        res_dir=res_dir,
        constants={"L": L, "k": k, "x0": x0},
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_square_loss(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
    timeout_sec: float,
):
    def f(x):
        return (1 - x) ** 2

    def _sp_f(x):
        return (1 - x) ** 2

    evaluate(
        domain=Domain.Real,
        distribution=Distribution(np.random.uniform, low=-5, high=5),
        infer_funcs=[f],
        sympy_funcs=[_sp_f],
        test_id="38_square_loss",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_savage_loss_library(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f, g],
        sympy_funcs=[_sp_f, _sp_g],
        test_id="39_savage_loss_library",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


def test_savage_loss_basis(
    agent: BaseAgent,
    res_dir: str,
    custom_tools: list[str],
    mcp_tools: list[str],
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
        infer_funcs=[f, g],
        sympy_funcs=[_sp_f, _sp_g],
        test_id="40_savage_loss_basis",
        agent=agent,
        res_dir=res_dir,
        custom_tools=custom_tools,
        mcp_tools=mcp_tools,
        timeout_sec=timeout_sec,
    )


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
        "--agent_type",
        required=True,
        choices=("openai", "bedrock"),
        help="the type of the agent",
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="the identifier of the model",
    )

    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="[openai|required] the base url of the OpenAI endpoint",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="[openai] the api key of the OpenAI endpoint "
        "obtained from the environment by default",
    )

    parser.add_argument(
        "--region_name",
        type=str,
        default=None,
        help="[bedrock|required] the region name of the Bedrock endpoint",
    )

    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="[bedrock] enable the thinking/reasoning mode of the model",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32_000,
        help="specify the maximum tokens of the model",
    )

    parser.add_argument(
        "--custom_tools",
        nargs="*",
        default=list(BaseAgent.custom_tool_prompts.keys()),
        help="specify which custom tools to use",
    )

    parser.add_argument(
        "--mcp_tools",
        nargs="*",
        default=list(BaseAgent.mcp_clients.keys()),
        help="specify which mcp tools to use",
    )

    parser.add_argument(
        "--timeout_sec",
        type=float,
        default=1800.0,
        help="the end-to-end timeout (sec) for each test case",
    )

    return parser


def get_agent_from_args(args):
    agent_type = args.agent_type

    if agent_type == "openai":
        if not args.api_key:
            args.api_key = os.getenv("OPENAI_API_KEY", None)
            if not args.api_key:
                log.critical("--api_key is required for the openai type")
                exit(1)

        if not args.base_url:
            log.critical("--base_url is required for the openai type")
            exit(1)

        agent = OpenAIAgent(
            model_id=args.model_id,
            base_url=args.base_url,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
        )

    elif agent_type == "bedrock":
        if not args.region_name:
            log.critical("--region_name is required for the bedrock type")
            exit(1)

        agent = BedrockAgent(
            model_id=args.model_id,
            region_name=args.region_name,
            enable_thinking=args.enable_thinking,
            max_tokens=args.max_tokens,
        )

    else:
        raise ValueError(f"Invalid agent type: {agent_type}")

    return agent


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

    test_identity(*test_args)  # 1
    test_exp(*test_args)  # 2
    test_exp_minus_one(*test_args)  # 3
    test_exp_div_by_x(*test_args)  # 4 - no solution
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
    test_sinc(*test_args)  # 27
    test_sinc_composite(*test_args)  # 28
    test_mod(*test_args)  # 29
    test_mod_mult(*test_args)  # 30
    test_int_mult(*test_args)  # 31

    # Activation functions
    test_tanh(*test_args)  # 32
    test_sigmoid(*test_args)  # 33
    test_softmax2_1(*test_args)  # 34
    test_softmax2_2(*test_args)  # 35
    test_logistic(*test_args)  # 36
    test_logistic_scaled(*test_args)  # 37

    # Loss function
    test_square_loss(*test_args)  # 38
    test_savage_loss_library(*test_args)  # 39
    test_savage_loss_basis(*test_args)  # 40

    log.info(f"Total Time: {time() - st:.2f}s")
