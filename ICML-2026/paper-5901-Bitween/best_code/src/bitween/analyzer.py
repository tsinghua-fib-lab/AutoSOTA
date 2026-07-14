from inspect import signature
from time import time

import numpy as np
import sympy

from bitween.config import Config
from bitween.miscs import Symbolic, TimeoutFunction, getLogger
from bitween.sampler import Distribution, Domain, sample

config = Config()
log = getLogger(__name__, config.logger_level)


def validate_preconditions(
    exprs: list[str],
    preconditions: dict[str, callable],
    variables: dict[str, float],
) -> tuple[bool, str]:
    if not preconditions:
        return True, ""

    for expr in exprs:
        sympy_expr = sympy.sympify(expr)
        call_dct = Symbolic.substitute_function_calls(sympy_expr, variables)

        for func_name, tuple_list in call_dct.items():
            for args_str, args in tuple_list:
                pre_func = preconditions.get(func_name, None)
                if pre_func is not None and not pre_func(*args):
                    error_msg = (
                        f"Precondition violated: "
                        f"{func_name}{args_str} used as "
                        f"{func_name}{str(args).replace(',)', ')')}"
                    )
                    return False, error_msg

    return True, ""


def latex_func(f: sympy.Function) -> str:
    sig = signature(f)
    args = [param.name for param in sig.parameters.values()]
    syms = [sympy.symbols(arg) for arg in args]

    return (
        f"{f.__name__}({', '.join(args)}) = "
        f"{sympy.latex(sympy.nsimplify(f(*syms), rational=True))}"
    )


def str_func(f: sympy.Function) -> str:
    sig = signature(f)
    args = [param.name for param in sig.parameters.values()]
    syms = [sympy.symbols(arg) for arg in args]

    return (
        f"{f.__name__}({', '.join(args)}) = "
        f"{str(sympy.nsimplify(f(*syms), rational=True))}"
    )


def verify(
    expr: sympy.Expr | sympy.Eq | str,
    functions: list[callable],
    domain: Domain = Domain.Real,
    constants: dict = None,
) -> tuple[bool, str]:
    """
    Proof by simplification

    Parameters
    ----------
    :param expr: the expression to be evaluated
    :param functions: the implementation of functions to be verified.
                They should be symbolic expressions that only uses
                `x`, `y`, `z`, `r`, `s`, `t` as variables.
    :param domain: the domain of variables
    :param constants: dictionary of name, values of constant variables

    :returns
    """

    log.debug(f"verifying: {expr} in domain: {domain}")

    st = time()

    functions = {fun.__name__.removeprefix("_sp_"): fun for fun in functions}

    domain_to_kwargs = {
        Domain.Real: {"real": True},
        Domain.Positive_Real: {"positive": True, "real": True},
        Domain.Integer: {"integer": True},
        Domain.Positive_Integer: {"positive": True, "integer": True},
    }
    symbol_kwargs = domain_to_kwargs[domain]

    constants = constants or {}

    try:
        expr = sympy.sympify(str(expr))

        if isinstance(expr, sympy.Eq):
            expr = expr.lhs

        variables = {
            str(var): sympy.Symbol(str(var), **symbol_kwargs)
            for var in expr.free_symbols
        }

    except Exception as e:
        msg = f"Exception during parsing of {expr}: {e}"
        log.warning(msg, exc_info=e)
        return False, msg

    # if the size of the expression is too large, sympy will take
    # a long time to simplify it
    if len(expr.args) > 20:
        msg = (
            f"Skipping verification of {expr} due to "
            "argument size being greater than 20"
        )
        log.info(msg)
        return False, msg

    try:
        lhs_value = eval(
            f"simplify({expr})",
            sympy.__dict__,
            functions | variables | constants,
        )

        proved = lhs_value == 0

        if proved:
            msg = ""
            log.debug(f"proved: {proved} \u2713")
        else:
            msg = f"Instead of zero, expr simplified to: {lhs_value}"
            log.debug(f"proved: {proved} | {msg}")

        return proved, msg

    except Exception as e:
        msg = f"Exception during verification of {expr}: {e}"
        log.error(msg, exc_info=e)
        return False, msg

    finally:
        log.debug(f"Verification Time: {time() - st:.2f}s")


def verify_with_timeout(
    expr: sympy.Expr | sympy.Eq | str,
    functions: list[callable],
    domain: Domain = Domain.Real,
    constants: dict = None,
    timeout_sec: float = 5.0,
) -> tuple[bool, str]:
    res, error_msg = TimeoutFunction.call_for(
        timeout_sec=timeout_sec,
        func=verify,
        args=(
            expr,
            functions,
            domain,
            constants,
        ),
    )

    if error_msg:
        res = (False, error_msg)

    return res


def property_test(  # noqa E123
    expr: sympy.Expr | sympy.Eq | str,
    functions: list[callable],
    distribution: Distribution = Distribution(np.random.uniform, low=-5, high=5),
    constants: dict = None,
    preconditions: dict[str, callable] = None,
    n: int = 30,
    max_attempts: int = None,
    epsilon: float = 1e-10,
) -> tuple[bool, str]:
    """
    Dynamic checking

    :param expr: the expression to be evaluated
    :param functions: the implementation of functions to be evaluated.
                It should be a list of either symbolic expressions
                or normal python functions.
    :param distribution: the distribution of the domain
    :param epsilon: the error tolerance
    :param n: the number of samples to be generated

    :return: True if the property is satisfied less than absolute epsilon
             in average, False otherwise

    :notes: If a function starting with a capital letter is passed in,
            we lower case it in order to match the variable name in
            the expression.
    """

    log.debug(f"property testing: {expr} with {n} samples and eps: {epsilon}")

    st = time()

    functions = {
        func.__name__.lower().removeprefix("_sp_"): (
            sympy.lambdify(func) if isinstance(func, sympy.Expr) else func
        )
        for func in functions
    }

    if not isinstance(expr, sympy.Expr):
        expr = sympy.sympify(expr)

    variables = [str(var) for var in expr.free_symbols]
    constants = constants or {}

    errors = []

    curr_iter = 0
    total_iter = 0
    max_attempts = max_attempts or 10 * n

    while curr_iter < n:
        if total_iter == max_attempts:
            error_msg = (
                f"Generated only {curr_iter} samples instead of {n}, "
                f"despite trying {max_attempts} attempts"
            )
            log.error(error_msg)
            return False, error_msg

        total_iter += 1
        variables = sample(distribution, list(variables))
        ok, error_msg = validate_preconditions(
            [expr],
            preconditions,
            variables,
        )
        if not ok:
            return False, error_msg

        try:
            errors.append(
                eval(
                    str(expr),
                    sympy.__dict__,
                    functions | variables | constants,
                )
            )

        except ZeroDivisionError as e:
            log.error(f"ZeroDivisionError: {e}, {variables}")
            continue

        except ValueError as e:
            log.error(f"ValueError: {e}, {variables}")
            continue
        curr_iter += 1

    mean_error = np.mean(errors)
    log.debug(f"mean error: {mean_error}")
    log.debug(f"Property Test Time: {time() - st:.2f}s")

    return abs(mean_error) <= epsilon, ""


def property_test_with_timeout(
    expr: sympy.Expr | sympy.Eq | str,
    functions: list[callable],
    distribution: Distribution = Distribution(np.random.uniform, low=-5, high=5),
    constants: dict = None,
    preconditions: dict[str, callable] = None,
    n: int = 30,
    max_attempts: int = None,
    epsilon: float = 1e-10,
    timeout_sec: float = 5.0,
) -> tuple[bool, str]:
    res, error_msg = TimeoutFunction.call_for(
        timeout_sec=timeout_sec,
        func=property_test,
        args=(
            expr,
            functions,
            distribution,
            constants,
            preconditions,
            n,
            max_attempts,
            epsilon,
        ),
    )

    if error_msg:
        res = (False, error_msg)

    return res


if __name__ == "__main__":  # noqa E123
    """
    Test cases
    """

    x, y, z, r, s, t = sympy.symbols("x y z r s t")
    out_sep = "------------------------------------------"
    in_sep = "---------------------"

    print(out_sep)

    def f(x):
        return sympy.tan(x)

    expr = "- f(x+y) + f(y) + f(y)*f(x)*f(x+y) + f(x)"

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def g(x):
        return sympy.tan(x)

    def f(x):
        return 1 / (1 + sympy.tan(x))

    expr = (
        "f(x+y) + f(y) - 2*f(y)*f(x+y) - 2*f(x)*f(x+y) "
        "+ 2*f(y)*f(x)*f(x+y) + f(x) - 1"
    )

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    # g(x) = g(x+y) - g(y) - g(y)*g(x)*g(x+y)
    expr = "f(x) - 1/(1 + g(x+y) - g(y) - g(y)*g(x)*g(x+y))"
    assert verify(expr, [f, g])[0]
    print(in_sep)
    assert property_test(expr, [f, g])[0]

    print(out_sep)

    def f(x):
        return sympy.tan(x)

    expr = "- f(x+y) + f(y) + f(y)*f(x)*f(x+y) "
    assert not verify(expr, [f])[0]
    print(in_sep)
    assert not property_test(expr, [f])[0]

    print(out_sep)

    def f(x):
        return sympy.tanh(x)

    expr = "f(x+y) + f(x+y)*f(x)*f(y) - f(x) - f(y)"
    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x):
        return (sympy.exp(2 * x) - 1) / (sympy.exp(2 * x) + 1)

    expr = "f(x+y) + f(x+y)*f(x)*f(y) - f(x) - f(y)"
    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x):
        return sympy.cosh(x)

    expr = "f(x-y) - 2*f(y)*f(x) + f(x+y)"

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x):
        return (sympy.exp(x) + sympy.exp(-x)) / 2

    expr = "f(x-y) - 2*f(y)*f(x) + f(x+y)"
    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x):
        return x * x

    expr = (
        "f(x-y) - 2 * f(x) - 2 * f(y) - f(y)*f(x+y) - f(y)*f(x-y) + "
        "2 * f(y)*f(x) + 2 * f(y)*f(y) + f(x+y)"
    )

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    expr = (
        "f(x1) + f(x2) + f(x3) - f(x1 + x2) - f(x1 + x3) - "
        "f(x2 + x3) + f(x1 + x2 + x3)"
    )

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x):
        c = 1
        return sympy.exp(c * x) - 1

    expr = "f(x+y) - f(x) - f(y) - f(x)*f(y)"

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x, y):
        return x + y

    expr = "- f(x, f(y, z)) + f(f(x, y), z)"

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x):
        return sympy.cos(x)

    expr = "f(x-y) - 2*f(y)*f(x) + f(x+y)"

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x):
        return sympy.cos(x)

    expr = (
        "f(x-y) - 2*f(y)*f(x) - 2*f(y)*f(y) - 2*f(x)*f(x) "
        "+ 2*f(x-y)*f(x+y) + f(x+y) + 2*1 "
    )

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    # sympy cannot simplify if we use cot(x) directly here
    def f(x):
        return 1 / (1 + 1 / sympy.tan(x))

    expr = "f(x) + f(y) - f(x+y) - 2*f(x)*f(y) + 2*f(x)*f(y)*f(x+y)"

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x):
        return 1 / x

    def pre_f(x):
        return x != 0

    expr = "- 2*f(y)*f(x+y) + f(y)*f(x) - 2*f(x)*f(x+y) + f(x)*f(y) "

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f], preconditions={"f": pre_f})[0]

    print(out_sep)

    def f(x):
        return x * x + x + 1

    expr = (
        "f(x-y) - 2*f(x) + 2*f(y) + f(y)*f(x+y) + f(y)*f(x-y) "
        "- 2*f(y)*f(x) - f(y)*f(y) - f(x)*f(x+y) - f(x)*f(x-y) "
        "+ 3*f(x)*f(x) - f(x-y)*f(x+y) + f(x+y) - 1 "
    )

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(in_sep)

    def g(x):
        # g(x+y) + g(x-y) = 2*g(x) + 2*g(y)
        return x * x

    def h(x):
        # h(x+y) + h(x-y) = 2*h(x)
        return x + 1

    # 2f(x) = g(x+y) + g(x-y) - 2*g(y) + h(x+y) + h(x-y)
    expr = "2*f(x) - g(x+y) - g(x-y) + 2*g(y) - h(x+y) - h(x-y)"

    assert verify(expr, [f, g, h])[0]
    print(in_sep)
    assert property_test(expr, [f, g, h])[0]

    print(out_sep)

    def f(x):
        return sympy.sin(x)

    def g(x):
        return sympy.cos(x)

    expr = "g(x)*g(x) + f(x)*f(x) - 1"

    assert verify(expr, [f, g])[0]
    print(in_sep)
    assert property_test(expr, [f, g])[0]

    print(out_sep)

    def f(x, y):
        return x * y

    def pre_f(x, y):
        return x != y

    expr = "- f(x+r,y+s) + f(x,s) + f(r,y) + f(r,s) + f(x,y) "

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f], preconditions={"f": pre_f})[0]

    print(out_sep)

    def f(x, y):
        return x * y + x

    expr = "- f(r,y-s)*f(x-r,s) + f(r,s)*f(x-r,y-s) "

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x, y):
        return x * y + x + y

    expr = (
        "f(x+r,y+s) - f(x+r,s) - f(r,y+s) + f(r,s)*f(x+r,y+s) "
        "- f(r,y+s)*f(x+r,s) + f(r,s) "
    )

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x, y):
        return x + y

    expr = "- f(x,s) - f(r,y) + f(r,s) + f(x,y)"

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x):
        return sympy.sinh(x)

    expr = "- f(y)*f(y) - f(x-y)*f(x+y) + f(x)*f(x)"

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x):
        return (sympy.exp(x) - sympy.exp(-x)) / 2

    expr = "- f(y)*f(y) - f(x-y)*f(x+y) + f(x)*f(x)"

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x):
        return sympy.sin(x)

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

    expr = "- f(y)*f(y) - f(x-y)*f(x+y) + f(x)*f(x)"

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)

    def f(x0, c):
        return 1 + 2 * sympy.sin(c * x0) + sympy.cos(x0) + x0

    expr = (
        "c**2 - 2*x0**2 - x0 + f(x0, c) - 2*sin(c*x0) "
        "- cos(x0) - Abs(c**2) + 2*Abs(x0**2) - 1"
    )

    assert verify(expr, [f])[0]
    print(in_sep)
    assert property_test(expr, [f])[0]

    print(out_sep)
