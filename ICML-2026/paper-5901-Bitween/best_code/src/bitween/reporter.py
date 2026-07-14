from inspect import signature

import numpy as np
from sympy import (
    Function,
    cos,
    cosh,
    exp,
    latex,
    sin,
    sinh,
    symbols,
    sympify,
    tan,
    tanh,
)

from bitween import milp
from bitween.analyzer import latex_func, str_func
from bitween.sampler import Distribution, Domain
from bitween.synthesizer import MILP, GeneratorMethod, generate


def print_methods(
    methods: list[GeneratorMethod], functions: list[callable]
):  # noqa U100
    """
    Print the methods and their properties

    :param methods: list of methods
    :param functions: list of functions
    """
    for method in methods:
        print(f"Method: {type(method).__name__} {method.version}")
        print(f"Domain: {method.domain}")
        print(f"Distribution: {method.distribution}")
        print(f"Basis: {method.template}")
        print(f"Max Degree: {method.max_degree}")
        print(f"Number of samples: {method.n}")
        print(f"Time spent: {round(method.time_spent, 2)}s")
        print(f"Synthesis calls: {method.synthesize_calls}")
        properties = method.get_distinct_properties()
        if len(properties) != len(method.properties):
            print("WARNING: Duplicate properties")
        for property in properties:
            print("Functions:")
            for f in functions:
                print(f"{str_func(f)}")
            print("Properties:")
            print(f"{property} = 0")


def table_section(method) -> str:
    """
    Generate the table of properties

    :param method: the method
    """
    var_list = []
    basis = "$F\\left("
    for func in method.template:
        f = sympify(func)
        args = f.free_symbols
        for arg in args:
            if latex(arg) not in var_list:
                var_list.append(latex(arg))
        basis += f"{latex(f)},"
    basis = basis[:-1] + "\\right) = 0$"
    var_list.sort()
    markdown = f"""\n# Method: {type(method).__name__} {method.version}
- {basis}
- ${', '.join(var_list)} \\in_{{\\mathcal{{U}}}} {method.domain.latex()}$
- ${method.distribution.latex()}$
- $d: {method.max_degree}$
- $n: {method.n}$
- $\\epsilon: {method.epsilon}$

---

"""
    return markdown


def table_entry(method: GeneratorMethod, functions: list[callable], domain: Domain):
    """
    Generate the table of properties

    :param method: the method
    :param functions: list of functions
    :param domain: domain of the variables
    """
    # Generate the table of properties
    entry = ""
    funcs = []
    var_set = set()
    for f in functions:
        sig = signature(f)
        args = [param.name for param in sig.parameters.values()]
        syms = [symbols(arg) for arg in args]
        var_set.update(syms)
        funcs.append(f"$\\displaystyle{{{latex_func(f)}}}$")
    var_list = [latex(var) for var in var_set]
    var_list.sort()
    for property in method.properties:
        verified = "$\\checkmark$" if property.verified else "$?$"
        entry += f"| ${', '.join(var_list)} \\in {domain.latex()}$ | {', '.join(funcs)} | ${latex(sympify(property.expr))} = 0$ | {verified} |\n"

    return f"{entry}"


if __name__ == "__main__":  # noqa E123
    """
    Evaluation Study (reproduce the results in the paper)
    """

    # dictionary of methods and their properties
    # key is a report's __repr__ string
    reports = {}

    x = symbols("x")
    f = Function("f")

    # get current time
    import time

    start_time = time.time()
    total_number_of_properties = 0

    # ------------------------------------------
    def f(x):
        return sin(x)

    def P(x, terms=40):
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

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["P(x+y)", "P(x-y)", "P(x)", "P(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f, P],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------
    def f(x):
        return cos(x)

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return tan(x)

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=3,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def cot(x):
        return 1 / tan(x)

    def f(x):
        return 1 / (1 + cot(x))

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=3,
        n=30,
        epsilon=5e-1,
    )

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    total_number_of_properties += len(methods[0].get_distinct_properties())

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return 1 / x

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return x / (1 - x)

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=3,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))
    # ------------------------------------------

    def f(x):
        return -(x / (1 - x))

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=3,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))
    # ------------------------------------------

    def f(x):
        return x

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return 1 / x * x * 5

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x-y)", "f(x)", "1"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "1"],  # function basis aka the template
        [f],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return exp(x)

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return exp(x) - 1

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return cosh(x)

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------
    def f(x):
        return (exp(x) + exp(-x)) / 2

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return x * x

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return tanh(x)

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=1, obj=1e-10)],  # methods
        ["f(x+y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=3,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return (exp(2 * x) - 1) / (exp(2 * x) + 1)

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=1, obj=1e-10)],  # methods
        ["f(x+y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=3,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------
    def f(x):
        return sinh(x)

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------
    def f(x):
        return (exp(x) - exp(-x)) / 2

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return x * x + x + 2

    def g(x):
        # g(x+y) + g(x-y) = 2*g(x) + 2*g(y)
        # g(x) = 1/2 * g(x+y) + 1/2 * g(x-y) - g(y)
        return x * x

    def h(x):
        # h(x+y) + h(x-y) = 2*h(x)
        # h(x) = 1/2 * h(x+y) + 1/2 * h(x-y))
        return x + 2

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x + y)", "f(x - y)", "f(x)", "f(y)", "g(x)", "h(x)", "1"],
        [
            "f(x+y)",
            "f(x-y)",
            "f(x)",
            "f(y)",
            "(1/2*g(x+y) + 1/2*g(x-y) - g(y))",
            "(1/2*h(x+y) + 1/2*h(x-y))",
            "1",
        ],
        [f, g, h],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f, g, h])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f, g, h]))

    # ------------------------------------------

    def f(x):
        return x**2 + sin(x)

    def g(x):
        # g(x+y) + g(x-y) = 2*g(x) + 2*g(y)
        # g(x) = 1/2 * g(x+y) + 1/2 * g(x-y) - g(y)
        return x**2

    def h(x):
        # h(x) = h(y)*h(y)/h(x) + h(x-y)*h(x+y)/h(x)
        return sin(x)

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-10)],  # methods
        ["f(x)", "g(x)", "h(x)", "1"],
        [
            "f(x)",
            "(1/2*g(x+y) + 1/2*g(x-y) - g(y))",
            "(1/h(x)*(h(y)*h(y)) + 1/h(x)*(h(x-y)*h(x+y)))",
            "1",
        ],
        [f, g, h],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f, g, h])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f, g, h]))

    # ------------------------------------------

    def f(x):
        return sin(x)

    def g(x):
        return cos(x)

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=1, obj=1e-10)],  # methods
        # fmt: off
        [
            "f(x)*f(x)", "g(x)*g(x)", "f(y)*f(y)", "g(y)*f(y)", "f(x+y)", "g(x+y)",
            "f(x-y)", "g(x-y)", "f(x)", "f(y)", "g(x)", "g(y)", "1",
        ],
        [
            "f(x)*f(x)", "g(x)*g(x)", "f(y)*f(y)", "g(y)*g(y)", "f(x+y)", "g(x+y)",
            "f(x-y)", "g(x-y)", "f(x)", "f(y)", "g(x)", "g(y)", "1",
        ],
        # fmt: on
        [f, g],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f, g])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f, g]))

    # ------------------------------------------

    def f(x, y):
        return x * y

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=1, obj=1e-10)],  # methods
        ["f(x+r,y+s)", "f(x,y)", "f(x,s)", "f(r,y)", "f(r,s)", "1"],
        ["f(x+r,y+s)", "f(x,y)", "f(x,s)", "f(r,y)", "f(r,s)", "1"],
        [f],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x, y):
        return x + y

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=1, obj=1e-10)],  # methods
        ["f(x+r,y+s)", "f(x,y)", "f(x,s)", "f(r,y)", "f(r,s)", "1"],
        ["f(x+r,y+s)", "f(x,y)", "f(x,s)", "f(r,y)", "f(r,s)", "1"],
        [f],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x, y):
        return x + y

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=1, obj=1e-10)],  # methods
        ["f(x, y)", "f(x, f(x, y))", "f(f(x, y), y)", "f(f(x, y), f(x, y))", "1"],
        ["f(x, y)", "f(x, f(x, y))", "f(f(x, y), y)", "f(f(x, y), f(x, y))", "1"],
        [f],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x, y):
        return x + y

    methods = generate(
        Domain.Integer,
        Distribution(np.random.randint, low=1, high=10),
        [MILP(k=1, timeout=3, max_bound=1, obj=1e-10)],  # methods
        # fmt: off
        ["f(x, y)", "f(x, f(x, y))", "f(f(x, y), y)", "f(f(x, y), f(x, y))", "f(f(x, y), z)", "f(x, f(y, z))", "1", ],
        ["f(x, y)", "f(x, f(x, y))", "f(f(x, y), y)", "f(f(x, y), f(x, y))", "f(f(x, y), z)", "f(x, f(y, z))", "1", ],
        # fmt: on
        [f],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x, y):
        return x * y + x + y

    methods = generate(
        Domain.Integer,
        Distribution(np.random.randint, low=1, high=10),
        [MILP(k=1, timeout=3, max_bound=1, obj=1e-10)],  # methods
        ["f(x,y)", "f(x+r,y+s)", "f(x+r,s)", "f(r,y+s)", "f(r,s)", "1"],
        ["f(x,y)", "f(x+r,y+s)", "f(x+r,s)", "f(r,y+s)", "f(r,s)", "1"],
        [f],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x, y):
        return x * y + x

    methods = generate(
        Domain.Integer,
        Distribution(np.random.randint, low=1, high=10),
        [MILP(k=1, timeout=3, max_bound=1, obj=1e-10)],  # methods
        ["f(x,y)", "f(x-r,y-s)", "f(x-r,s)", "f(r,y-s)", "f(r,s)", "1"],
        ["f(x,y)", "f(x-r,y-s)", "f(x-r,s)", "f(r,y-s)", "f(r,s)", "1"],
        [f],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x, y):
        return x * y + x

    methods = generate(
        Domain.Integer,
        Distribution(np.random.randint, low=1, high=10),
        [MILP(k=1, timeout=3, max_bound=1, obj=1e-10)],  # methods
        ["f(x,y)", "f(x+r,y+s)", "f(x+r,s)", "f(r,y+s)", "f(r,s)", "1"],
        ["f(x,y)", "f(x+r,y+s)", "f(x+r,s)", "f(r,y+s)", "f(r,s)", "1"],
        [f],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return x * x - 5 * x + 6

    methods = generate(
        Domain.Integer,
        Distribution(np.random.randint, low=1, high=10),
        [MILP(k=1, timeout=3, max_bound=16, obj=1e-10)],  # methods
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],
        [f],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return x

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=1, obj=1e-12)],  # methods
        ["f(x)", "f(x-x1)", "f(x-x2)", "f(x-x3)", "f(x1)", "f(x2)", "f(x3)", "1"],
        ["f(x)", "f(x-x1)", "f(x-x2)", "f(x-x3)", "f(x1)", "f(x2)", "f(x3)", "1"],
        [f],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return x**2

    methods = generate(
        Domain.Integer,
        Distribution(np.random.randint, low=1, high=10),
        [MILP(k=1, timeout=3, max_bound=1, obj=1e-12)],  # methods
        # fmt: off
        ["f(x1+x2+x3)", "f(x1+x2)", "f(x2+x3)", "f(x1+x3)", "f(x1)", "f(x2)", "f(x3)", "1"],
        ["f(x1+x2+x3)", "f(x1+x2)", "f(x2+x3)", "f(x1+x3)", "f(x1)", "f(x2)", "f(x3)", "1"],
        # fmt: on
        [f],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x):
        return x

    methods = generate(
        Domain.Integer,
        Distribution(np.random.randint, low=1, high=10),
        [MILP(k=1, timeout=3, max_bound=1, obj=1e-12)],  # methods
        # fmt: off
        ["f(x1+x2+x3)", "f(x1+x2)", "f(x2+x3)", "f(x1+x3)", "f(x1)", "f(x2)", "f(x3)", "1"],
        ["f(x1+x2+x3)", "f(x1+x2)", "f(x2+x3)", "f(x1+x3)", "f(x1)", "f(x2)", "f(x3)", "1"],
        # fmt: on
        [f],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))

    # ------------------------------------------

    def f(x, y, z):
        return 1 / 3 * (x + y + z)

    methods = generate(
        Domain.Integer,
        Distribution(np.random.randint, low=1, high=10),
        [MILP(k=1, timeout=2, max_bound=1, obj=1e-14)],  # methods
        # fmt: off
        ["f(x1+x2+x3, y1+y2+y3, z1+z2+z3)", "f(x1+x2,y1+y2,z1+z2)", "f(x2+x3,y2+y3,z2+z3)", "f(x1+x3,y1+y3,z1+z3)", "f(x1,y1,z1)", "f(x2,y2,z2)", "f(x3,y3,z3)", "1"],
        ["f(x1+x2+x3, y1+y2+y3, z1+z2+z3)", "f(x1+x2,y1+y2,z1+z2)", "f(x2+x3,y2+y3,z2+z3)", "f(x1+x3,y1+y3,z1+z3)", "f(x1,y1,z1)", "f(x2,y2,z2)", "f(x3,y3,z3)", "1"],
        # fmt: on
        [f],
        max_degree=1,
        n=30,
        epsilon=5e-2,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [f])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [f]))
    # ------------------------------------------

    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n - 1)

    def S(x, terms=60):
        # NOTE: when terms=20 or 40, the prop does not meet the obj. func
        """
        sigmoid function
        ----------------
        Taylor approximation of sigmoid function
        `S(x) = 1/2 + x/4 - x^3/48 + x^5/4800 - ...`
        """
        exp_approx = 1 + sum(
            [((-1) ** n) * (x**n) / factorial(n) for n in range(1, terms)]
        )
        return 1 / (1 + 1 / exp_approx)

    def s(x):
        """
        sigmoid function
        """
        return 1 / (1 + exp(-x))

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=2, obj=1e-12)],  # methods
        ["S(x+y)", "S(x-y)", "S(x)", "S(y)", "1"],  # how to call functions
        ["s(x+y)", "s(x-y)", "s(x)", "s(y)", "1"],  # function basis aka the template
        [s, S],
        max_degree=2,
        n=30,
        epsilon=5e-1,
    )

    total_number_of_properties += len(methods[0].get_distinct_properties())

    print("------------------------------------------")
    print_methods(methods, [s])
    print("------------------------------------------")

    for method in methods:
        reports.setdefault(method.__repr__(), []).append((method, [s]))

    # ------------------------------------------

    # write markdown file
    with open(
        f"reports/univariate_{milp.__version__}.md", mode="w", newline="\n"
    ) as file:
        # for each key generate a section, and write the table
        for key in reports:
            file.write(table_section(reports[key][0][0]))
            # write the table header
            file.write(
                "Domain | Function | Property | Verified |\n| --- | --- | ------------------ | --- |\n"
            )
            for method, functions in reports[key]:
                file.write(table_entry(method, functions, domain=Domain.Real))

    elapsed_time = time.time() - start_time
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print(f"total_number_of_properties:{total_number_of_properties}")
