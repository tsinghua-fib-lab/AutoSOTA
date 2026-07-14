import csv
import time
from math import log2

import numpy as np
from gurobipy import GRB
from sympy import Abs, Expr, Function, pi, sqrt, sympify

from bitween.analyzer import property_test, verify
from bitween.milp import milp_synthesis
from bitween.property import Property
from bitween.refiner import refine
from bitween.sampler import Distribution, Domain, sample
from bitween.terms import canonicalize, get_terms, get_values_terms


class GeneratorMethod(object):
    def __init__(self, version):
        """Base class for generator methods"""
        self._domain = Domain.Real
        self._distribution = Distribution(np.random.uniform, low=-5, high=5)
        self._max_degree = 2
        self._n = 30
        self._epsilon = 1e-13
        self._version = version
        self._properties: list[Property] = []
        self._time_spent = 0.0
        self._last_time = None
        self._template = ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"]

    @property
    def version(self):
        """Get the version of the generator method"""
        return self._version

    @property
    def properties(self) -> list[Property]:
        """Get the list of inferred properties"""
        return self._properties

    def has_property(self, property: Property | Expr):
        """Check whether an expression or a property with the given expression exists"""
        for p in self._properties:
            if p == property:
                return True
        return False

    def get_distinct_properties(self) -> set[Property]:
        """Get the distinct properties"""
        return refine(self._properties)

    def add_property(self, expr: Expr, verified: bool):
        """Add a new property with the given expression"""
        p = Property(expr, self.time, verified)
        self._properties.append(p)
        return p

    @property
    def time_spent(self) -> float:
        """Get the time spent in seconds"""
        return self._time_spent

    def start_timer(self):
        """Start the timer"""
        self._last_time = time.time()

    def end_timer(self):
        """End the timer and update the time spent"""
        if self._last_time is not None:
            self._time_spent += time.time() - self._last_time
            self._last_time = None

    def time(self) -> float:
        """Get the elapsed time"""
        if self._last_time is not None:
            return time.time() - self._last_time
        return 0.0

    @property
    def domain(self) -> Domain:
        """Get the domain"""
        return self._domain

    @domain.setter
    def domain(self, domain):
        """Set the domain"""
        self._domain = domain

    @property
    def distribution(self) -> Distribution:
        """Get the distribution"""
        return self._distribution

    @distribution.setter
    def distribution(self, distribution):
        """Set the distribution"""
        self._distribution = distribution

    @property
    def max_degree(self) -> int:
        """Get the maximum degree"""
        return self._max_degree

    @max_degree.setter
    def max_degree(self, max_degree):
        """Set the maximum degree"""
        self._max_degree = max_degree

    @property
    def n(self) -> int:
        """Get the number of samples"""
        return self._n

    @n.setter
    def n(self, n):
        """Set the number of samples"""
        self._n = n

    @property
    def epsilon(self) -> float:
        """Get the epsilon value"""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        """Set the epsilon value"""
        self._epsilon = epsilon

    @property
    def template(self) -> list[str]:
        """Get the template"""
        return self._template

    @template.setter
    def template(self, template):
        """Set the template"""
        self._template = template


class MILP(GeneratorMethod):
    def __init__(self, k=3, timeout=5, max_bound=4, obj=1e-12):
        """
        MILP generator method

        :param k: the number of MILP iteration for each pivot term
        :param timeout: the timeout for each MILP iteration
        :param max_bound: the maximum bound for each coefficient [-max_bound, max_bound]
        :param obj: the maximum objective threshold to be considered as a property
        """
        super().__init__("0.0.1")
        self._k = k
        self._timeout = timeout
        self._max_bound = max_bound
        self._objective = obj
        self._synthesize_calls = 0

    @property
    def synthesize_calls(self):
        """Get the number of times synthesize has been called"""
        return self._synthesize_calls

    @property
    def k(self):
        """Get the number of MILP iterations for each pivot term"""
        return self._k

    @property
    def timeout(self):
        """Get the timeout for each MILP iteration"""
        return self._timeout

    @property
    def max_bound(self):
        """Get the maximum bound for each coefficient [-max_bound, max_bound]"""
        return self._max_bound

    @property
    def objective(self):
        """Get the maximum objective threshold to be considered as a property"""
        return self._objective

    def synthesize(self, data, terms, pivot, blocked, bound, sr_vals, sr_func):
        """
        synthesize the properties
        """
        self._synthesize_calls += 1
        return milp_synthesis(
            data, terms, pivot, blocked, bound, self.timeout, 1, sr_vals, sr_func
        )

    def __repr__(self):
        return (
            f"MILP(k={self.k}, timeout={self._timeout}, max_bound={self._max_bound}, "
            f"obj={self._objective}, version={self.version}, domain={self.domain}, "
            f"distribution={self.distribution}, max_degree={self.max_degree}, n={self.n}, "
            f"epsilon={self.epsilon}, template={self.template})"
        )


class SR_MILP(MILP):
    """
    MILP-based symbolic regression
    """

    pass


class LR(GeneratorMethod):
    """
    Linear regression-based Random Self-Reducibile property generator
    """

    def __init__(self):
        pass


def generate(  # noqa C901
    domain: Domain,  # domain of the samples
    distribution: Distribution,  # distribution of the samples
    methods: list[GeneratorMethod],  # methods to generate properties
    calls: list[str],  # function calls to function basis
    template: list[str],  # template for function basis
    functions: list[callable],  # function implementations of calls
    max_degree: int = 1,  # maximum degree
    n: int = 30,  # number of samples
    zero_bias: bool = False,  # whether to add zero bias in the samples
    epsilon: float = 1e-13,  # epsilon for property test
    sr_func: str | None = None,  # symbolic regression function
    composition_functions: list[Function] = [],  # composition functions
    depth=1,
):
    """
    Generate properties for the given functions and parameters.

    :param domain: domain of the samples
    :param distribution: distribution of the samples
    :param methods: methods to generate properties
    :param calls: function calls to function basis
    :param template: template for function basis
    :param functions: functions to be used to generate samples
    :param max_degree: maximum degree
    :param n: number of samples
    :param epsilon: epsilon for property test
    :param composition_functions: composition functions
    :param depth: depth of composition
    """
    # initialize methods
    for method in methods:
        method.domain = domain
        method.distribution = distribution
        method.template = template
        method.max_degree = max_degree
        method.n = n
        method.epsilon = epsilon
    # list of properties inferred
    properties: list[Property] = []
    # get a function dict
    func_dict = {func.__name__: func for func in functions}
    # for each degree
    for degree in range(1, max_degree + 1):
        all_terms = get_terms(degree, template, "1", composition_functions, depth=depth)
        print(f"All terms: {all_terms}")
        # for each pivot term
        for pivot in all_terms:
            print("------------------------------------------")
            print("Degree: " + str(degree) + ", Pivot: " + str(pivot))
            # clear the file
            with open("data.csv", mode="w", newline="") as file:
                file.write("")  # Write an empty string to clear the file

            with open("data.csv", mode="a", newline="\n") as file:
                writer = csv.writer(file)

                var_list = []
                for call in calls:
                    f = sympify(call)
                    args = f.free_symbols
                    for arg in args:
                        if arg not in var_list:
                            var_list.append(arg)

                i = n
                numeric_error = False
                zero_rotation = 0  # NOTE: zero rotation heuristics for samples
                zero_division_error = False
                sr_vals = []  # NOTE: special case for symbolic regression
                while i > 0:
                    var_dict = sample(distribution, var_list)

                    if zero_bias and i <= len(var_list) + 1 and not zero_division_error:
                        if i == 1:
                            # NOTE: all zero heuristics for samples
                            for var in var_list:
                                var_dict[str(var)] = 0
                        else:
                            var_dict[str(var_list[zero_rotation])] = 0
                            zero_rotation += 1

                    values = []
                    for call in calls:
                        try:
                            values.append(
                                eval(
                                    call,
                                    func_dict | var_dict,
                                    {"sqrt": sqrt, "Abs": Abs, "pi": pi},
                                )
                            )
                        except ZeroDivisionError as e:
                            print(f"ZeroDivisionError: {e}, {var_dict}")
                            numeric_error = True
                            zero_division_error = True
                            break

                    if numeric_error:
                        numeric_error = False
                        continue

                    (values, terms) = get_values_terms(
                        degree,
                        values,
                        template,
                        "1",
                        composition_functions,
                        depth=depth,
                    )

                    # putting important term to the end
                    important = terms.index(pivot)
                    values = (
                        values[:important]
                        + values[important + 1 :]
                        + [values[important]]
                    )
                    terms = (
                        terms[:important] + terms[important + 1 :] + [terms[important]]
                    )

                    # removing constant
                    constant = terms.index("1")
                    values = values[:constant] + values[constant + 1 :]
                    terms = terms[:constant] + terms[constant + 1 :]

                    i -= 1

                    # NOTE: Symbolic Regression
                    if sr_func is not None:
                        sr_vals.append(
                            eval(
                                sr_func,
                                func_dict | var_dict,
                                {"sqrt": sqrt, "Abs": Abs},
                            )
                        )

                    writer.writerow(values)

                print(terms)

            # Load the CSV file into a numpy array
            data = np.genfromtxt("data.csv", delimiter=",")
            # print(data)

            def prop_test(expr) -> bool:
                # if there is a function starting with capital letter, find the function
                # whose name is the same but starting with lower case letter
                # and remove it from the function list
                # e.g. delete f(x) if F(x) <-> f(x)
                functions_ = list(functions)
                for func in functions:
                    if func.__name__[0].isupper():
                        for j, func_ in enumerate(functions_):
                            if func_.__name__ == func.__name__.lower():
                                functions_.pop(j)
                                break

                return property_test(
                    expr,
                    functions_,
                    distribution=distribution,
                    epsilon=epsilon,
                    n=n,
                )

            def prop_verify(property: Expr) -> bool:
                return verify(property, functions)

            for m in methods:
                m.start_timer()
                if isinstance(m, MILP):
                    bound = m.max_bound
                    while bound > 0:  # NOTE: bound heuristics for MILP
                        status, expr, obj, costs = m.synthesize(
                            data, terms.copy(), pivot, None, bound, sr_vals, sr_func
                        )
                        if status == GRB.OPTIMAL:
                            # check if the objective is small enough
                            if abs(obj) < m.objective:
                                # check if the epsilon is small enough
                                if sr_func is not None:
                                    expr = sympify(expr) - sympify(sr_func)
                                if prop_test(expr):
                                    expr = canonicalize(expr)
                                    if not m.has_property(expr):
                                        p = m.add_property(expr, prop_verify(expr))
                                        if p not in properties:
                                            properties.append(p)
                                        # k times block the complex term and synthesize again
                                        blocking_count = m.k
                                        # NOTE: term blocking heuristics for MILP
                                        while blocking_count > 0 and len(costs) > 0:
                                            blocked = costs.pop()
                                            if blocked == pivot:
                                                blocking_count -= 1
                                                continue  # break blocking evaluation
                                            if blocked == "1":
                                                break  # break blocking evaluation
                                            print("Blocking: " + str(blocked))
                                            status_, expr_, obj_, _ = m.synthesize(
                                                data,
                                                terms.copy(),  # this object is modified in synthesize
                                                pivot,
                                                blocked,
                                                bound,
                                                sr_vals,
                                                sr_func,
                                            )
                                            if status_ == GRB.OPTIMAL:
                                                if abs(obj_) < m.objective:
                                                    if sr_func is not None:
                                                        expr = sympify(expr_) - sympify(
                                                            sr_func
                                                        )
                                                    if prop_test(expr_):
                                                        expr = canonicalize(expr_)
                                                        if not m.has_property(expr):
                                                            p = m.add_property(
                                                                expr, prop_verify(expr)
                                                            )
                                                            if p not in properties:
                                                                properties.append(p)
                                                    else:
                                                        print(
                                                            "Test failed: " + str(expr)
                                                        )
                                                else:
                                                    print(
                                                        "Obj. too large: " + str(obj_)
                                                    )
                                            blocking_count -= 1
                                else:
                                    print("Test failed: " + str(expr))
                            else:
                                print("Obj. too large: " + str(obj))
                        # NOTE: optimize if found solution's coefficients are less than bound, exit
                        bound = round(log2(bound))
                m.end_timer()

            print("------------------------------------------")
    # all properties explored
    if len(properties) == 0:
        print("No properties found for the given parameters :(")
    else:
        print("Properties found:")
        print("------------------------------------------")
        for i, p in enumerate(properties):
            print(f"{p} = 0 ({p.verified})")
            if i < len(properties) - 1:
                print("---------------------")
    print("------------------------------------------")
    return methods


if __name__ == "__main__":  # noqa E123
    """
    Test cases
    """
    from sympy import sin  # noqa F401
    from sympy import (
        Abs,
        Function,
        cos,
        exp,
        log,
        sign,
        sqrt,
        symbols,
        sympify,
        tan,
    )

    from bitween.reporter import print_methods

    # ------------------------------------------

    x = symbols("x")
    f = Function("f")

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

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=2, timeout=5, max_bound=2, obj=1e-12)],  # methods
        ["F(x+y)", "F(x-y)", "F(x)", "F(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f, F],
        max_degree=2,
        n=30,
        epsilon=1e-13,
    )

    print("------------------------------------------")
    print_methods(methods, f)
    print("------------------------------------------")

    assert "f(x)**2 - f(y)**2 - f(x - y)*f(x + y)" in [
        str(prop) for method in methods for prop in method.properties
    ]

    # ------------------------------------------
    # exit()
    x, x0, c = symbols("x x0 c", real=True)
    f = Function("f")

    def sinP(x, terms=40):
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

    def F(x0, c):
        return 1 + 2 * sinP(c * x0) + cos(x0) + x0

    def f(x0, c):
        return 1 + 2 * sin(c * x0) + cos(x0) + x0**2

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=3, max_bound=10, obj=1e-12)],  # methods
        ["x0", "c", "1"],
        ["x0", "c", "1"],
        [f],
        max_degree=2,
        n=50,
        epsilon=5e-1,
        sr_func="f(x0, c)",
        # fmt: off
        composition_functions=[x, sin(x), cos(x), tan(x), exp(x), sqrt(x), log(x), sign(x), Abs(x)],
        # fmt: on
        depth=1,
    )

    print("------------------------------------------")
    print_methods(methods, f)
    print("------------------------------------------")
    exit()
    # def f(x):
    #     return x * (sign(x) + 1) / 2

    # methods = generate(
    #     Domain.Real,
    #     Distribution(np.random.uniform, low=-5, high=5),
    #     [MILP(k=4, timeout=3, max_bound=4, obj=1e-12)],  # methods
    #     ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # how to call functions
    #     ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # function basis aka the template
    #     [f],
    #     max_degree=1,
    #     n=30,
    #     epsilon=1e-13,
    # )

    # print("------------------------------------------")
    # print_methods(methods, f)
    # print("------------------------------------------")

    # ------------------------------------------

    def f(x):
        return sqrt(x)

    # methods = generate(
    #     Domain.Real,
    #     Distribution(np.random.uniform, low=-5, high=5),
    #     [MILP(k=4, timeout=3, max_bound=10, obj=1e-12)],  # methods
    #     ["f(x)", "f(x+1)", "f(x+2)", "f(x-1)", "f(x-2)", "x", "1"],
    #     ["f(x)", "f(x+1)", "f(x+2)", "f(x-1)", "f(x-2)", "x", "1"],
    #     [f],
    #     max_degree=2,
    #     n=30,
    #     epsilon=1e-13,
    # )

    # methods = generate(
    #     Domain.Real,
    #     Distribution(np.random.uniform, low=-5, high=5),
    #     [MILP(k=4, timeout=3, max_bound=10, obj=1e-12)],  # methods
    #     ["f(x)", "f(x+1)", "f(x+2)", "f(2*x)", "f(3*x)", "x", "x+1", "x+2", "1"],
    #     ["f(x)", "f(x+1)", "f(x+2)", "f(2*x)", "f(3*x)", "x", "x+1", "x+2", "1"],
    #     [f],
    #     max_degree=2,
    #     n=30,
    #     epsilon=1e-13,
    # )

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=4, timeout=3, max_bound=10, obj=1e-12)],  # methods
        ["f(x)", "f(x+1)", "x", "1"],
        ["f(x)", "f(x+1)", "x", "1"],
        [f],
        max_degree=2,
        n=30,
        epsilon=1e-13,
    )
    # NOTE: x - f(x + 1)**2 + 1 = 0
    # NOTE: f(x)**2 - f(x + 1)**2 + 1 = 0
    # NOTE: x - f(x)**2 = 0

    print("------------------------------------------")
    print_methods(methods, f)
    print("------------------------------------------")

    # ------------------------------------------

    def f(x):
        return 1 / sqrt(x)

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=4, timeout=3, max_bound=10, obj=1e-12)],  # methods
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=3,
        n=30,
        epsilon=1e-13,
    )

    print("------------------------------------------")
    print_methods(methods, f)
    print("------------------------------------------")

    # ------------------------------------------
    x = symbols("x")
    f = Function("f")

    def f(x):
        return sin(x)

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=4, timeout=5, max_bound=1, obj=1e-12)],  # methods
        ["f(x+pi)", "f(x-pi)", "f(x)", "f(-x)", "1"],
        ["f(x+pi)", "f(x-pi)", "f(x)", "f(-x)", "1"],
        [f],
        max_degree=1,
        n=30,
        epsilon=1e-13,
    )

    print("------------------------------------------")
    print_methods(methods, f)
    print("------------------------------------------")

    # exit()
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
        return 1 / (1 + exp(-x))

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=4, timeout=5, max_bound=2, obj=1e-12)],  # methods
        ["S(x+y)", "S(x-y)", "S(x)", "S(y)", "1"],  # how to call functions
        ["s(x+y)", "s(x-y)", "s(x)", "s(y)", "1"],  # function basis aka the template
        [s, S],
        max_degree=2,
        n=30,
        epsilon=1e-13,
    )

    print("------------------------------------------")
    print_methods(methods, s)
    print("------------------------------------------")

    # assert sympify(
    #     "s(x)*s(x - y) + s(x)*s(x + y) - s(x) + s(y)*s(x - y) - s(y)*s(x + y) - 2*s(x - y)*s(x + y) + s(x + y)"
    # ) in {prop for method in methods for prop in method.properties}

    def f(x):
        return x / (1 - x)

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=5, max_bound=2, obj=1e-12)],  # methods
        ["f(x+y)", "f(x)", "f(y)", "1"],  # how to call functions
        ["f(x+y)", "f(x)", "f(y)", "1"],  # function basis aka the template
        [f],
        max_degree=3,
        n=30,
        epsilon=5e-1,
    )

    print("------------------------------------------")
    print_methods(methods, f)
    print("------------------------------------------")

    assert "f(x)*f(y)*f(x + y) + 2*f(x)*f(y) + f(x) + f(y) - f(x + y)" in [
        str(prop) for method in methods for prop in method.properties
    ]

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
        [MILP(k=1, timeout=5, max_bound=2, obj=1e-10)],  # methods
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

    print("------------------------------------------")
    print_methods(methods, f, g, h)
    print("------------------------------------------")

    assert [str(prop) for method in methods for prop in method.properties] == [
        "f(x) + g(y) - g(x - y)/2 - g(x + y)/2 - h(x - y)/2 - h(x + y)/2"
    ]

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
        [MILP(k=1, timeout=5, max_bound=2, obj=1e-10)],  # methods
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

    print("------------------------------------------")
    print_methods(methods, f, g, h)
    print("------------------------------------------")

    assert [str(prop) for method in methods for prop in method.properties] == [
        "f(x) + g(y) - g(x - y)/2 - g(x + y)/2 - h(y)**2/h(x) - h(x - y)*h(x + y)/h(x)"
    ]

    def f(x):
        return x

    methods = generate(
        Domain.Real,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=5, max_bound=1, obj=1e-12)],  # methods
        ["f(x)", "f(x-x1)", "f(x-x2)", "f(x-x3)", "f(x1)", "f(x2)", "f(x3)", "1"],
        ["f(x)", "f(x-x1)", "f(x-x2)", "f(x-x3)", "f(x1)", "f(x2)", "f(x3)", "1"],
        [f],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    print("------------------------------------------")
    print_methods(methods, f)
    print("------------------------------------------")

    assert "f(x) - f(x1) - f(x2) + f(x3) - f(x - x1) - f(x - x2) + f(x - x3)" in {
        str(prop) for method in methods for prop in method.properties
    }

    def f(x):
        return x**2

    methods = generate(
        Domain.Integer,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=2, max_bound=1, obj=1e-12)],  # methods
        # fmt: off
        ["f(x1+x2+x3)", "f(x1+x2)", "f(x2+x3)", "f(x1+x3)", "f(x1)", "f(x2)", "f(x3)", "1"],
        ["f(x1+x2+x3)", "f(x1+x2)", "f(x2+x3)", "f(x1+x3)", "f(x1)", "f(x2)", "f(x3)", "1"],
        # fmt: on
        [f],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    print("------------------------------------------")
    print_methods(methods, f)
    print("------------------------------------------")

    assert (
        "f(x1) + f(x2) + f(x3) - f(x1 + x2) - f(x1 + x3) - f(x2 + x3) + f(x1 + x2 + x3)"
        in {str(prop) for method in methods for prop in method.properties}
    )

    def f(x, y, z):
        return 1 / 3 * (x + y + z)

    methods = generate(
        Domain.Integer,
        Distribution(np.random.uniform, low=-5, high=5),
        [MILP(k=1, timeout=2, max_bound=1, obj=1e-12)],  # methods
        # fmt: off
        ["f(x1+x2+x3, y1+y2+y3, z1+z2+z3)", "f(x1+x2,y1+y2,z1+z2)", "f(x2+x3,y2+y3,z2+z3)", "f(x1+x3,y1+y3,z1+z3)", "f(x1,y1,z1)", "f(x2,y2,z2)", "f(x3,y3,z3)", "1"],
        ["f(x1+x2+x3, y1+y2+y3, z1+z2+z3)", "f(x1+x2,y1+y2,z1+z2)", "f(x2+x3,y2+y3,z2+z3)", "f(x1+x3,y1+y3,z1+z3)", "f(x1,y1,z1)", "f(x2,y2,z2)", "f(x3,y3,z3)", "1"],
        # fmt: on
        [f],
        max_degree=1,
        n=30,
        epsilon=5e-1,
    )

    print("------------------------------------------")
    print_methods(methods, f)
    print("------------------------------------------")

    # TODO: we cannot find this property most of the time
    # assert sympify(
    #     "f(x1, y1, z1) + f(x2, y2, z2) + f(x3, y3, z3) - f(x1 + x2 + x3, y1 + y2 + y3, z1 + z2 + z3)"
    # ) in {prop.expr for method in methods for prop in method.properties}

    # ------------------------------------------

    # def f(x):
    #     return x / sqrt(1 + x**2)

    # methods = generate(
    #     Domain.Real,
    #     Distribution(np.random.uniform, low=-5, high=5),
    #     [MILP(k=1, timeout=3, max_bound=2, obj=1e-12)],  # methods
    #     # fmt: off
    #     ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "sqrt(f(x)*f(y))", "sqrt(f(x))", "sqrt(f(y))", "1"],
    #     ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "sqrt(f(x)*f(y))", "sqrt(f(x))", "sqrt(f(y))", "1"],
    #     # fmt: off
    #     [f],
    #     max_degree=2,
    #     n=30,
    #     epsilon=5e-1,
    # )

    # print("------------------------------------------")
    # print_methods(methods, f)
    # print("------------------------------------------")
