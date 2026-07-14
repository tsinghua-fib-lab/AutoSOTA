from sympy import (
    Abs,
    Expr,
    Function,
    Symbol,
    cos,
    cosh,
    exp,
    log,
    pi,
    sign,
    simplify,
    sin,
    sinh,
    sqrt,
    symbols,
    sympify,
    tan,
    tanh,
)


# Timos code, deprecated, but kept for reference
def getValuesAndTerms(degree, values, terms, o):
    """
    returns a tuple of two lists, the first one is the values of the terms of the
    expression, the second one is the terms of the expression, both lists are sorted
    by the values of the terms

    :param degree: the degree of the expression
    :param values: the values of the terms of the expression
    :param terms: the terms of the expression
    :param o: the one symbol

    returns: a tuple of two lists, the first one is the values of the terms of the

    Remark:
    --------
    This function is deprecated, but kept for reference
    """

    def sconcato(o):
        def sconcat(v):
            return lambda y: y if v == o else v + "*" + y

        return sconcat

    def concat(v):
        return lambda y: v * y

    if degree == 1:
        return (values, terms)
    else:
        if len(values) == 0:
            return (values, terms)
        else:
            v = values[len(values) - 1]
            vname = terms[len(terms) - 1]
            out1 = list(
                map(concat(v), getValuesAndTerms(degree - 1, values, terms, o)[0])
            )
            out1names = list(
                map(
                    sconcato(o)(vname),
                    getValuesAndTerms(degree - 1, values, terms, o)[1],
                )
            )
            out2 = []
            out2names = []
            lt = values[:]
            lt.pop()
            nt = terms[:]
            nt.pop()
            (out3, out3names) = getValuesAndTerms(degree, lt, nt, o)

            return (out1 + out2 + out3, out1names + out2names + out3names)


def simplify_real_valued_expression(expr: Expr) -> bool:
    """
    to cancel out terms such as `c**2 - Abs(c)**2`
    """
    if isinstance(expr, str):
        expr = sympify(expr)

    functions = {str(f.func): Function(str(f.func)) for f in expr.atoms(Function)}

    if not isinstance(expr, Expr):
        expr = sympify(expr)

    # NOTE: in order to make c**2 == Abs(c)**2, we need to turn on the real flag
    variables = {
        str(var): Symbol(str(var), real=True, positive=True)
        for var in expr.free_symbols
    }

    expr = eval(
        f"simplify({expr})",
        functions | variables,
        {
            "simplify": simplify,
            "sqrt": sqrt,
            "Abs": Abs,
            "sin": sin,
            "cos": cos,
            "tan": tan,
            "tanh": tanh,
            "sinh": sinh,
            "cosh": cosh,
            "exp": exp,
            "log": log,
            "sign": sign,
            "pi": pi,
        },
    )

    return expr


def canonicalize(expr: str | Expr) -> Expr:
    """
    returns canonical form of the expression. It gets the sign by separating the
    multiplicative constant from the first expression of the sum of expressions, and
    if it is negative, multiply the whole expression by -1

    :param expr: the expression
    :return: the canonical form of the expression.
    """
    expr = simplify_real_valued_expression(expr)
    first_term = expr.as_ordered_terms()[0]
    coeff = first_term.as_coeff_Mul()[0]
    if sign(coeff) == -1:
        return -1 * expr
    else:
        return expr


def get_values_terms(max_degree, values, terms, one_symbol, functions, depth=1):
    """
    returns the terms of the expression

    :param max_degree: the maximum degree of the expression
    :param values: the values of the terms of the expression
    :param terms: the terms of the expression
    :param one_symbol: the one symbol

    :return: a tuple of two lists, the first one is the values of the terms of the
        expression, the second one is the terms of the expression,
        both lists are ordered
    """

    def generate_combinations(values, terms, depth, current_value, current_term):
        if depth == 1:
            for i, term in enumerate(terms):
                yield current_value * values[i], current_term + [term]
        else:
            for i, term in enumerate(terms):
                if term != one_symbol:
                    yield from generate_combinations(
                        values[: i + 1],
                        terms[: i + 1],
                        depth - 1,
                        current_value * values[i],
                        current_term + [term],
                    )

    def compose(functions, depth, base_values, base_terms: list):
        if depth == 0:
            return base_values, base_terms

        prev_values, prev_terms = compose(functions, depth - 1, base_values, base_terms)

        new_terms = []
        new_values = []

        count = 0
        for expr in functions:
            for i, term in enumerate(prev_terms):
                if term == one_symbol:
                    count += 1
                    if count > 1:
                        continue
                if isinstance(expr, Expr):
                    new_values.append(
                        expr.subs(expr.free_symbols.pop(), prev_values[i]).evalf()
                    )
                elif isinstance(expr, str):
                    expr = sympify(expr)
                    new_values.append(
                        expr.subs(expr.free_symbols.pop(), prev_values[i]).evalf()
                    )
                else:
                    raise ValueError("Unsupported type in Expression")

                if isinstance(term, str):
                    term = sympify(term)

                if isinstance(expr, Expr):
                    new_terms.append(str(expr.subs(expr.free_symbols.pop(), term)))
                else:
                    raise ValueError("Unsupported type in term")

        return new_values, new_terms

    result = []
    # Add degree 1 terms
    result.extend(terms)

    # Generate terms based on degree up to max_degree
    for degree in range(2, max_degree + 1):
        for combination_values, combination in generate_combinations(
            values, terms, degree, 1, []
        ):
            values.append(combination_values)
            result.append("*".join(combination))

    # if functions to compose are not given, return the result
    if len(functions) == 0:
        return values, result
    else:
        return compose(functions, depth, values, result)


def get_terms(max_degree, terms, one_symbol, composition_functions, depth=1):
    """
    returns the terms of the expression

    :param max_degree: the maximum degree of the expression
    :param terms: the terms of the expression
    :param one_symbol: the one symbol

    :return: a tuple of two lists, the first one is the values of the terms of the
        expression, the second one is the terms of the expression,
        both lists are ordered
    """
    result = []

    # Exclude the one_symbol from terms
    terms = [term for term in terms if term != one_symbol]

    # Helper function to generate combinations
    def generate_combinations(terms, depth, current_term):
        if depth == 1:
            for term in terms:
                yield current_term + [term]
        else:
            for i, term in enumerate(terms):
                yield from generate_combinations(
                    terms[: i + 1], depth - 1, current_term + [term]
                )

    def compose(functions, depth, base_terms):
        if depth == 0:
            return base_terms

        previous_terms = compose(functions, depth - 1, base_terms)
        new_terms = []

        for expr in functions:
            for term in previous_terms:
                if isinstance(term, str):
                    term = sympify(term)

                if isinstance(expr, Expr):
                    new_terms.append(str(expr.subs(expr.free_symbols.pop(), term)))
                else:
                    raise ValueError("Unsupported type in term")

        return new_terms

    # Add degree 1 terms
    result.extend(terms)

    # Generate terms based on degree up to max_degree
    for degree in range(2, max_degree + 1):
        for combination in generate_combinations(terms, degree, []):
            result.append("*".join(combination))

    if len(composition_functions) == 0:
        return result
    else:
        result = compose(composition_functions, depth, result)
        return result


def compose_terms(functions: list, depth=1, base_terms=[Symbol("x")]):
    """
    Generate all possible terms from a set of functions based on a given depth.

    :param functions: List of function symbols.
    :param depth: Depth of nesting.
    :param base_terms: List of terms to start with.

    :return: List of generated terms.
    """
    if depth == 0:
        return base_terms

    previous_terms = compose_terms(functions, depth - 1, base_terms)
    new_terms = []

    for expr in functions:
        for term in previous_terms:
            if isinstance(term, str):
                term = sympify(term)

            if isinstance(expr, Expr):
                new_terms.append(str(expr.subs(expr.free_symbols.pop(), term)))
            else:
                raise ValueError("Unsupported type in term")

    return new_terms


def test_canonicalize():
    x = Symbol("x")
    assert canonicalize(x + 2 * x - 3) == 3 * x - 3
    assert canonicalize(-x + 2 * x - 3) == x - 3
    assert canonicalize(-2 * x - 3) == 2 * x + 3
    assert canonicalize(-3 * x * x - 3) == 3 * x * x + 3
    print("canonicalize() passed")


def test_get_values_terms():
    values, terms = get_values_terms(3, [2, 3, 1], ["x", "y", "1"], "1")
    print(values, terms)
    assert (
        [2, 3, 1, 4, 6, 9, 8, 12, 18, 27],
        ["x", "y", "1", "x*x", "y*x", "y*y", "x*x*x", "y*x*x", "y*y*x", "y*y*y"],
    ) == (values, terms)


def test_gen_value_term():
    def generate_combinations(
        values: list, terms: list, depth, current_value, current_term
    ):
        if depth == 1:
            for i, term in enumerate(terms):
                yield current_value * values[i], current_term + [term]
        else:
            for i, term in enumerate(terms):
                yield from generate_combinations(
                    values[: i + 1],
                    terms[: i + 1],
                    depth - 1,
                    current_value * values[i],
                    current_term + [term],
                )

    values = [2, 3, 4]
    terms = ["x", "y", "z"]

    for value, term in generate_combinations(values, terms, 3, 1, []):
        print(value, term)


if __name__ == "__main__":  # noqa E123
    """
    Test cases
    """

    from sympy import Function, Symbol, cos, sin, sqrt, symbols, tan

    print("--------------------------------------------------")

    test_gen_value_term()

    print("--------------------------------------------------")

    test_get_values_terms()

    print("--get_values_terms()------------------------------")

    # composition_functions = [lambda x: x**2, lambda x: x**3]
    x = Symbol("x")

    functions = [x, cos(x), sin(x)]
    print(f"functions: {functions}")
    values = [2, 3, 1]
    terms = ["x", "y", "1"]
    print(f"terms: {terms}")
    print(f"values: {values}")
    values, terms = get_values_terms(2, values, terms, "1", functions, depth=1)
    print("----")
    print(f"terms: {terms}")
    print(f"values: {values}")

    print("--get_terms()------------------------------------")

    functions = [x, cos(x), sin(x)]
    print(f"functions: {functions}")
    values = [2, 3, 1]
    terms = ["x", "y", "1"]
    print(f"terms: {terms}")
    terms = get_terms(2, terms, "1", functions, depth=1)
    print("----")
    print(f"terms: {terms}")

    print("--simplify abs terms------------------------------------")

    expr = x**2 - Abs(x) ** 2
    print(expr)
    expr = simplify_real_valued_expression(expr)
    print(expr)
    assert expr == 0

    print("--------------------------------------------------")

    test_canonicalize()

    print("--------------------------------------------------")

    terms = get_terms(2, ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"], "1")
    print(terms)
    terms.sort()
    rhs = [
        "f(x+y)",
        "f(x-y)",
        "f(x)",
        "f(y)",
        "f(y)*f(x+y)",
        "f(y)*f(x-y)",
        "f(y)*f(x)",
        "f(y)*f(y)",
        "f(x)*f(x+y)",
        "f(x)*f(x-y)",
        "f(x-y)*f(x+y)",
        "f(x-y)*f(x-y)",
        "f(x+y)*f(x+y)",
        "f(x)*f(x)",
    ]
    rhs.sort()
    assert terms == rhs

    print("--------------------------------------------------")

    terms = get_terms(3, ["f(x+y)", "f(x-y)", "f(x)", "f(y)", "1"], "1")
    print(terms)
    terms.sort()
    rhs = [
        "f(x+y)",
        "f(x-y)",
        "f(x)",
        "f(y)",
        "f(y)*f(x+y)",
        "f(y)*f(x-y)",
        "f(y)*f(x)",
        "f(y)*f(y)",
        "f(x)*f(x+y)",
        "f(x)*f(x-y)",
        "f(x-y)*f(x+y)",
        "f(x-y)*f(x-y)",
        "f(x+y)*f(x+y)",
        "f(y)*f(y)*f(x+y)",
        "f(y)*f(y)*f(x-y)",
        "f(y)*f(y)*f(x)",
        "f(y)*f(y)*f(y)",
        "f(y)*f(x)*f(x+y)",
        "f(y)*f(x)*f(x-y)",
        "f(y)*f(x)*f(x)",
        "f(y)*f(x-y)*f(x+y)",
        "f(y)*f(x-y)*f(x-y)",
        "f(y)*f(x+y)*f(x+y)",
        "f(x)*f(x)*f(x+y)",
        "f(x)*f(x)*f(x-y)",
        "f(x)*f(x)*f(x)",
        "f(x)*f(x-y)*f(x+y)",
        "f(x)*f(x-y)*f(x-y)",
        "f(x)*f(x+y)*f(x+y)",
        "f(x-y)*f(x-y)*f(x+y)",
        "f(x-y)*f(x-y)*f(x-y)",
        "f(x-y)*f(x+y)*f(x+y)",
        "f(x+y)*f(x+y)*f(x+y)",
        "f(x)*f(x)",
    ]
    rhs.sort()
    assert terms == rhs

    def composition(functions: list, depth=1, base_terms=[Symbol("x")]):
        if depth == 0:
            return base_terms

        previous_terms = composition(functions, depth - 1, base_terms)
        new_terms = []

        for func in functions:
            for term in previous_terms:
                new_terms.append(func(term))

        return new_terms

    x, y = symbols("x y")
    functions = [sin, cos, tan]
    base_terms = [x, y, x * y, x**2, y**2, sqrt(x), sqrt(y)]
    depth = 1
    print(composition(functions, depth, base_terms))

    print("--------------------------------------------------")

    x, y = symbols("x y")
    f = Function("f")
    g = Function("g")
    h = Function("h")
    functions = [f, g, h, sqrt]
    base_terms = [x, y]
    depth = 2
    print(composition(functions, depth, base_terms))

    print("--------------------------------------------------")
