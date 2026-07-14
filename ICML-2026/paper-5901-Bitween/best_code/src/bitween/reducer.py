import functools
from time import time
from typing import Iterable

import sympy

from bitween.config import Config
from bitween.miscs import TimeoutFunction, getLogger

config = Config()
log = getLogger(__name__, config.logger_level)


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}  # This dictionary will keep the rank of each node

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            # Union by rank
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1  # Increase the rank if they were the same

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0  # Initialize the rank of new nodes


class Reducer:

    def __init__(self):
        pass

    @classmethod
    def get_vars(cls, props) -> list[sympy.Symbol]:
        """
        Returns a list of uniq variables from a list of properties

        >>> a,b,c,x = sympy.symbols('a b c x')
        >>> assert [a, b, c, x] == Reducer.get_vars([sympy.Expr(x**(a*b) + a**2+b+2), sympy.Eq(c**2-b,100), sympy.Gt(b**2 + c**2 + a**3,1)])
        >>> assert Reducer.get_vars(a**2+b+5*c+2) == [a, b, c]
        >>> assert Reducer.get_vars(x+x**2) == [x]
        >>> assert Reducer.get_vars([3]) == []
        >>> assert Reducer.get_vars((3,'x + c',x+b)) == [b, x]
        """

        props = props if isinstance(props, Iterable) else [props]
        props = (p for p in props if isinstance(p, (sympy.Expr, sympy.Rel)))
        vs = (v for p in props for v in p.free_symbols)
        return sorted(set(vs), key=str)

    @classmethod
    def find_and_substitute_terms(
        cls, ps: list[sympy.Expr], symbols: dict[str : sympy.Symbol] = {}
    ) -> list[sympy.Expr]:
        """
        Extract terms from the given list of expressions, and replace them with symbols
        Break complex terms here such that f(x)*(y) is a term, not a product,
        make f(x) and f(y) as terms.
        """

        # TODO: remove "&" from the expression in C e.g. `frexp(a, &e)`
        for i, p in enumerate(ps):
            ps[i] = sympy.sympify(str(p).replace("&", ""))

        for i, eqt in enumerate(ps):
            mapping = {}
            for e in eqt.args:
                if e.is_Mul or e.is_Pow:
                    for arg in e.args:
                        if arg.is_Pow:
                            mapping[arg.base] = sympy.Symbol(str(arg.base))
                        elif not arg.is_number:
                            mapping[arg] = sympy.Symbol(str(arg).replace(" ", ""))
                else:
                    if not e.is_number:
                        mapping[e] = sympy.Symbol(str(e).replace(" ", ""))

            ps[i] = eqt.subs(mapping)

        return ps

    @classmethod
    def reduce_eqts(
        cls,
        ps: list[sympy.Expr],
        timeout_sec: int = 60.0,
    ) -> list[sympy.Expr]:
        """
        Return the basis (e.g., a min subset of ps that implies ps)
        of the set of polynomial eqts using Groebner basis.
        Warning 1: Grobner basis sometimes results in a larger set of eqts,
        in which case we return the original set of eqts.
        Warning 2: seems to get stuck often.  So had to give it "nice" polynomials

        >>> a, y, b, q, k = sympy.symbols('a y b q k')


        # >>> rs = Reducer.reduce_eqts([a*y-b==0,q*y+k-x==0,a*x-a*k-b*q==0])
        __main__:DEBUG:Grobner basis: got 2 ps from 3 ps
        # >>> assert set(rs) == set([a*y - b == 0, q*y + k - x == 0])

        # >>> rs =  Reducer.reduce_eqts([x*y==6,y==2,x==3])
        __main__:DEBUG:Grobner basis: got 2 ps from 3 ps
        # >>> assert set(rs) == set([x - 3 == 0, y - 2 == 0])

        # Attribute error occurs when only 1 var, thus return as is
        # >>> rs =  Reducer.reduce_eqts([x*x==4,x==2])
        __main__:ERROR:'Ideal_1poly_field' object has no attribute 'radical'
        # >>> assert set(rs) == set([x == 2, x**2 == 4])
        """

        if len(ps) <= 1:
            return ps

        st = time()
        try:
            # NOTE: break complex terms here such that f(x)*(y) is a term,
            # not a product, make f(x) and f(y) as terms
            ps = cls.find_and_substitute_terms(ps)

            new_ps, error_msg = TimeoutFunction.call_for(
                timeout_sec=timeout_sec,
                func=sympy.groebner,
                args=(ps, *cls.get_vars(ps)),
            )

            if error_msg:
                log.warning(error_msg)

            elif new_ps is not None:
                new_ps = [x for x in new_ps]
                if len(new_ps) < len(ps):
                    log.debug(f"Grobner basis: from {len(ps)} to {len(new_ps)} ps")
                    return new_ps

            # fallback
            return ps

        except Exception as e:
            log.error(f"reduce_eqts: aborted due to exception: {e}")
            return ps

        finally:
            log.info(f"reduce_eqts took time: {time() - st:.2f}s")

    @staticmethod
    def elim_denom(p: sympy.Expr | sympy.Rel) -> sympy.Expr | sympy.Rel:
        """
        Eliminate (Integer) denominators in expression operands.
        Will not eliminate if denominators is a var (e.g.,  (3*x)/(y+2)).

        >>> x,y,z = sympy.symbols('x y z')

        >>> Reducer.elim_denom(sympy.Rational(3, 4)*x**2 + sympy.Rational(7, 5)*y**3)
        15*x**2 + 28*y**3

        >>> Reducer.elim_denom(x + y)
        x + y

        >>> Reducer.elim_denom(-sympy.Rational(3,2)*x**2 - sympy.Rational(1,24)*z**2)
        -36*x**2 - z**2

        >>> Reducer.elim_denom(15*x**2 - 12*z**2)
        15*x**2 - 12*z**2

        """
        denoms = [sympy.fraction(a)[1] for a in p.args]
        if all(denom == 1 for denom in denoms):  # no denominator like 1/2
            return p
        return p * sympy.lcm(denoms)

    @classmethod
    def get_coefs(cls, p: sympy.Expr | sympy.Rel) -> list[sympy.core.numbers.Integer]:
        """
        Return coefficients of an expression

        >>> x,y,z = sympy.symbols('x y z')
        >>> Reducer.get_coefs(3*x+5*x*y**2)
        [3, 5]
        """

        p = p.lhs if p.is_Equality else p
        return list(p.as_coefficients_dict().values())

    @classmethod
    def remove_ugly(cls, ps: list[sympy.Expr]) -> list[sympy.Expr]:

        @functools.cache
        def is_nice_coef(c: int | float) -> bool:
            return abs(c) <= config.ugly_factor or c % 10 == 0 or c % 5 == 0

        @functools.cache
        def is_nice_eqt(eqt: sympy.Expr | sympy.Rel) -> bool:
            return len(eqt.args) <= config.ugly_factor and all(
                is_nice_coef(c) for c in cls.get_coefs(eqt)
            )

        ps_ = []
        for p in ps:
            if is_nice_eqt(p):
                ps_.append(p)
            else:
                log.debug(f"ignoring large coefs {str(p)[:50]} ..")

        return ps_

    @staticmethod
    def normalize_expr(expr: sympy.Expr) -> sympy.Expr:
        """Normalize the expression so that the first significant term's coefficient is positive."""
        # Expand the expression first to simplify any compounded terms
        expr = sympy.expand(expr)

        # Choose the term to dictate the sign; here, we pick the first term when sorted
        # This assumes terms are collected by variables for consistency
        terms = expr.as_ordered_terms()

        # Check the leading term's coefficient to determine if we need to flip the signs
        if terms[0].as_coeff_mul()[0] < 0:  # Checking if the coefficient is negative
            expr = -expr  # Multiply the whole expression by -1

        return expr

    @staticmethod
    def is_similar(expr1: sympy.Expr, expr2: sympy.Expr, threshold=0.1):
        """Checks if two expressions are similar within a given threshold."""

        terms1 = {term: coeff for term, coeff in expr1.as_coefficients_dict().items()}
        terms2 = {term: coeff for term, coeff in expr2.as_coefficients_dict().items()}
        for term in set(terms1.keys()).union(terms2.keys()):
            coef1 = terms1.get(term, 0)
            coef2 = terms2.get(term, 0)
            if abs(coef1 - coef2) > threshold * max(abs(coef1), abs(coef2), 1):
                return False
        return True

    @classmethod
    def merge_equations(cls, eqts: list[sympy.Expr]) -> list[sympy.Expr]:
        """
        Merge similar equations into a single equation
        """

        if not eqts:
            return eqts

        n_eqts = len(eqts)

        for i, eq in enumerate(eqts):
            eqts[i] = cls.elim_denom(cls.normalize_expr(eq))

        eqts = cls.remove_ugly(eqts)

        uf = UnionFind()
        for i, eq in enumerate(eqts):
            uf.add(i)
            for j in range(i):
                # if cls.has_same_terms(eqts[i], eqts[j]):
                if cls.is_similar(eqts[i], eqts[j]):
                    uf.union(i, j)

        # Grouping equations by their root parent
        groups = {}
        for i, eq in enumerate(eqts):
            root = uf.find(i)
            groups.setdefault(root, []).append(eq)

        # print(groups)

        # Choose best representation from each group
        equivalence_classes = []
        for group in groups.values():
            # equivalence_classes.append(min(group, key=lambda e: len(str(e))))
            selected = find_best_polynomial(group)
            equivalence_classes.append(selected)

        log.debug(f"Union Find: from {n_eqts} to {len(equivalence_classes)} ps:")
        for eq in equivalence_classes:
            log.debug(f"{eq} = 0")

        eqts = cls.reduce_eqts(equivalence_classes)
        eqts = [cls.elim_denom(s) for s in eqts]
        eqts = cls.remove_ugly(eqts)

        return eqts

    @classmethod
    def extract_coefficients(cls, expr: sympy.Expr):
        # Use as_ordered_terms to get the terms in a default order
        ordered_terms = expr.as_ordered_terms()
        # Extract just the coefficients from each term
        return [list(term.as_coefficients_dict().keys())[0] for term in ordered_terms]

    @classmethod
    def has_same_terms(cls, expr1: sympy.Expr, expr2: sympy.Expr) -> bool:
        """Check if two expressions have the same terms."""
        terms1 = cls.extract_coefficients(expr1)
        terms2 = cls.extract_coefficients(expr2)
        return terms1 == terms2


def is_effectively_integer(n, tol=1e-2):
    """Check if a sympy number is close enough to an integer to be treated as one."""
    if isinstance(n, sympy.Integer):
        return True
    elif isinstance(n, sympy.Float):
        # Calculate the difference between the number and its nearest integer
        nearest_int = round(n)
        if abs(n - nearest_int) < tol:
            return True
    return False


def is_favored_fraction(n):
    """Check if the number is a favored multiple, including multiples of 0.5, 0.33, 0.25, and 0.2."""
    if is_effectively_integer(n * 2):
        return True
    if is_effectively_integer(n * 3):
        return True
    if is_effectively_integer(n * 4):
        return True
    if is_effectively_integer(n * 5):
        return True
    if is_effectively_integer(n * 6):
        return True
    if is_effectively_integer(n * 7):
        return True
    return False


def decimal_length(n):
    """Calculate the number of significant digits after the decimal point."""
    if is_effectively_integer(n):
        return 0
    else:
        # Convert the number to a string using enough precision
        str_n = str(n.evalf(15))  # Use evalf to convert to decimal string accurately
        if "." in str_n:
            decimal_part = str_n.split(".")[1]
            # Remove trailing zeros for the purpose of counting significant digits
            decimal_part = decimal_part.rstrip("0")
            return len(decimal_part)
        return 0


def score_polynomial(coeffs):
    # print(f"Scoring polynomial: {coeffs}")
    score = 0
    for coeff in coeffs:
        s = 0
        if is_effectively_integer(coeff):
            s = 5  # Higher points for integer coefficients
            # print(f"Score: {s} for `{coeff}`")
        elif is_favored_fraction(coeff):
            s = 3  # Additional points for favored fractions
            # print(f"Score: {s} for `{coeff}`")
        else:
            a = decimal_length(coeff)
            s = 2 - a  # Base score adjusted by decimal simplicity
            # print(f"Score: 2 - {a} = {s} for `{coeff}`")
        penalty = decimal_length(coeff) / 10  # Penalize based on complexity
        # print(f"Penalty: {penalty}")
        score += s - penalty
        # print(f"Total score: {score}")
    return score


def find_best_polynomial(polynomials):
    """
    Find the best polynomial from a list of polynomials based on a scoring system.
    """
    coeff_lists = [
        [term.as_coeff_mul()[0] for term in e.as_ordered_terms()] for e in polynomials
    ]

    best_score = float("-inf")
    best_poly = None

    for index, poly in enumerate(coeff_lists):
        current_score = score_polynomial(poly)
        # print(f"Score: {current_score} for {poly}")
        if current_score > best_score:
            best_score = current_score
            best_poly = polynomials[index]

    return best_poly


if __name__ == "__main__":

    from sympy.abc import X, Y, v, x, y

    X, Y, x, y, v = sympy.symbols("X Y x y v")

    # Example usage
    eq1 = 2 * X * y + X - 2 * Y * x - 2 * Y + v
    eq2 = -2 * X * y - X + 2 * Y * x + 2 * Y - v

    eq3 = -X * y - X / 2 + Y * x + Y - v / 2
    eq4 = -X * y - X / 2 + Y * x + 101 / 100 * Y - v / 2

    eq5 = X - x + 1
    eq6 = -X + x - 1

    eq7 = x - y**5 / 5 - y**4 / 2 - 33 * y**3 / 100 + 3 * y / 100
    eq8 = -5 * x + y**5 + 5 * y**4 / 2 + 167 * y**3 / 100 - 17 * y / 100
    eq9 = -30 * x + 6 * y**5 + 15 * y**4 + 10 * y**3 - y

    # settings.UGLY_FACTOR = 100
    # equations = Reducer.merge_equations([eq7, eq8])
    # equations = Reducer.merge_equations([eq1, eq2, eq3])
    # equations = Reducer.merge_equations([eq3, eq4])
    # equations = Reducer.merge_equations([eq3, eq4, eq5, eq6, eq7, eq8, eq9])
    equations = Reducer.merge_equations([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9])
    for eq in equations:
        print(f"{eq} = 0")
