"""
To run doctest
$ python -m doctest -v src/bitween/miscs.py
"""

import functools
import itertools
import logging
from typing import Any, Callable, Iterable, Optional

import numpy as np
import sympy
from func_timeout import FunctionTimedOut, func_timeout
from sympy.core.function import AppliedUndef

from bitween.config import Config

_default_logging_format = "%(name)s:%(levelname)s:%(message)s"
_empty_logging_format = "%(message)s"


def getLogger(
    name: str,
    level: int,
    empty_format: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        _empty_logging_format if empty_format else _default_logging_format
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def addFileHandler(
    logger: logging.Logger,
    file: str,
    empty_format: bool = False,
) -> logging.Handler:
    fch = logging.FileHandler(file)
    fch.setLevel(logger.getEffectiveLevel())
    formatter = logging.Formatter(
        _empty_logging_format if empty_format else _default_logging_format
    )
    fch.setFormatter(formatter)
    logger.addHandler(fch)
    return fch


def removeFileHandler(logger: logging.Logger, handler: logging.Handler):
    logger.removeHandler(handler)


def getLogLevel(level: int) -> int:
    assert level in set(range(5))

    if level == 0:
        return logging.CRITICAL
    elif level == 1:
        return logging.ERROR
    elif level == 2:
        return logging.WARNING
    elif level == 3:
        return logging.INFO
    else:
        return logging.DEBUG


config = Config()
log = getLogger(__name__, config.logger_level)


class TimeoutFunction:
    @staticmethod
    def call_for(
        timeout_sec: Optional[float],
        func: callable,
        args: tuple = None,
        kwargs: dict = None,
    ) -> tuple[any, str]:
        args = args or ()
        kwargs = kwargs or {}
        func_name = func.__name__

        try:
            out = func_timeout(timeout_sec, func, args, kwargs)
            return out, ""

        except FunctionTimedOut:
            msg = f"Calling `{func_name}` timed out after {timeout_sec} sec"
            log.warning(msg)
            return None, msg

        except Exception as e:
            msg = f"Exception when calling `{func_name}`: {e}"
            log.error(msg, exc_info=e)
            return None, msg


class Symbolic:
    @staticmethod
    def is_expr(x) -> bool:
        return isinstance(x, sympy.Expr)

    @classmethod
    def get_vars(cls, props) -> list[sympy.Symbol]:
        """
        Returns a list of uniq variables from a list of properties

        >>> a,b,c,x = sympy.symbols('a b c x')
        >>> assert [a, b, c, x] == Symbolic.get_vars([sympy.Expr(x**(a*b) + a**2+b+2), sympy.Eq(c**2-b,100), sympy.Gt(b**2 + c**2 + a**3,1)])
        >>> assert Symbolic.get_vars(a**2+b+5*c+2) == [a, b, c]
        >>> assert Symbolic.get_vars(x+x**2) == [x]
        >>> assert Symbolic.get_vars([3]) == []
        >>> assert Symbolic.get_vars((3,'x + c',x+b)) == [b, x]
        """

        props = props if isinstance(props, Iterable) else [props]
        props = (p for p in props if isinstance(p, (sympy.Expr, sympy.Rel)))
        vs = (v for p in props for v in p.free_symbols)
        return sorted(set(vs), key=str)

    @staticmethod
    @functools.cache
    def str2rat(s: str) -> sympy.Rational:
        """
        Convert the input 's' to a rational number if possible.

        Examples:
        >>> print(Symbolic.str2rat('.3333333'))
        3333333/10000000
        >>> print(Symbolic.str2rat('3/7'))
        3/7
        >>> print(Symbolic.str2rat('1.'))
        1
        >>> print(Symbolic.str2rat('1.2'))
        6/5
        >>> print(Symbolic.str2rat('.333'))
        333/1000
        >>> print(Symbolic.str2rat('-.333'))
        -333/1000
        >>> print(Symbolic.str2rat('-12.13'))
        -1213/100
        """
        return sympy.Rational(s)

    @staticmethod
    def get_terms(symbols: list[sympy.Symbol], deg: int) -> list:
        """
        get a list of terms from the given list of vars and deg
        the number of terms is len(rs) == binomial(len(symbols)+d, d)

        >>> a,b,c,d,e,f = sympy.symbols('a b c d e f')
        >>> ts = Symbolic.get_terms([a, b], 3)
        >>> assert ts == [1, a, b, a**2, a*b, b**2, a**3, a**2*b, a*b**2, b**3]
        >>> Symbolic.get_terms([a,b,c,d,e,f], 3)
        [1, a, b, c, d, e, f, a**2, a*b, a*c, a*d, a*e, a*f, b**2, b*c, b*d, b*e, b*f, c**2, c*d, c*e, c*f, d**2, d*e, d*f, e**2, e*f, f**2, a**3, a**2*b, a**2*c, a**2*d, a**2*e, a**2*f, a*b**2, a*b*c, a*b*d, a*b*e, a*b*f, a*c**2, a*c*d, a*c*e, a*c*f, a*d**2, a*d*e, a*d*f, a*e**2, a*e*f, a*f**2, b**3, b**2*c, b**2*d, b**2*e, b**2*f, b*c**2, b*c*d, b*c*e, b*c*f, b*d**2, b*d*e, b*d*f, b*e**2, b*e*f, b*f**2, c**3, c**2*d, c**2*e, c**2*f, c*d**2, c*d*e, c*d*f, c*e**2, c*e*f, c*f**2, d**3, d**2*e, d**2*f, d*e**2, d*e*f, d*f**2, e**3, e**2*f, e*f**2, f**3]
        """

        assert deg >= 0, deg
        assert symbols, symbols

        # ss_ = ([1] if ss else (1,)) + ss
        symbols_ = [1] + symbols
        combs = itertools.combinations_with_replacement(symbols_, deg)
        terms = [sympy.prod(c) for c in combs]
        return terms

    @staticmethod
    def show_removed(s: str, orig_siz: int, new_siz: int, elapsed_time: float) -> None:
        assert orig_siz >= new_siz, (orig_siz, new_siz)
        n_removed = orig_siz - new_siz
        log.debug(
            f"{s}: removed {n_removed} invs "
            f"in {elapsed_time:.2f}s (orig {orig_siz}, new {new_siz})"
        )

    @classmethod
    def linear_solve(cls, pivot, terms, data):
        sample_size = data.shape[0]
        if config.use_sample_rate_milp and config.sample_rate_milp > 1:
            sample_size = int(len(terms) * config.sample_rate_milp)
            threshold = 20
            if sample_size < threshold:
                if data.shape[0] > threshold:
                    sample_size = threshold
                else:  # use all data
                    sample_size = data.shape[0]

            if sample_size < data.shape[0]:
                sample_indices = np.random.choice(
                    data.shape[0], sample_size, replace=False
                )
                data = data[sample_indices]
            else:
                sample_size = data.shape[0]

        data = data.astype(int)

        # Create a row of zeros with the same length as the other rows in arr
        new_row = np.zeros(data.shape[1])
        new_row[terms.index(pivot)] = 1  # Set the pivot to 1
        new_row[-1] = 1  # Set the last element to -1
        data = np.vstack([data, new_row])

        matrix = sympy.Matrix(data)

        # turn terms into sympy symbols
        vars = [sympy.Symbol(term) for term in terms]

        log.debug(
            f"solving {len(terms)} terms using " f"{data.shape[0]} eqts for {pivot}"
        )

        sol = sympy.linsolve(matrix, vars)

        # print(sol)

        if sol == sympy.EmptySet:
            return None, sample_size

        vals = list(list(sol)[0])

        if all(v == 0 for v in vals):
            return None, sample_size

        # eqts_ = cls.instantiate_template(terms, vars, vals)
        expr = 0
        for i, v in enumerate(vals):
            if isinstance(v, sympy.Number):
                expr += sympy.Rational(v) * sympy.sympify(terms[i])
            elif isinstance(v, sympy.Symbol):
                expr += sympy.sympify(terms[i])

        # return [sympy.Eq(eqt, 0) for eqt in eqts_]
        return expr, sample_size

    # NOTE: This method is not used, it is just for reference
    @classmethod
    def solve_eqts(
        cls,
        eqts: list[sympy.Expr | sympy.Rel],
        terms: list[Any],
        uks: list[sympy.Symbol],
    ) -> list[sympy.Eq]:
        assert eqts, eqts
        assert terms, terms
        assert uks, uks
        assert len(terms) == len(uks), (terms, uks)
        # assert len(eqts) >= len(uks), (len(eqts), len(uks))

        log.debug(f"solving {len(uks)} uks using {len(eqts)} eqts")
        sol = sympy.linsolve(eqts, uks)
        # print(eqts)
        # print(uks)
        # print(sol)
        vals = list(list(sol)[0])

        if all(v == 0 for v in vals):
            return []

        eqts_ = cls.instantiate_template(terms, uks, vals)
        log.debug(f"got {len(eqts_)} eqts after instantiating")
        eqts_ = cls.refine(eqts_)
        log.debug(f"got {len(eqts_)} eqts after refinement")
        # return [sympy.Eq(eqt, 0) for eqt in eqts_]
        return eqts_

    # NOTE: This method is not used, it is just for reference
    @classmethod
    def instantiate_template(cls, terms: list, uks: list, vs: list) -> list:
        """
        Instantiate a template with solved coefficient values

        # sage:var('uk_0,uk_1,uk_2,uk_3,uk_4,r14,r15,a,b,y')
        (uk_0, uk_1, uk_2, uk_3, uk_4, r14, r15, a, b, y)

        # sage:sols = [{uk_0: -2*r14 + 7/3*r15, uk_1: -1/3*r15, uk_4: r14, uk_2: r15, uk_3: -2*r14}]
        # sage:Miscs.instantiate_template(uk_1*a + uk_2*b + uk_3*x + uk_4*y + uk_0 == 0, sols)
        [-2*x + y - 2 == 0, -1/3*a + b + 7/3 == 0]

        # sage:Miscs.instantiate_template(uk_1*a + uk_2*b + uk_3*x + uk_4*y + uk_0 == 0, [])
        []
        """
        assert terms, terms
        assert uks, uks
        assert len(terms) == len(uks) == len(vs), (terms, uks, vs)

        cs = [(t, u, v) for t, u, v in zip(terms, uks, vs) if v != 0]
        terms_, uks_, vs_ = zip(*cs)

        eqt = sum(t * v for t, v in zip(terms_, vs_))

        uk_vs = cls.get_vars(vs_)

        if not uk_vs:
            return eqt

        sols = [
            eqt.xreplace({uk: (1 if j == i else 0) for j, uk in enumerate(uk_vs)})
            for i, uk in enumerate(uk_vs)
        ]
        return sols

    @staticmethod
    def simplify_idxs(
        ordered_idxs: list[int], imply_f: Callable[[set[int], int], bool]
    ) -> list[int]:
        """
        attempt to remove i in idxs if imply_f returns true
        Note: the order of idxs determine what to get checked (and removed)
        """
        assert isinstance(ordered_idxs, list), ordered_idxs
        assert ordered_idxs == list(range(len(ordered_idxs))), ordered_idxs

        results = set(ordered_idxs)

        for i in reversed(ordered_idxs):
            if i not in results:
                continue
            others = results - {i}
            if others and imply_f(others, i):
                results = others

        return sorted(results)

    @classmethod
    def find_and_substitute_terms(cls, ps: list[sympy.Expr]):
        """
        Extract terms from the given list of expressions, and replace them with symbols
        """
        for i, p in enumerate(ps):
            ps[i] = sympy.sympify(str(p))

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
    def reduce_eqts(cls, ps: list[sympy.Expr]) -> list[sympy.Expr]:
        """
        Return the basis (e.g., a min subset of ps that implies ps)
        of the set of polynomial eqts using Groebner basis.
        Warning 1: Grobner basis sometimes results in a larger set of eqts,
        in which case we return the original set of eqts.
        Warning 2: seems to get stuck often.  So had to give it "nice" polynomials

        >>> a, y, b, q, k = sympy.symbols('a y b q k')


        # >>> rs = Symbolic.reduce_eqts([a*y-b==0,q*y+k-x==0,a*x-a*k-b*q==0])
        __main__:DEBUG:Grobner basis: got 2 ps from 3 ps
        # >>> assert set(rs) == set([a*y - b == 0, q*y + k - x == 0])

        # >>> rs =  Symbolic.reduce_eqts([x*y==6,y==2,x==3])
        __main__:DEBUG:Grobner basis: got 2 ps from 3 ps
        # >>> assert set(rs) == set([x - 3 == 0, y - 2 == 0])

        # Attribute error occurs when only 1 var, thus return as is
        # >>> rs =  Symbolic.reduce_eqts([x*x==4,x==2])
        __main__:ERROR:'Ideal_1poly_field' object has no attribute 'radical'
        # >>> assert set(rs) == set([x == 2, x**2 == 4])
        """

        if len(ps) <= 1:
            return ps

        # NOTE: break complex terms here such that f(x)*(y) is a term, not a
        # product, make f(x) and f(y) as terms
        ps = cls.find_and_substitute_terms(ps)

        ps_ = sympy.groebner(ps, *cls.get_vars(ps))
        ps_ = [x for x in ps_]
        log.debug(f"Grobner basis: from {len(ps)} to {len(ps_)} ps")
        return ps_ if len(ps_) < len(ps) else ps

    @staticmethod
    def elim_denom(p: sympy.Expr | sympy.Rel) -> sympy.Expr | sympy.Rel:
        """
        Eliminate (Integer) denominators in expression operands.
        Will not eliminate if denominators is a var (e.g.,  (3*x)/(y+2)).

        >>> x,y,z = sympy.symbols('x y z')

        >>> Symbolic.elim_denom(sympy.Rational(3, 4)*x**2 + sympy.Rational(7, 5)*y**3)
        15*x**2 + 28*y**3

        >>> Symbolic.elim_denom(x + y)
        x + y

        >>> Symbolic.elim_denom(-sympy.Rational(3,2)*x**2 - sympy.Rational(1,24)*z**2)
        -36*x**2 - z**2

        >>> Symbolic.elim_denom(15*x**2 - 12*z**2)
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
        >>> Symbolic.get_coefs(3*x+5*x*y**2)
        [3, 5]
        """

        p = p.lhs if p.is_Equality else p
        return list(p.as_coefficients_dict().values())

    @classmethod
    def remove_ugly(
        cls, ps: list[sympy.Expr | sympy.Rel]
    ) -> list[sympy.Expr | sympy.Rel]:
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

    @classmethod
    def refine(cls, eqts: list[sympy.Expr | sympy.Rel]) -> list[sympy.Expr | sympy.Rel]:
        if not eqts:
            return eqts

        eqts = [cls.elim_denom(s) for s in eqts]
        eqts = cls.remove_ugly(eqts)
        eqts = cls.reduce_eqts(eqts)
        eqts = [cls.elim_denom(s) for s in eqts]
        eqts = cls.remove_ugly(eqts)

        return eqts

    @staticmethod
    def substitute_function_calls(
        expr: sympy.Expr,
        variables: dict,
    ) -> dict[str, list[tuple[tuple[str], tuple[float]]]]:
        res_dct = {}

        for subexpr in sympy.preorder_traversal(expr):
            if isinstance(subexpr, AppliedUndef):
                func_name = subexpr.func.name

                args = subexpr.args
                str_args = str(args).replace(",)", ")")
                subs_args = tuple(map(lambda e: e.subs(variables), args))

                value = (str_args, subs_args)

                res_dct.setdefault(func_name, [])
                res_dct[func_name].append(value)

        return res_dct


if __name__ == "__main__":
    import doctest

    doctest.testmod()
