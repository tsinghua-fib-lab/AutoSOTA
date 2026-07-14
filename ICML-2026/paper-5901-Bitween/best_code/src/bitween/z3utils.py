from __future__ import annotations

import ast
import functools
import operator
from time import time
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import sympy
import z3

from bitween.config import Config, InvariantType
from bitween.miscs import Symbolic, getLogger

config = Config()
log = getLogger(__name__, config.logger_level)


class Z3:
    zTrue = z3.BoolVal(True)
    zFalse = z3.BoolVal(False)
    TIMEOUT = config.z3_timeout * 1000

    @classmethod
    def _process_fs(
        cls: Type[Z3],
        fs: List[None | z3.ExprRef],
        and_or_f: Callable[[List[z3.ExprRef]], z3.ExprRef],
    ) -> Union[None, z3.ExprRef]:

        assert isinstance(fs, list) and all(
            isinstance(f, z3.ExprRef) or f is None for f in fs
        ), fs

        fs = [f for f in fs if f is not None]
        if not fs:
            return None
        if len(fs) == 1:
            return fs[0]

        return and_or_f(fs)

    @classmethod
    def _and(
        cls: Type[Z3], fs: List[Union[None, z3.ExprRef]]
    ) -> Union[None, z3.ExprRef]:
        return cls._process_fs(fs, z3.And)

    @classmethod
    def _or(
        cls: Type[Z3], fs: List[Union[None, z3.ExprRef]]
    ) -> Union[None, z3.ExprRef]:
        return cls._process_fs(fs, z3.Or)

    @classmethod
    def is_var(cls: Type[Z3], v: Any) -> bool:
        return z3.is_const(v) and v.decl().kind() == z3.Z3_OP_UNINTERPRETED

    @classmethod
    def _get_vars(cls: Type[Z3], f: z3.ExprRef, rs: Set[Any]):
        """
        Helper method to obtain variables from a formula f recursively.
        Results are stored in the list rs.
        """
        assert isinstance(f, z3.ExprRef) or z3.is_const(f), f
        if z3.is_const(f):
            if cls.is_var(f):
                rs.add(f)
        else:
            for c in f.children():
                cls._get_vars(c, rs)

    @classmethod
    @functools.cache
    def get_vars(cls: Type[Z3], f: z3.ExprRef) -> FrozenSet[z3.ExprRef]:
        """
        >>> x,y,z = z3.Ints("x y z")
        >>> assert(Z3.get_vars(z3.And(x + y == z , y + z == z)) == {z, y, x})
        """
        assert isinstance(f, z3.ExprRef), f

        rs: Set[Any] = set()
        cls._get_vars(f, rs)
        return frozenset(rs)

    @classmethod
    def create_solver(
        cls: Type[Z3], maximize: bool = False
    ) -> Union[z3.Optimize, z3.Solver]:
        assert isinstance(maximize, bool), maximize

        solver = z3.Optimize() if maximize else z3.Solver()
        solver.set(timeout=cls.TIMEOUT)
        solver.set("timeout", cls.TIMEOUT)
        return solver

    @classmethod
    def extract(
        cls: Type[Z3], models: List[z3.ModelRef], f: Callable[[str], str]
    ) -> Tuple[List[Dict[str, str]], bool]:

        assert (
            models is None
            or models is False
            or (
                isinstance(models, list)
                and all(isinstance(m, z3.ModelRef) for m in models)
                and models
            )
        ), models

        cexs: List = list()
        is_succ = models is not None
        if is_succ and models:  # disproved
            cexs = []
            for model in models:
                cex = {}
                for v in model:
                    mv = str(model[v])
                    try:
                        cex[str(v)] = mv if f is None else f(mv)
                    except ValueError:
                        # mlog.warning('cannot analyze {}'.format(model))
                        pass
                cexs.append(cex)
        return cexs, is_succ

    @classmethod
    def get_models(
        cls: Type[Z3], f: z3.ExprRef, k: int
    ) -> Tuple[Union[None, bool, List[Union[z3.Optimize, z3.Solver]]], int]:
        """
        Returns the first k models satisfiying f.
        If f is not satisfiable, returns False.
        If f cannot be solved, returns None
        If f is satisfiable, returns the first k models
        Note that if f is a tautology, i.e., True, then the result is []
        """
        assert z3.is_expr(f), f
        assert k >= 1, k
        solver = cls.create_solver(maximize=False)
        solver.add(f)
        models = []
        i = 0
        while solver.check() == z3.sat and i < k:
            i = i + 1
            m = solver.model()
            if not m:  # if m == []
                log.warning("sat but no model")
                break
            models.append(m)
            # create new constraint to block the current model
            ands = []
            for v in m:
                try:
                    e = v() == m[v]
                    ands.append(e)
                except z3.Z3Exception:
                    """
                    when the model contains functions, e.g.,
                    [..., div0 = [(3, 2) -> 1, else -> 0]]
                    """
                    # mlog.warning('cannot analyze {}'.format(m))
                    pass
            block = z3.Not(z3.And(ands))
            solver.add(block)

        stat = solver.check()
        if stat == z3.unknown:  # for unknown/unsat/sat, use == instead of is
            rs: Union[None, bool, List[Any]] = None
        elif stat == z3.unsat and i == 0:
            rs = False
        else:
            if models:
                rs = models
            else:
                # tmp fix,  ProdBin has a case when
                # stat is sat but model is []
                # so tmp fix is to treat that as unknown
                rs = None
                stat = z3.unknown

        assert not (isinstance(rs, list) and not rs), rs
        return rs, stat

    @classmethod
    def is_valid(cls: Type[Z3], claim: z3.ExprRef) -> bool:
        _, stat = cls.get_models(z3.Not(claim), 1)
        return stat == z3.unsat

    @classmethod
    def imply(cls: Type[Z3], fs: List[z3.ExprRef], g: z3.ExprRef) -> bool:
        """
        >>> var('x y')
        (x, y)
        >>> assert Z3.imply([x-6==0],x*x-36==0)
        >>> assert Z3.imply([x-6==0,x+6==0],x*x-36==0)
        >>> assert not Z3.imply([x*x-36==0],x-6==0)
        >>> assert not Z3.imply([x-6==0],x-36==0)
        >>> assert Z3.imply([x-7>=0], x>=6)
        >>> assert not Z3.imply([x-7>=0], x>=8)
        >>> assert not Z3.imply([x-6>=0], x-7>=0)
        >>> assert not Z3.imply([x-7>=0,y+5>=0],x+y-3>=0)
        >>> assert Z3.imply([x-7>=0,y+5>=0],x+y-2>=0)
        >>> assert Z3.imply([x-2*y>=0,y-1>=0],x-2>=0)
        >>> assert not Z3.imply([],x-2>=0)
        >>> assert Z3.imply([x-7>=0,y+5>=0],x+y-2>=0)
        >>> assert Z3.imply([x**2-9>=0,x>=0],x-3>=0)
        >>> assert Z3.imply([x-6==0],x*x-36==0)
        >>> assert not Z3.imply([x+7>=0,y+5>=0],x*y+36>=0)
        >>> assert not Z3.imply([x+7>=0,y+5>=0],x*y+35>=0)
        >>> assert not Z3.imply([x+7>=0,y+5>=0],x*y-35>=0)
        >>> assert not Z3.imply([x+7>=0],x-8>=0)
        >>> assert Z3.imply([x+7>=0],x+8>=0)
        >>> assert Z3.imply([x>=7,y>=5],x*y>=35)
        >>> assert not Z3.imply([x>=-7,y>=-5],x*y>=35)

        # >>> assert not Z3.imply([1/2*x**2 - 3/28*x + 1 >= 0],1/20*x**2 - 9/20*x + 1 >= 0,use_reals=True)
        # >>> assert Z3.imply([1/20*x**2 - 9/20*x + 1 >= 0],1/2*x**2 - 3/28*x + 1 >= 0,use_reals=True)
        # >>> assert Z3.imply([x+7>=0],x+8.9>=0,use_reals=True)

        """

        if not fs:
            return False  # conservative approach

        fs = [Z3.parse(str(f)) for f in fs]
        g = Z3.parse(str(g))

        return cls._imply(fs, g)

    @classmethod
    def _imply(
        cls: Type[Z3],
        fs: z3.ExprRef | List[z3.ExprRef],
        g: z3.ExprRef,
        is_conj: Optional[bool] = True,
    ) -> bool:
        assert z3.is_expr(g), g

        if is_conj:  # And(fs) => g
            if z3.is_expr(fs):
                claim = z3.Implies(fs, g)
            else:
                claim = z3.Implies(z3.And(fs), g)
        else:  # g => Or(fs)
            if z3.is_expr(fs):
                claim = z3.Implies(g, fs)
            else:
                claim = z3.Implies(g, z3.Or(fs))

        models, _ = cls.get_models(z3.Not(claim), k=1)
        return models is False

    @classmethod
    def parse(cls, node):
        """
        Parse a string to a Z3 expression
        E.g.,  parse("x>=10*10")

        Note cannot parse something like tCtr == y - 1/2*sqrt(4*y**2 - 8*x + 4*y + 1) + 1/2
        """
        # print(ast.dump(node))

        if isinstance(node, str):
            node = node.replace("^", "**")

            tnode = ast.parse(node)
            tnode = tnode.body[0].value
            try:
                expr = cls.parse(tnode)
                expr = z3.simplify(expr)
                return expr
            except NotImplementedError:
                log.error(f"cannot parse: '{node}'\n{ast.dump(tnode)}")
                raise

        elif isinstance(node, ast.BoolOp):
            vals = [cls.parse(v) for v in node.values]
            op = cls.parse(node.op)
            return op(vals)

        elif isinstance(node, ast.And):
            return z3.And

        elif isinstance(node, ast.Or):
            return z3.Or

        elif isinstance(node, ast.BinOp):
            left = cls.parse(node.left)
            right = cls.parse(node.right)
            op = cls.parse(node.op)
            return op(left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = cls.parse(node.operand)
            op = cls.parse(node.op)
            return op(operand)

        elif isinstance(node, ast.Compare):
            assert len(node.ops) == 1 and len(node.comparators) == 1, ast.dump(node)
            left = cls.parse(node.left)
            right = cls.parse(node.comparators[0])
            op = cls.parse(node.ops[0])
            return op(left, right)

        elif isinstance(node, ast.Name):
            if config.invariant_type == InvariantType.INT:
                return z3.Int(str(node.id))
            elif (
                config.invariant_type == InvariantType.REAL
                or config.invariant_type == InvariantType.MIXED
            ):
                return z3.Real(str(node.id))
            else:
                raise NotImplementedError(config.invariant_type)
        elif isinstance(node, ast.Num):
            return z3.IntVal(str(node.n))
        elif isinstance(node, ast.Add):
            return operator.add
        elif isinstance(node, ast.Mult):
            return operator.mul
        elif isinstance(node, ast.Div):
            return operator.truediv  # tvn:  WARNING: might not be accurate
        elif isinstance(node, ast.FloorDiv):
            return operator.truediv  # tvn:  WARNING: might not be accurate
        elif isinstance(node, ast.Mod):
            return operator.mod
        elif isinstance(node, ast.Pow):
            return operator.pow
        elif isinstance(node, ast.Sub):
            return operator.sub
        elif isinstance(node, ast.USub):
            return operator.neg
        elif isinstance(node, ast.Eq):
            return operator.eq
        elif isinstance(node, ast.NotEq):
            return operator.ne
        elif isinstance(node, ast.Lt):
            return operator.lt
        elif isinstance(node, ast.LtE):
            return operator.le
        elif isinstance(node, ast.Gt):
            return operator.gt
        elif isinstance(node, ast.GtE):
            return operator.ge

        else:
            raise NotImplementedError(ast.dump(node))

    @staticmethod
    @functools.cache
    def simplify(f: z3.ExprRef) -> z3.ExprRef:
        assert z3.is_expr(f), f
        simpl = z3.Tactic("ctx-solver-simplify")
        simpl = z3.TryFor(simpl, config.z3_timeout)
        try:
            f = simpl(f).as_expr()
        except z3.Z3Exception:
            pass
        return f

    @staticmethod
    def to_smt2_str(
        f: z3.ExprRef, status: str = "unknown", name: str = "benchmark", logic: str = ""
    ) -> str:
        v = (z3.Ast * 0)()
        s = z3.Z3_benchmark_to_smtlib_string(
            f.ctx_ref(), name, logic, status, "", 0, v, f.as_ast()
        )
        return s

    @classmethod
    def from_smt2_str(cls: Type[Z3], s: str) -> z3.ExprRef:
        assertions = z3.parse_smt2_string(s)
        expr = cls.zTrue if not assertions else assertions[0]
        assert z3.is_expr(expr), expr
        return expr

    @classmethod
    def model_str(
        cls: Type[Z3], m: Union[List, z3.ModelRef], as_str: bool = True
    ) -> Union[List, str]:
        """
        Returned a 'sorted' model by its keys.
        e.g. if the model is y = 3 , x = 10, then the result is
        x = 10, y = 3

        EXAMPLES:
        see doctest examples from function prove()

        """
        assert m is None or m == [] or isinstance(m, z3.ModelRef)

        if m:
            vs = [(v, m[v]) for v in m]
            vs = sorted(vs, key=lambda a: str(a[0]))
            if as_str:
                return "\n".join(f"{k} = {v}" for (k, v) in vs)
            else:
                return vs
        else:
            return str(m) if as_str else m

    @classmethod
    def check_sat(cls: Type[Z3], exprs: list[sympy.Expr]) -> bool:
        assert exprs, "exprs is empty"

        # Create a Solver instance
        solver = z3.Solver()
        solver.set(unsat_core=True)
        solver.set(timeout=config.z3_timeout * 1000)

        counter = 0
        equations = {}
        for e in exprs:
            key = f"eqt_{counter}"
            equations[key] = e
            z3_expr = cls.parse(str(e))
            solver.assert_and_track(z3_expr == 0, key)
            counter += 1

        result = solver.check()
        if result == z3.unsat:
            unsat_equations = []
            for core in solver.unsat_core():
                unsat_equations.append(equations[str(core)])
            return z3.unsat, unsat_equations
        elif result == z3.sat:
            return z3.sat, solver.model()
        else:
            return z3.unknown, None

    @classmethod
    def _get_expr(cls: Type[Z3], p: Any) -> z3.ExprRef:
        assert isinstance(p, sympy.Expr), p
        return cls.parse(str(p)) == 0

    @classmethod
    def _simplify_slow(cls, ps: list[sympy.Expr], others: list[sympy.Expr], msg):
        """
        Simplify given properties ps using properties in both ps and others
        e.g., remove g if  ps_exclude_g & others => g
        Note: this task is slow
        """

        if len(ps) < 2:
            log.debug(f"no need to simplify {msg} {len(ps)}")
            return ps

        # NOTE: create a dictionary to map a new z3 variable name to a sympy variable
        counter = 0
        table = {}
        subs = {}
        for p in ps + others:
            for v in p.free_symbols:
                if v not in subs:
                    symbol = sympy.Symbol(f"uk_{counter}")
                    subs[v] = symbol
                    table[symbol] = v
                    counter += 1

        # NOTE: substitute sympy variables with new symbols
        ps = [p.subs(subs) for p in ps]
        others = [p.subs(subs) for p in others]

        st = time()
        conj = [cls._get_expr(p) for p in others] if others else []
        ps_exprs = [cls._get_expr(p) for p in ps]

        def _imply(js, i):
            iexpr = ps_exprs[i]

            # assert iexpr.decl().kind() != z3.Z3_OP_EQ, iexpr
            jexprs = [ps_exprs[j] for j in js]
            ret = Z3._imply(conj + jexprs, iexpr, is_conj=True)
            # print('{} => {} ret{}'.format(jexprs, iexpr, ret))
            return ret

        results = Symbolic.simplify_idxs(list(range(len(ps))), _imply)
        results = [ps[i] for i in results]

        Symbolic.show_removed(
            f"_simplify_slow {msg}", len(ps), len(results), time() - st
        )

        # NOTE: substitute back
        results = [r.subs(table) for r in results]

        return results


if __name__ == "__main__":

    x, y, z = sympy.symbols("x y z")
    print(Z3.check_sat([x * x - 4, x - 3, y - 2, z - 3]))

    x, y, a, b, q, r = sympy.symbols("x y a, b, q, r")
    exprs = [a * y - b, q * y + r - x, -a * r + a * x - b * q]

    for e in exprs:
        print(f"{e} -> {Z3.parse(str(e)) == 0}")

    results = Z3._simplify_slow(exprs, [], "test")
    for r in results:
        print(r)

    print()
