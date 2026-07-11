from sympy import Function
from sympy import Idx, S, symbols
from sympy.concrete.summations import summation
from sympy.core.add import Add
from sympy.core.function import expand as _expand
from sympy.core.mul import Mul
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.stats import Expectation
from sympy.stats.joint_rv import JointRandomSymbol
from sympy.stats.joint_rv_types import MultivariateNormalDistribution
from sympy.stats.rv import is_random
from sympy.tensor.indexed import Indexed


class ConstIdx(Idx):
    def _eval_derivative(self, x):
        return S.Zero


class GaussExpec(Expectation):
    def __new__(cls, expr, condition=None, **kwargs):
        obj = super().__new__(cls, expr, condition, **kwargs)
        return obj

    def expand(self, **hints):
        hints = hints | {"kronecker": True}
        self = self._eval_expand_basic(**hints)
        for hint, value in hints.items():
            if hasattr(self, f"_eval_expand_{hint}") and value:
                self = getattr(self, f"_eval_expand_{hint}")(**hints)
        return self

    def _eval_expand_basic(self, **hints):
        expr = self.args[0]
        condition = self._condition

        if not is_random(expr):
            return expr

        if isinstance(expr, Add):
            return Add.fromiter(
                GaussExpec(a, condition=condition).expand(**hints) for a in expr.args
            )

        expand_expr = _expand(expr)
        if isinstance(expand_expr, Add):
            return Add.fromiter(
                GaussExpec(a, condition=condition).expand(**hints)
                for a in expand_expr.args
            )

        if isinstance(expr, Mul):
            rv = []
            nonrv = []
            for a in expr.args:
                if is_random(a):
                    rv.append(a)
                else:
                    nonrv.append(a)
            pure_RV_expec = GaussExpec(
                Mul.fromiter(rv), condition=condition
            )  # .expand(**hints)

            if not len(nonrv) == 0:
                return (Mul.fromiter(nonrv) * pure_RV_expec).expand(**hints)

            return pure_RV_expec
        return self

    # Pull out Kronecker deltas from Gaussian expectations
    def _eval_expand_kronecker(self, **hints):
        expr = self.args[0]
        if isinstance(expr, Mul):
            kroneckers = []
            rest = []
            for a in expr.args:
                if isinstance(a, KroneckerDelta):
                    # TODO: Allow for arbitrary dimensions
                    a = KroneckerDelta(*a.indices, (0, 3))
                    kroneckers.append(a)
                else:
                    rest.append(a)
            return Mul.fromiter(kroneckers) * GaussExpec(
                Mul.fromiter(rest), condition=self._condition
            )
        elif isinstance(expr, Add):
            return Add.fromiter(
                GaussExpec(a, condition=self._condition) for a in expr.args
            )

        return self

    def doit(self, **hints):
        deep = hints.get("deep", True)
        if deep:
            expr = self.args[0]
            expr = expr.doit(**hints)
            return GaussExpec(expr, condition=self._condition)
        return self

    # Use integration by parts (IBP) to reduce higher-dimensional Gaussian expectations to
    # lower-dimensional ones by trading with derivatives
    def _eval_expand_ibp(self, **hints):
        expr = self.args[0]
        condition = self._condition
        # Iterate through the factors to find a pure Gaussian variable that will be used for IBP
        if isinstance(expr, Mul):
            for i, a in enumerate(expr.args):
                if isinstance(a, Indexed):
                    ind = a.args[1]
                    vec = a.args[0]
                    if isinstance(vec, JointRandomSymbol):
                        dist = vec.args[1].args[1]
                        if isinstance(dist, MultivariateNormalDistribution):
                            mu = dist.args[0]
                            dim = mu.rows
                            if not mu.is_zero_matrix:
                                raise NotImplementedError(
                                    "IBP of non-zero mean multivariate normal distributions are not supported yet."
                                )
                            cov = dist.args[1]
                            rest = Mul.fromiter(expr.args[:i] + expr.args[i + 1 :])
                            j = symbols("j", integer=True)

                            rest_deriv = rest.diff(vec[j])

                            # If the derivatives yield the same Gaussian variable as a factor, we will never
                            # finish, thus we stop here
                            if isinstance(rest_deriv, Add):
                                for summand in rest_deriv.args:
                                    _, factors = summand.as_coeff_mul()
                                    if vec[ind] in factors:
                                        return self
                            if isinstance(rest_deriv, Mul):
                                if vec in rest_deriv.args:
                                    return self

                            terms = summation(cov[ind, j] * rest_deriv, (j, 0, dim - 1))
                            expanded_expec = GaussExpec(
                                terms, condition=condition
                            ).expand(**hints)

                            return expanded_expec

        if isinstance(expr, Add):
            return Add.fromiter(GaussExpec(a, condition=condition) for a in expr.args)
        return self


def _get_indices(expr):
    if isinstance(expr, Add) or isinstance(expr, Mul):
        indices = set()
        for a in expr.args:
            indices.update(_get_indices(a))
        return indices

    if isinstance(expr, Function):
        input = expr.args[0]
        return _get_indices(input)
    if isinstance(expr, Indexed):
        return {expr.args[1]}

    return set()


def extract_gaussian_variable(expr):
    z = None
    if isinstance(expr, JointRandomSymbol):
        dist = expr.args[1].args[1]
        if isinstance(dist, MultivariateNormalDistribution):
            return expr

    if isinstance(expr, Mul):
        for a in expr.args:
            if is_random(a):
                z = extract_gaussian_variable(a)
                break

    if isinstance(expr, Indexed):
        vec = expr.args[0]
        z = extract_gaussian_variable(vec)

    if hasattr(expr, "args"):
        for a in expr.args:
            if is_random(a):
                z = extract_gaussian_variable(a)

    if z is not None:
        return z

    raise NotImplementedError(
        "The gaussian variable extraction is not implemented for this type of expression."
    )


def extract_covariance_matrix(z):
    dist = z.args[1].args[1]
    return dist.args[1]


class ElementwiseFunction(Function):
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(Function, *args, **kwargs)
        return obj
