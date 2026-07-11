import inspect
import hashlib
import pickle
import warnings

import numpy as np
import sympy
from ntkunlimited.recursions.numerical_integration import comp_fn_normal_expec_all
from ntkunlimited.recursions.symbolic_inverse_operations import inverse_contract_simp
from ntkunlimited.recursions.symbolic_gaussian_expectation import (
    ConstIdx,
    extract_covariance_matrix,
    extract_gaussian_variable,
    ElementwiseFunction,
)
from ntkunlimited.recursions.symbolic_inverse_operations import InverseMatrixSymbol
from ntkunlimited.recursions.symmetric_tensor import TensorSymmetry
from sympy import (
    Idx,
    IndexedBase,
    MatrixSymbol,
    lambdify,
    symbols,
    sympify,
    expand,
)
from sympy.printing.numpy import SciPyPrinter
from sympy.printing.pycode import SymPyPrinter
from sympy.stats import MultivariateNormal
from sympy.tensor.indexed import Indexed
from sympy.utilities.iterables import permutations
from ntkunlimited.recursions.config import cache_dir
from functools import lru_cache

import ntkunlimited.utils
from ntkunlimited.utils import cacheable_function
import sys


@lru_cache(maxsize=1024)
def hash_expr(expr):
    """Creates a hash from a sympy expression that is persistent over different runtimes.

    Args:
        expr (sympy expression): The sympy expression to hash.

    Returns:
        str: The hash of the expression.
    """
    h = hashlib.md5()
    expr_str = sympy.srepr(expr)
    h.update(expr_str.encode("utf-8"))
    return h.hexdigest()


class GaussExpecNumeric:
    """Class to compute Gaussian expectations numerically from symbolic expressions with caching.

    The idea is to create a dictionary with all computed gaussian expectations, indexed by the hash
    of the corresponding symbolic expression that defines the integrand. The method also analyzes
    the symmetry of the integrand to avoid redundant computations and reduce the effective dimension
    of the integrand.
    """

    def __init__(self, cub_conf):
        self.cub_conf = cub_conf
        self.cached_expecs = {}
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.filename = cache_dir / "gaussexpec_numeric_cache.pkl"

    def __call__(
        self,
        expr,
        z,
        cov,
        idx_map,
        sym=None,
        idx_order_str=None,
        tensor_dim=None,
        data_size=None,
    ):
        return self.compute_expec(
            expr, z, cov, idx_map, sym, idx_order_str, tensor_dim, data_size
        )

    def save_cache(self):
        with open(self.filename, "wb") as f:
            setup = {
                "cub_conf": self.cub_conf,
                "cached_expecs": self.cached_expecs,
            }
            pickle.dump(setup, f)

    def load_cache(self):
        try:
            with open(self.filename, "rb") as f:
                setup = pickle.load(f)
                cub_conf = setup["cub_conf"]
                if cub_conf != self.cub_conf:
                    raise ValueError(
                        "Cubature configuration of does not match the one used in the cache."
                    )
                self.cached_expecs = setup["cached_expecs"]
        except FileNotFoundError:
            print(
                f"Cache file {self.filename} not found. Starting with an empty cache."
            )

    def clear_cache(self):
        self.cached_expecs = {}
        if self.filename.exists():
            self.filename.unlink()
            print(f"Cache file {self.filename} deleted.")

    def compute_expec(
        self, expr, z, cov, idx_map, sym, idx_order_str, tensor_dim, data_size
    ):
        # def compute_expec(self, expr, z, cov, sym, tensor_dim, data_size):
        """Compute the Gaussian expectation of a symbolic expression numerically.

        Args:
            expr (sympy expression): The symbolic expression representing the integrand.
            z (sympy symbol): The Gaussian random variable over which the expectation is taken.
            cov (sympy matrix): The covariance matrix of the Gaussian variable.
            idx_map (dict): A mapping from abstract sympy indices to their concrete integer values.
            sym (list of tuples, optional): List of permutation symmetries of the tensor. Defaults to None.
            tensor_dim (int, optional): The dimension of the tensor. Defaults to None.
            data_size (int, optional): The size of the data. Defaults to None.

        Returns:
            float: The computed Gaussian expectation.
        """
        cov_id = ntkunlimited.utils.hash_array(cov)
        expr_id = hash_expr(expr)
        expec = None
        # print(f"Integrand id: {expr_id}\nExpression:\n{expr}\n")
        if cov_id in self.cached_expecs:
            if expr_id in self.cached_expecs[cov_id]:
                expec = self.cached_expecs[cov_id][expr_id]
                # print("Using cached value.\n\n")
                # print(f"New integrand id: {expr_id}\nExpression:\n{expr}\n")
        else:
            self.cached_expecs[cov_id] = {}

        tensor_sym = TensorSymmetry(sym, tensor_dim=tensor_dim, data_size=data_size)
        # idxs = sorted(expr.atoms(ConstIdx), key=lambda x: str(x))
        idx_order = []
        for idx_name in idx_order_str:
            for k in idx_map.keys():
                if k.name == idx_name:
                    idx_order.append(k)
                    break

        if expec is None:
            expr = sympify(expr)

            # If we already have the cached value, jump to the end
            fn = lambdify([z] + idx_order, expr, modules=["scipy"])
            # print(f"Source of numerical fn for a particular integrand:\n{inspect.getsource(fn)}")
            expec = comp_fn_normal_expec_all(
                fn,
                cov,
                tensor_sym,
                self.cub_conf,
            )
            self.cached_expecs[cov_id][expr_id] = expec

        entry_comp = tuple([idx_map[idx] for idx in idx_order])
        return expec[entry_comp]


@lru_cache(maxsize=1024)
def _is_equal(expr1, expr2):
    return expr1.equals(expr2)


@lru_cache(maxsize=1024)
def _check_integrand_symmetry(expr, z):
    x_ids = expr.atoms(Indexed)
    ids = [x_idx.indices[0] for x_idx in x_ids if x_idx.base == z]
    ids = sorted(ids, key=lambda x: str(x))
    n = len(ids)
    dim = int(ids[0].upper - ids[0].lower + 1)
    if n > 4:
        warnings.warn(
            f"Expression has {n} indices. This might lead to a combinatorial explosion "
            "in the number of permutations."
        )
    # All permutations
    syms = []
    for perm_ids in permutations(range(n)):
        perm = [ids[j] for j in perm_ids]
        # Dummy Idx to avoid substitution collisions
        dummy = [ConstIdx(f"_d{i.name}", dim) for i in ids]

        to_dummy = {orig: d for orig, d in zip(ids, dummy)}
        to_perm = {d: new for d, new in zip(dummy, perm)}

        # Two-step substitution
        swapped_expr = expr.xreplace(to_dummy).xreplace(to_perm)
        # Check if the expression is equal to the original expression
        if _is_equal(expr, swapped_expr):
            syms.append(perm_ids)

    return syms, ids, n, dim


# For the same underlying expression we can reuse the same numerical function. We just need to keep
# track of the indices.
class IndexStandardizer:
    def __init__(self):
        self.known_standardized_expr = []

    def _replace_idx_with_base(self, expr):
        replacements = {indexed: indexed.base for indexed in expr.atoms(Indexed)}
        return expr.xreplace(replacements), replacements

    def _replace_indices(self, expr, idx_mapping):
        replaced_expr = expr.xreplace(idx_mapping)
        return replaced_expr

    def _standardize_indices(self, expr):
        idxs = list(expr.atoms(ConstIdx))
        # TODO: Try to find a better way of standardizing. This is of barely any use
        idxs = sorted(idxs, key=lambda x: str(x))
        std_mapping = {k: ConstIdx(f"i{k.name[-1]}", k.upper + 1) for k in idxs}
        replaced_expr = self._replace_indices(expr, std_mapping)
        return replaced_expr, std_mapping

    def __call__(self, expr):
        std_expr, std_mapping = self._standardize_indices(expr)

        if std_expr not in self.known_standardized_expr:
            self.known_standardized_expr.append(std_expr)
        return std_expr, std_mapping


class UnconcretePrinter(SymPyPrinter):
    """A printer that leaves expressions symbolic and handles custom types that we implemented.

    The idea is to use this when turning symbolic expressions to numeric functions, where some
    subroutines like the gaussian numerical expectation method still need the abstract symbolic
    integrands.
    """

    def _print_Idx(self, expr):
        return f"ConstIdx('{expr.name}', {expr.upper + 1})"

    def _print_Indexed(self, expr):
        # sympy.stats.joint_rv.JointRandomSymbol
        # (z, JointPSpace(z, MultivariateNormalDistribution(Matrix([\n[0],\n[0],\n[0],\n[0]]), K)))
        return f"IndexedBase({self._print(expr.base)})[{self._print(expr.indices[0])}]"

    def _print_symbolic_dict(self, d):
        return (
            "{"
            + ", ".join(f"{self._print(k)}: {self._print(v)}" for k, v in d.items())
            + "}"
        )


class ExtendedSciPyPrinter(SciPyPrinter):
    """This printer prints most expressions in scipy terms, but leaves the GaussExpec arguments
    symbolic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unconcrete_printer = UnconcretePrinter()
        self.index_standardizer = IndexStandardizer()

    def _print_InverseMatrixSymbol(self, expr):
        # name = f"{expr.args[0].name}inv"
        name = f"numpy.linalg.inv({self._print(expr.args[0])})"
        return name

    def _print_GaussExpec(self, expr):
        rv = expr.args[0]
        z = extract_gaussian_variable(rv)
        std_expr, idx_mapping = self.index_standardizer(rv)
        syms, idx_order, tensor_dim, data_size = _check_integrand_symmetry(std_expr, z)
        cov = extract_covariance_matrix(z)
        if tensor_dim > 9:
            raise RuntimeError("more than 9 indices are currently not supported.")
        idx_reverse_map = {v: k.name for k, v in idx_mapping.items()}
        idx_order_str = [idx.name for idx in idx_order]
        str_repr = (
            f"gaussexpec_numeric({self.unconcrete_printer._print(std_expr)},"
            f"IndexedBase('{z.name}'), "
            f"{self._print(cov)}, "
            f"{self.unconcrete_printer._print_symbolic_dict(idx_reverse_map)}, "
            f"sym={syms}, idx_order_str={idx_order_str}, tensor_dim={tensor_dim}, data_size={data_size})"
        )

        return str_repr


def simplify_gaussian_expec(expr, act_fn, subs={}):
    """Apply symbolic simplification (mostly integration by parts) and optional substitutions."""
    sig = ElementwiseFunction("sig")
    print("Simplifying expression ...")
    reduced_expr = expr.expand(ibp=True)
    reduced_expr, _ = inverse_contract_simp(reduced_expr)
    reduced_expr = reduced_expr.doit()
    reduced_expr = expand(reduced_expr)
    # print(f"After ibp and contractions:\n{reduced_expr}\n")
    reduced_expr = reduced_expr.subs(sig, act_fn)
    for k, v in subs.items():
        reduced_expr = reduced_expr.subs(k, v)

    reduced_expr = reduced_expr.doit()
    # print(f"Simplified expression:\n{reduced_expr}\n")
    return reduced_expr


def make_numeric(expr, args, custom_modules):
    print("Creating numerical function ...")
    f = lambdify(
        args,
        expr,
        modules=[custom_modules, "scipy"],
        printer=ExtendedSciPyPrinter,
    )
    print("Numerical function created.")
    # print("Source of f:")
    # print(inspect.getsource(f))
    return f


def make_efficient_numeric(
    expr, act_fn, args, cache_name, args_hasher, gaussexpec_numeric, subs={}
):
    custom_modules = custom_gaussexpec | {"gaussexpec_numeric": gaussexpec_numeric}

    @cacheable_function(cache_name, args_hasher)
    def simplify_and_numericalize(expr, act_fn, args, subs):
        reduced_expr = simplify_gaussian_expec(expr, act_fn, subs)
        f = make_numeric(reduced_expr, args, custom_modules)
        return f

    f = simplify_and_numericalize(expr, act_fn, args, subs)
    # print(f"Source of f:\n{inspect.getsource(f)}")

    # Need to update the references in global in case the cached function was loaded
    f.__globals__.update(custom_modules)

    simplify_and_numericalize.save_cache()
    return f


# REMARK: This can probably be integrated inside the printers somehow
custom_gaussexpec = {
    "IndexedBase": IndexedBase,
    "ImmutableDenseMatrix": sympy.matrices.immutable.ImmutableDenseMatrix,
    "sympy": sympy,
    "ConstIdx": ConstIdx,
}

# This is a simple test case
if __name__ == "__main__":
    dim = 4

    K = MatrixSymbol("K", dim, dim)
    Kinv = InverseMatrixSymbol(K)
    z = MultivariateNormal(
        "z",
        [
            0,
        ]
        * dim,
        K,
    )

    # JointRandomVariable has no proper iterator implementation, this is a quick fix
    z._iterable = False

    a1 = Idx("a1", dim)
    a2 = Idx("a2", dim)

    x = symbols("x")
    # expr = GaussExpec(z[a1]) + x
    expr = Kinv[a1, a2]
    K_inv_replaced = MatrixSymbol("K_inv_replaced", dim, dim)

    expr = expr.subs(Kinv, K_inv_replaced)

    f = lambdify(
        [K_inv_replaced, a1, a2],
        expr,
        modules=[custom_gaussexpec, "scipy"],
        printer=ExtendedSciPyPrinter,
        dummify=False,
        use_imps=False,
    )
    print(f"Source of f:\n{inspect.getsource(f)}")

    print(f(np.random.rand(4, 4), 0, 0))
    print("Done.")
