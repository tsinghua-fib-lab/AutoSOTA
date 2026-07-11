from sympy import MatrixSymbol, symbols, Sum, KroneckerDelta
from sympy.matrices.expressions.matexpr import MatrixExpr, MatrixElement
from sympy.core import Basic
from sympy.core.sympify import sympify
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.sympify import _sympify
from sympy import expand


class SymmetricMatrixSymbol(MatrixSymbol):
    """A symbolic matrix symbol explicitly declared symmetric."""

    def __new__(cls, name, n):
        n = _sympify(n)
        cls._check_dim(n)

        if isinstance(name, str):
            name = sympify(name)
        obj = Basic.__new__(cls, name, n)
        return obj

    @property
    def is_symmetric(self):
        return True

    @property
    def shape(self):
        return (self.args[1],) * 2


def is_explicitly_symmetric(M):
    # True only if SymPy *knows* it's symmetric
    return getattr(M, "is_symmetric", None) is True


class InverseMatrixSymbol(MatrixExpr):
    """Symbolic inverse of a MatrixSymbol."""

    is_commutative = False

    def __new__(cls, A):
        A = sympify(A)
        if not isinstance(A, MatrixSymbol):
            raise TypeError("InverseMatrixSymbol only works with MatrixSymbol.")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return MatrixExpr.__new__(cls, A)

    @property
    def is_symmetric(self):
        return A.is_symmetric

    @property
    def arg(self):
        return self.args[0]

    @property
    def shape(self):
        return self.arg.shape

    def _eval_inverse(self):
        return self.arg

    def _entry(self, i, j, **kwargs):
        return MatrixElement(self, i, j)

    def __str__(self):
        return f"{self.arg}inv"

    def _sympystr(self, printer):
        return f"{printer._print(self.arg)}^(-1)"


def inverse_contract_simp(expr, sum_idxs=[]):
    expr = sympify(expr)
    expr = expand(expr)

    if not expr.has(InverseMatrixSymbol):
        return expr, sum_idxs

    if isinstance(expr, Sum):
        sum_ranges = expr.args[1:]
        sum_idxs += [i.args[0] for i in sum_ranges]
        summands, new_sum_idxs = inverse_contract_simp(expr.args[0], sum_idxs)
        remaining_sum_ranges = [
            sum_range for sum_range in sum_ranges if sum_range.args[0] in new_sum_idxs
        ]
        if len(remaining_sum_ranges) == 0:
            expr = summands
        else:
            expr = Sum(summands, *remaining_sum_ranges)

    if expr.is_Add:
        expr = Add(*[inverse_contract_simp(a, sum_idxs)[0] for a in expr.args])

    if expr.is_Mul:
        # TODO: Can currently only contract matrix pairs at the same level. Factors spread over
        # e.g. nested sums are not recognized right now.
        factors = expr.args
        inv_matrix_elms = []
        matrix_elms = []
        simplified_factors = []
        for f in factors:
            f, sum_idxs = inverse_contract_simp(f, sum_idxs)
            simplified_factors.append(f)
            if isinstance(f, MatrixElement):
                if isinstance(f.parent, InverseMatrixSymbol):
                    inv_matrix_elms.append(f)
                else:
                    matrix_elms.append(f)
        expr = Mul(*simplified_factors)

        for inv_elm in inv_matrix_elms:
            # Check if the inverse matrix element is summed over

            # Find matching matrix element with the same indices
            for mat_elm in matrix_elms:
                if mat_elm.parent != inv_elm.parent.arg:
                    continue

                # Check if they share an index
                shared_indices = set(mat_elm.indices) & set(inv_elm.indices)
                if len(shared_indices) == 0:
                    continue

                if not is_explicitly_symmetric(mat_elm.parent):
                    # Check that it is actually a matrix multiplication with a matrix and its inverse
                    if not (
                        mat_elm.indices[0] == inv_elm.indices[1]
                        or (mat_elm.indices[1] == inv_elm.indices[0])
                    ):
                        continue

                # Check if the contraction index of the matrices is actually summed over
                contract_indices = shared_indices & set(sum_idxs)

                # We take the first available contraction, the other contraction indices are left
                # for later
                cont_idx = contract_indices.pop()

                # Check that the rest is independent of the summation index
                remaining_factors = [f for f in factors if f not in (inv_elm, mat_elm)]
                if any(f.has(cont_idx) for f in remaining_factors):
                    continue

                uncontracted_indices = set(inv_elm.indices) | set(mat_elm.indices)
                uncontracted_indices.remove(cont_idx)
                uncontracted_indices = tuple(uncontracted_indices)
                if len(uncontracted_indices) == 1:
                    uncontracted_indices = 2 * uncontracted_indices
                if len(uncontracted_indices) < 1 or len(uncontracted_indices) > 2:
                    raise ValueError(
                        "Invalid number of uncontracted indices after contraction."
                    )

                # FIX: This currently assumes that the matrix is square
                dim_matrix = mat_elm.parent.shape[0]

                # Replace the matrix multiplication with KroneckerDelta
                # TODO: This currently assumes that we are summing over the whole matrix axis
                expr = expr.subs(
                    inv_elm * mat_elm, KroneckerDelta(*uncontracted_indices)
                )
                sum_idxs = [i for i in sum_idxs if i != cont_idx]

                # This one has been contracted, so we have to remove it
                matrix_elms.remove(mat_elm)
                break

    return expr, sum_idxs


# This is just a simple test case
if __name__ == "__main__":
    n = 3
    i, j, k = symbols("i j k", integer=True)
    A = SymmetricMatrixSymbol("A", n)
    Ainv = InverseMatrixSymbol(A)
    B = MatrixSymbol("B", n, n)
    c = symbols("c", real=True)

    expr = Sum(A[i, j] * Ainv[j, k] * A[k, k] + (c * B[j, j] + B[i, i]), (j, 0, n - 1))
    print("Original expression:")
    print(expr)

    simplified, _ = inverse_contract_simp(expr)
    print("\nSimplified expression:")
    print(simplified)
