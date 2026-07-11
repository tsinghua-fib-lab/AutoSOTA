from ntkunlimited.recursions.symbolic_gaussian_expectation import (
    ElementwiseFunction,
    ConstIdx,
)
from ntkunlimited.recursions.symbolic_inverse_operations import (
    InverseMatrixSymbol,
    SymmetricMatrixSymbol,
)
from sympy.stats import MultivariateNormal
from sympy import MatrixSymbol, symbols, Function, erf, sqrt, diff
from sympy.tensor.array.expressions import ArraySymbol


# Common symbols needed for the Gaussian expectations
sig = ElementwiseFunction("sig")
dim = 4

K = SymmetricMatrixSymbol("K", dim)
G1 = MatrixSymbol("G1", dim, dim)
H = MatrixSymbol("H", dim, dim)
H1 = MatrixSymbol("H1", dim, dim)
Kinv = InverseMatrixSymbol(K)

V = ArraySymbol("V", (dim, dim, dim, dim))
F = ArraySymbol("F", (dim, dim, dim, dim))
D = ArraySymbol("D", (dim, dim, dim, dim))
A = ArraySymbol("A", (dim, dim, dim, dim))
B = ArraySymbol("B", (dim, dim, dim, dim))

z = MultivariateNormal(
    "z",
    [
        0,
    ]
    * dim,
    K,
)

a1 = ConstIdx("a1", dim)
a2 = ConstIdx("a2", dim)
a3 = ConstIdx("a3", dim)
a4 = ConstIdx("a4", dim)

b1 = ConstIdx("b1", dim)
b2 = ConstIdx("b2", dim)
b3 = ConstIdx("b3", dim)
b4 = ConstIdx("b4", dim)

g1 = ConstIdx("g1", dim)
g2 = ConstIdx("g2", dim)
g3 = ConstIdx("g3", dim)
g4 = ConstIdx("g4", dim)

h1 = ConstIdx("h1", dim)
h2 = ConstIdx("h2", dim)
h3 = ConstIdx("h3", dim)
h4 = ConstIdx("h4", dim)


n_l, n_lm1 = symbols("n_l n_lm1", integer=True)
C_W = symbols("C_W")


def Omega(i1, i2):
    return sig(z[i1]) * sig(z[i2]) + C_W * H[i1, i2] * diff(sig(z[i1]), z[i1]) * diff(
        sig(z[i2]), z[i2]
    )


class Gelu(Function):
    @classmethod
    def eval(cls, arg):
        # always return the closed form
        return arg / 2 * (1 + erf(arg / sqrt(2)))
