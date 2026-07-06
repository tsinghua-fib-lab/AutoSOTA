"""multivariate_polynomial_basis.py: Implements multivariate polynomial approximators.
"""
import copy
import random
import itertools
import numpy as np
import torch
from algorithms import rl_approximator_interfaces as rl_approximator


__author__ = "Stefan Huber and Hannes Unger"
__version__ = "0.2.0"
__email__ = "{firstname.lastname}@fh-salzburg.ac.at"


INIT_WEIGHT = 1e-6


def univar_power(d):
    """Return uni-variate monomial of degree d"""
    return np.poly1d([1, 0]) ** d


def univar_chebychev(d):
    """Return uni-variate Chebyshev polynomial of degree d"""
    if d == 0:
        return np.poly1d([1])
    elif d == 1:
        return np.poly1d([1, 0])
    else:
        return 2 * univar_chebychev(1) * univar_chebychev(d - 1) - univar_chebychev(d - 2)


def multivarpoly_by_univar(univarbasis, ds):
    """Return a multi-variate polynomial as a product of
    uni-variate polynomial in each dimension using a given
    basis of uni-variate polynomials."""

    def p(xs):
        """A multi-variate polynomial as a product of uni-variate polynomials."""
        return np.prod([univarbasis(d)(xs[i]) for i, d in enumerate(ds)])

    return p


def multivar_power(ds):
    """Multi-variate monomial with given sequence ds of degrees (exponents)."""
    return multivarpoly_by_univar(univar_power, ds)


def multivar_chebyshev(ds):
    """Multi-variate Chebyshev polynomial with given sequence ds of degrees."""
    return multivarpoly_by_univar(univar_chebychev, ds)


def bivar_basis(multivarbasis, d):
    """Basis for space of bi-variate polynomials of max-degree d using
    the function multivarbasis to generate basis elements."""
    return [multivarbasis([d1, d2]) for d1 in range(d + 1) for d2 in range(d + 1)]


def bivar_power_basis(d):
    """Power basis for space of bi-variate polynomials of max-degree d."""
    return bivar_basis(multivar_power, d)


def bivar_chebyshev_basis(d):
    """Chebyshev basis for space of bi-variate polynomials of max-degree d."""
    return bivar_basis(multivar_chebyshev, d)


def nvar_basis(multivarbasis, n, d):
    """Basis for space of n-variate polynomials of max-degree d using
    the function multivarbasis to generate basis elements."""
    #return [multivarbasis([d1, d2]) for d1 in range(d + 1) for d2 in range(d + 1)]
    # Generate all combinations of `d` indices ranging from 0 to d
    combinations = itertools.product(range(d + 1), repeat=n)
    # Apply `multivarbasis` to each combination
    return [multivarbasis(list(comb)) for comb in combinations]


def nvar_power_basis(n, d):
    """Power basis for space of n-variate polynomials of max-degree d."""
    return nvar_basis(multivar_power, n, d)


def nvar_chebyshev_basis(n, d):
    """Power basis for space of n-variate polynomials of max-degree d."""
    return nvar_basis(multivar_chebyshev, n, d)


def function_approx_coeffs(basis, ps, fs):
    """Return sequence of coefficients \alpha_j such that \sum_j \alpha_j basis[j]
    approximates the function with values fs[i] at locations ps[i] according
    least-square error."""

    evals = [[b(p) for b in basis] for p in ps]
    evalspinv = np.linalg.pinv(evals)

    return evalspinv @ fs


def function_approx(basis, ps, fs):
    coeffs = function_approx_coeffs(basis, ps, fs)

    def f(*x):
        return sum([c * b(x) for c, b in zip(coeffs, basis)])

    return f


def l2_difference(f, g, ps):
    ds = np.array([f(*p) - g(*p) for p in ps])
    return sum(ds**2) / len(ps)


class MultiVarPoly(rl_approximator.Approximator):
    def __init__(self, dim=2, degree=2, basis='chebyshev', initialization='constant', flat_init_offset=0.2, coeffs=None):
        self.dim = dim
        self.degree = degree
        if basis == 'power':
            self.model = nvar_power_basis(dim, degree)
        if basis == 'chebyshev':
            self.model = nvar_chebyshev_basis(dim, degree)

        if coeffs is not None:
            self.coeffs = torch.tensor(copy.deepcopy(coeffs), requires_grad=True)
        else:
            if initialization == 'constant':
                self.coeffs = torch.tensor([INIT_WEIGHT for _ in range(len(self.model))], requires_grad=True)
            elif initialization == 'flat':
                self.coeffs = torch.tensor([0.0 for _ in range(len(self.model))], requires_grad=True)
                self.coeffs.data[0] = (flat_init_offset)
            else:
                self.coeffs = torch.tensor([random.uniform(0, INIT_WEIGHT) for _ in range(len(self.model))], requires_grad=True)

    def evaluate_point(self, x):
        return sum([c * b(x) for c, b in zip(self.coeffs, self.model)])

    def evaluate(self, *data):
        #return np.array([[self.evaluate_point([x,y]) for x in data_x] for y in data_y]) # for 2d data
        grid = np.meshgrid(*data, indexing='ij')
        points = np.stack(grid, axis=-1)
        return torch.from_numpy(np.apply_along_axis(self.evaluate_point, -1, points))

    def evaluate_basis_vectors_at(self, x):
        #return tf.constant([b(x) for b in self.model], dtype=tf.float32)
        return [b(x) for b in self.model]

    def fit(self, coord_points, target_values):
        self.coeffs = function_approx_coeffs(self.model, coord_points, target_values)