from escnn.kernels.basis import (
    EmptyBasisException,
    KernelBasis,
    AdjointBasis,
    UnionBasis,
)

from escnn.kernels.harmonic_polynomial_r3 import HarmonicPolynomialR3Generator

from escnn.kernels.steerable_filters_basis import SteerableFiltersBasis, PointBasis
from escnn.kernels.polar_basis import (
    GaussianRadialProfile,
    SphericalShellsBasis,
    CircularShellsBasis,
)
from escnn.kernels.sparse_basis import (
    SparseOrbitBasis,
    SparseOrbitBasisWithIcosahedralSymmetry,
)

from escnn.kernels.steerable_basis import SteerableKernelBasis, IrrepBasis

from .learnable_basis import PrelimLearnableWignerEckartBasis
from .kernels.wignereckart_solver import LearnableWignerEckartBasis
from escnn.kernels.wignereckart_solver import (
    WignerEckartBasis,
    RestrictedWignerEckartBasis,
)

from escnn.kernels.r2 import *
from escnn.kernels.r3 import *


import escnn.group


def kernels_on_point(
    in_repr: escnn.group.Representation,
    out_repr: escnn.group.Representation,
    layer_id,
    prelim=False,
    **kwargs,
) -> KernelBasis:
    r"""

    Generates kernels for single points (linear layer).

    Args:
        in_repr (Representation): the representation specifying the transformation of the input feature field
        out_repr (Representation): the representation specifying the transformation of the output feature field
        layer_id (int): id for the layer. Layers with the same id
                    share the degree of equivariance
        prelim (bool, default=True): determines whether to use the preliminary approach
                    of learning the degree of equivariance or the Fourier-based approach.
        kwargs: additional arguments

    """
    assert in_repr.group == out_repr.group

    group = in_repr.group
    # Add keyword to Steerablekernelsbasis for the learnablebasis to
    if not prelim:
        mapping_kwargs = dict(layer_id=layer_id)
        mapping_kwargs["mapping_kwargs"] = kwargs
        basis = SteerableKernelBasis(
            PointBasis(group),
            in_repr,
            out_repr,
            LearnableWignerEckartBasis,
            **mapping_kwargs,
        )
    else:
        basis = SteerableKernelBasis(
            PointBasis(group),
            in_repr,
            out_repr,
            PrelimLearnableWignerEckartBasis,
            layer_id=layer_id,
        )
    return basis
