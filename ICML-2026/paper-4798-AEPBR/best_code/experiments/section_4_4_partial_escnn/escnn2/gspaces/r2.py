from __future__ import annotations

from escnn import gspaces
from ..kernels.r2 import (
    kernels_O2_act_R2,
    kernels_SO2_act_R2,
    kernels_O2_subgroup_act_R2,
    kernels_SO2_subgroup_act_R2,
    KernelBasis,
)

from .utils import rotate_array_2d

from escnn.group import *

import numpy as np

from typing import Tuple, Union, Callable, List

__all__ = [
    "GSpace2D",
    #################
    "rot2dOnR2",
    "flipRot2dOnR2",
    "flip2dOnR2",
    "trivialOnR2",
]


class GSpace2D(gspaces.GSpace):
    def __init__(self, sg_id: Tuple, maximum_frequency: int = 6):
        r"""

        Describes reflectional and rotational symmetries of the plane :math:`\R^3`.

        .. note ::
            A point :math:`\bold{v} \in \R^2` is parametrized using an :math:`(X, Y)` convention,
            i.e. :math:`\bold{v} = (x, y)^T`.
            The representation :attr:`escnn.gspaces.GSpace2D.basespace_action` also assumes this convention.

            However, when working with data on a pixel grid, the usual :math:`(-Y, X)` convention is used.
            That means that, in a 4-dimensional feature tensor of shape ``(B, C, D1, D2)``, the last dimension
            is the X axis while the second last is the (inverted) Y axis.
            Note that this is consistent with 2D images, where a :math:`(-Y, X)` convention is used.

        """
        o2 = o2_group(maximum_frequency=maximum_frequency)
        _sg_id = o2._process_subgroup_id(sg_id)
        fibergroup, inclusion, restriction = o2.subgroup(_sg_id)

        # TODO - catch sg_id and build a dictionary of more meaningful names
        # use the input sg_id instead of the processed one to avoid adding the adjoint parameter unless specified
        name = f"{fibergroup}_on_R2[{sg_id}]"

        self._sg_id = _sg_id
        self._inclusion = inclusion
        self._restriction = restriction
        self._base_action = o2.irrep(1, 1).restrict(_sg_id)

        super(GSpace2D, self).__init__(fibergroup, 2, name)

    def restrict(self, id: Tuple) -> Tuple[gspaces.GSpace, Callable, Callable]:
        r"""

        Build the :class:`~escnn.group.GSpace` associated with the subgroup of the current fiber group identified by
        the input ``id``

        Args:
            id (tuple): the id of the subgroup

        Returns:
            a tuple containing

                - **gspace**: the restricted gspace

                - **back_map**: a function mapping an element of the subgroup to itself in the fiber group of the original space

                - **subgroup_map**: a function mapping an element of the fiber group of the original space to itself in the subgroup (returns ``None`` if the element is not in the subgroup)

        """

        o2 = o2_group()
        sg_id = o2._combine_subgroups(self._sg_id, id)
        sg, inclusion, restriction = self.fibergroup.subgroup(id)

        return GSpace2D(sg_id), inclusion, restriction

    @property
    def rotations_order(self):
        return self._sg_id[1]

    @property
    def flips_order(self):
        return 1 if self._sg_id[0] is not None else 0

    def _basis_generator(
        self,
        in_repr: Representation,
        out_repr: Representation,
        rings: List[float],
        sigma: List[float],
        learnable=False,
        **kwargs,
    ) -> KernelBasis:
        r"""
        Method that builds the analytical basis that spans the space of equivariant filters which
        are intertwiners between the representations induced from the representation ``in_repr`` and ``out_repr``.

        `kwargs` can be used to specify `maximum_frequency`

        Args:
            in_repr (Representation): the input representation
            out_repr (Representation): the output representation
            rings (list): radii of the rings where to sample the bases
            sigma (list): parameters controlling the width of each ring where the bases are sampled.

        Returns:
            the basis built

        """
        # TODO - add max_offset for cyclic and dihedral groups!
        if "maximum_frequency" in kwargs:
            maximum_frequency = kwargs["maximum_frequency"]
            del kwargs["maximum_frequency"]
        else:
            maximum_frequency = None

        if "layer_id" in kwargs:
            layer_id = kwargs["layer_id"]
            del kwargs["layer_id"]
        else:
            layer_id = None
        if self._sg_id[0] is not None and self._sg_id[1] == -1:
            return kernels_O2_act_R2(
                in_repr,
                out_repr,
                rings,
                sigma,
                axis=self._sg_id[0] / 2,
                maximum_frequency=maximum_frequency,
                filter=None,
                learnable=learnable,
                layer_id=layer_id,
                **kwargs,
            )
        elif self._sg_id == (None, -1):
            return kernels_SO2_act_R2(
                in_repr,
                out_repr,
                rings,
                sigma,
                maximum_frequency=maximum_frequency,
                filter=None,
                learnable=learnable,
                layer_id=layer_id,
                **kwargs,
            )
        elif self._sg_id[0] is None:
            sg_id = self._sg_id[1]
            return kernels_SO2_subgroup_act_R2(
                in_repr,
                out_repr,
                sg_id,
                rings,
                sigma,
                adjoint=None,
                maximum_frequency=maximum_frequency,
                filter=None,
                learnable=learnable,
                layer_id=layer_id,
                **kwargs,
            )
        else:
            return kernels_O2_subgroup_act_R2(
                in_repr,
                out_repr,
                self._sg_id,
                rings,
                sigma,
                axis=0.0,
                adjoint=None,
                maximum_frequency=maximum_frequency,
                filter=None,
                learnable=learnable,
                layer_id=layer_id,
                **kwargs,
            )

    @property
    def basespace_action(self) -> Representation:
        return self._base_action

    def __eq__(self, other):
        if isinstance(other, GSpace2D):
            return self._sg_id == other._sg_id
        else:
            return False

    def __hash__(self):
        return 1000 * hash(self.name) + hash(self._sg_id)


def rot2dOnR2(N: int = -1, maximum_frequency: int = 6) -> GSpace2D:
    r"""

    Describes rotation symmetries of the plane :math:`\R^2`.

    If ``N > 1``, the gspace models *discrete* rotations by angles which are multiple of :math:`\frac{2\pi}{N}`
    (:class:`~e2cnn.group.CyclicGroup`).
    Otherwise, if ``N=-1``, the gspace models *continuous* planar rotations (:class:`~e2cnn.group.SO2`).
    In that case the parameter ``maximum_frequency`` is required to specify the maximum frequency of the irreps of
    :class:`~e2cnn.group.SO2` (see its documentation for more details)

    Args:
        N (int): number of discrete rotations (integer greater than 1) or ``-1`` for continuous rotations
        maximum_frequency (int): maximum frequency of :class:`~e2cnn.group.SO2`'s irreps if ``N = -1``

    """
    assert isinstance(N, int)
    assert N == -1 or N > 0
    sg_id = None, N
    return GSpace2D(sg_id, maximum_frequency=maximum_frequency)


def flipRot2dOnR2(
    N: int = -1, maximum_frequency: int = 6, axis: float = np.pi / 2.0
) -> GSpace2D:
    r"""
    Describes reflectional and rotational symmetries of the plane :math:`\R^2`.

    Reflections are applied with respect to the line through the origin with an angle ``axis`` degrees with respect
    to the *X*-axis.

    If ``N > 1``, this gspace models reflections and *discrete* rotations by angles multiple of :math:`\frac{2\pi}{N}`
    (:class:`~e2cnn.group.DihedralGroup`).
    Otherwise, if ``N=-1`` (by default), the class models reflections and *continuous* planar rotations
    (:class:`~e2cnn.group.O2`).
    In that case, the parameter ``maximum_frequency`` is required to specify the maximum frequency of the irreps of
    :class:`~e2cnn.group.O2` (see its documentation for more details)

    .. note ::

        All axes obtained from the axis defined by ``axis`` with a rotation in the symmetry group are equivalent.
        For instance, if ``N = 4``, an axis :math:`\beta` is equivalent to the axis :math:`\beta + \pi/2`.
        It follows that for ``N = -1``, i.e. in case the symmetry group contains all continuous rotations, any
        reflection axis is theoretically equivalent.
        In practice, though, a basis for equivariant convolutional filter sampled on a grid is affected by the
        specific choice of the axis. In general, choosing an axis aligned with the grid (an horizontal or a
        vertical axis, i.e. :math:`0` or :math:`\pi/2`) is suggested.

    Args:
        N (int): number of discrete rotations (integer greater than 1) or -1 for continuous rotations
        maximum_frequency (int): maximum frequency of :class:`~e2cnn.group.O2` 's irreps if ``N = -1``
        axis (float, optional): the slope of the axis of the flip (in radians)

    """

    assert isinstance(N, int)
    assert N == -1 or N > 0
    sg_id = 2 * axis, N
    return GSpace2D(sg_id, maximum_frequency=maximum_frequency)


def flip2dOnR2(axis: float = np.pi / 2) -> GSpace2D:
    r"""

    Describes reflectional symmetries of the plane :math:`\R^2`.

    Reflections are applied along the line through the origin with an angle ``axis`` degrees with respect to
    the *X*-axis.

    Args:
        axis (float, optional): the slope of the axis of the reflection (in radians).
                                By default, the vertical axis is used (:math:`\pi/2`).

    """
    sg_id = 2 * axis, 1
    return GSpace2D(sg_id, maximum_frequency=1)


def trivialOnR2() -> GSpace2D:
    r"""
    Describes the plane :math:`\R^2` without considering any origin-preserving symmetry.
    This is modeled by choosing trivial fiber group :math:`\{e\}`.

    .. note ::
        This models the symmetries of conventional *Convolutional Neural Networks* which are not equivariant to
        origin preserving transformations such as rotations and reflections.

    """
    sg_id = (None, 1)
    return GSpace2D(sg_id, maximum_frequency=1)
