from __future__ import annotations

from escnn import gspaces
from escnn import kernels

from ..kernels.r3 import *

from escnn.group import *

from .utils import linear_transform_array_3d

import numpy as np

from typing import Tuple, Union, Callable, List

__all__ = [
    "GSpace3D",
    ###########
    "flipRot3dOnR3",
    "rot3dOnR3",
    "fullIcoOnR3",
    "icoOnR3",
    "fullOctaOnR3",
    "octaOnR3",
    "dihedralOnR3",
    "rot2dOnR3",
    "conicalOnR3",
    "fullCylindricalOnR3",
    "cylindricalOnR3",
    "mirOnR3",
    "invOnR3",
    "trivialOnR3",
]


class GSpace3D(gspaces.GSpace):
    def __init__(self, sg_id: Tuple, maximum_frequency: int = 2):
        r"""

        Describes reflectional and rotational symmetries of the plane :math:`\R^3`.

        .. note ::
            A point :math:`\bold{v} \in \R^3` is parametrized using an :math:`(X, Y, Z)` convention,
            i.e. :math:`\bold{v} = (x, y, z)^T`.
            The representation :attr:`escnn.gspaces.GSpace3D.basespace_action` also assumes this convention.

            However, when working with voxel data, the :math:`(-Z, -Y, X)` convention is used.
            That means that, in a 5-dimensional feature tensor of shape ``(B, C, D1, D2, D3)``, the last dimension
            is the X axis, the second last the (inverted) Y axis and then the (inverted) Z axis.
            Note that this is consistent with 2D images, where a :math:`(-Y, X)` convention is used.

            This is especially relevant when transforming a :class:`~escnn.nn.GeometricTensor` or when building
            convolutional filters in :class:`~escnn.nn.R3Conv` which should be equivariant to subgroups of :math:`\O3`
            (e.g. when choosing the rotation axis for :func:`~escnn.gspaces.rot2dOnR3`).

        """

        o3 = o3_group(maximum_frequency=maximum_frequency)
        _sg_id = o3._process_subgroup_id(sg_id)
        fibergroup, inclusion, restriction = o3.subgroup(_sg_id)

        # TODO - catch sg_id and build a dictionary of more meaningful names
        # use the input sg_id instead of the processed one to avoid adding the adjoint parameter unless specified
        name = f"{fibergroup}_on_R3[{sg_id}]"

        self._sg_id = _sg_id
        self._inclusion = inclusion
        self._restriction = restriction
        self._base_action = o3.standard_representation().restrict(_sg_id)

        super(GSpace3D, self).__init__(fibergroup, 3, name)

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

        o3 = o3_group()
        sg_id = o3._combine_subgroups(self._sg_id, id)
        sg, inclusion, restriction = self.fibergroup.subgroup(id)

        return GSpace3D(sg_id), inclusion, restriction

    def _basis_generator(
        self,
        in_repr: Representation,
        out_repr: Representation,
        rings: List[float],
        sigma: List[float],
        learnable=False,
        **kwargs,
    ) -> kernels.KernelBasis:
        r"""
        Method that builds the analitical basis that spans the space of equivariant filters which
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
        if self._sg_id[:2] == (True, "so3"):
            return kernels_O3_act_R3(
                in_repr,
                out_repr,
                rings,
                sigma,
                maximum_frequency=maximum_frequency,
                adjoint=None,
                learnable=learnable,
                layer_id=layer_id,
                **kwargs,
            )
        elif self._sg_id[:2] == (False, "so3"):
            return kernels_SO3_act_R3(
                in_repr,
                out_repr,
                rings,
                sigma,
                maximum_frequency=maximum_frequency,
                adjoint=None,
                learnable=learnable,
                layer_id=layer_id,
                **kwargs,
            )
        elif self._sg_id[0] == False:
            sg_id = self._sg_id[1:]
            if isinstance(sg_id[-1], GroupElement):
                # the adjoint is an O(3) group element
                # convert it to an SO(3) element
                # not that even if the adjoint contains the 3d inversion, we can ignore it
                # (since O(3) is a direct product, the inversion commutes with any 3D rotation)
                adj = sg_id[-1]
                so3 = so3_group()
                adj = so3.element(adj.value[1], adj.param)
                sg_id = sg_id[:-1] + (adj,)
            return kernels_SO3_subgroup_act_R3(
                in_repr,
                out_repr,
                sg_id,
                rings,
                sigma,
                maximum_frequency=maximum_frequency,
                adjoint=None,
                learnable=learnable,
                layer_id=layer_id,
                **kwargs,
            )
        else:
            return kernels_O3_subgroup_act_R3(
                in_repr,
                out_repr,
                self._sg_id,
                rings,
                sigma,
                maximum_frequency=maximum_frequency,
                adjoint=None,
                learnable=learnable,
                layer_id=layer_id,
                **kwargs,
            )

    @property
    def basespace_action(self) -> Representation:
        return self._base_action

    def __eq__(self, other):
        if isinstance(other, GSpace3D):
            return self._sg_id == other._sg_id
        else:
            return False

    def __hash__(self):
        return 1000 * hash(self.name) + hash(self._sg_id)


########################################################################################################################


def flipRot3dOnR3(maximum_frequency: int = 2) -> GSpace3D:
    r"""
    Describes 3D rotation and inversion symmetries in the space :math:`\R^3`.

    .. todo ::
        rename to invRot3dOnR3?

    Args:
        maximum_frequency (int): maximum frequency of :class:`~e2cnn.group.O3`'s irreps

    """
    sg_id = "o3"
    return GSpace3D(sg_id, maximum_frequency=maximum_frequency)


def rot3dOnR3(maximum_frequency: int = 2) -> GSpace3D:
    r"""
    Describes 3D rotation symmetries in the space :math:`\R^3`.

    Args:
        maximum_frequency (int): maximum frequency of :class:`~e2cnn.group.SO3`'s irreps

    """
    sg_id = "so3"
    return GSpace3D(sg_id, maximum_frequency=maximum_frequency)


def fullIcoOnR3() -> GSpace3D:
    sg_id = True, "ico"
    return GSpace3D(sg_id, maximum_frequency=4)


def icoOnR3() -> GSpace3D:
    r"""
    Describes 3D rotation symmetries of a Icosahedron (or Dodecahedron) in the space :math:`\R^3`
    """
    sg_id = False, "ico"
    return GSpace3D(sg_id, maximum_frequency=4)


def fullOctaOnR3() -> GSpace3D:
    sg_id = True, "octa"
    return GSpace3D(sg_id, maximum_frequency=3)


def octaOnR3() -> GSpace3D:
    r"""
    Describes 3D rotation symmetries of an Octahedron (or Cube) in the space :math:`\R^3`
    """
    sg_id = False, "octa"
    return GSpace3D(sg_id, maximum_frequency=3)


def dihedralOnR3(
    n: int = -1,
    axis: float = np.pi / 2,
    adjoint: GroupElement = None,
    maximum_frequency: int = 2,
) -> GSpace3D:
    r"""
    Describes 2D rotation symmetries along the :math:`Z` axis in the space :math:`\R^3` and :math:`\pi` rotations
    along the ``axis`` in the :math:`XY` plane, i.e. the rotations inside the plane :math:`XY` and reflections around
    the ``axis``.

    The ``adjoint`` parameter can be a :class:`~escnn.group.GroupElement` of :class:`~escnn.group.O3`.
    If not ``None`` (which is equivalent to the identity), this specifies another :math:`\SO2` subgroup of :math:`\O3`
    which is adjoint to the :math:`\SO2` subgroup of rotations around the :math:`Z` axis.
    If ``adjoint`` is the group element :math:`A \in \O3`, the new subgroup would then represent rotations around the
    axis :math:`A^{-1} \cdot (0, 0, 1)^T`.

    If ``N > 1``, the gspace models *discrete* rotations by angles which are multiple of :math:`\frac{2\pi}{N}`
    (:class:`~e2cnn.group.CyclicGroup`).
    Otherwise, if ``N=-1``, the gspace models *continuous* planar rotations (:class:`~e2cnn.group.SO2`).
    In that case the parameter ``maximum_frequency`` is required to specify the maximum frequency of the irreps of
    :class:`~e2cnn.group.SO2` (see its documentation for more details)

    Args:
        N (int): number of discrete rotations (integer greater than 1) or ``-1`` for continuous rotations
        adjoint (GroupElement, optional): an element of :math:`\O3`
        maximum_frequency (int): maximum frequency of :class:`~e2cnn.group.SO2`'s irreps if ``N = -1``

    """
    assert isinstance(n, int)
    assert n == -1 or n > 0

    sg_id = False, 2 * axis, n

    if adjoint is not None:
        sg_id += (adjoint,)

    return GSpace3D(sg_id, maximum_frequency=maximum_frequency)


def rot2dOnR3(
    n: int = -1, adjoint: GroupElement = None, maximum_frequency: int = 2
) -> GSpace3D:
    r"""

    Describes 2D rotation symmetries along the :math:`Z` axis in the space :math:`\R^3`, i.e. the rotations inside the
    plane :math:`XY`.

    ``adjoint`` is a :class:`~escnn.group.GroupElement` of :class:`~escnn.group.O3`.
    If not ``None`` (which is equivalent to the identity), this specifies another :math:`\SO2` subgroup of :math:`\O3`
    which is adjoint to the :math:`\SO2` subgroup of rotations around the :math:`Z` axis.
    If ``adjoint`` is the group element :math:`A \in \O3`, the new subgroup would then represent rotations around the
    axis :math:`A^{-1} \cdot (0, 0, 1)^T`.

    If ``N > 1``, the gspace models *discrete* rotations by angles which are multiple of :math:`\frac{2\pi}{N}`
    (:class:`~e2cnn.group.CyclicGroup`).
    Otherwise, if ``N=-1``, the gspace models *continuous* planar rotations (:class:`~e2cnn.group.SO2`).
    In that case the parameter ``maximum_frequency`` is required to specify the maximum frequency of the irreps of
    :class:`~e2cnn.group.SO2` (see its documentation for more details)

    Args:
        N (int): number of discrete rotations (integer greater than 1) or ``-1`` for continuous rotations
        adjoint (GroupElement, optional): an element of :math:`\O3`
        maximum_frequency (int): maximum frequency of :class:`~e2cnn.group.SO2`'s irreps if ``N = -1``

    """
    assert isinstance(n, int)
    assert n == -1 or n > 0
    sg_id = False, False, n

    if adjoint is not None:
        sg_id += (adjoint,)

    return GSpace3D(sg_id, maximum_frequency=maximum_frequency)


def conicalOnR3(
    n: int = -1,
    axis: float = np.pi / 2.0,
    adjoint: GroupElement = None,
    maximum_frequency: int = 2,
) -> GSpace3D:
    assert isinstance(n, int)
    assert n == -1 or n > 0

    sg_id = "cone", 2 * axis, n

    if adjoint is not None:
        sg_id += (adjoint,)

    return GSpace3D(sg_id, maximum_frequency=maximum_frequency)


def mirOnR3(axis: float = np.pi / 2, adjoint: GroupElement = None) -> GSpace3D:
    r"""

    Describes mirroring with respect to a plane in the space :math:`\R^3`.

    .. todo ::
        Document what ``axis`` and ``adjoint`` describe or change parameters, just getting a :math:`\bold{v} \in \R^3`
        vector in input which specifies the mirroring axis.

    """

    sg_id = "cone", 2 * axis, 1

    if adjoint is not None:
        sg_id += (adjoint,)

    return GSpace3D(sg_id, maximum_frequency=1)


def fullCylindricalOnR3(
    n: int = -1,
    axis: float = np.pi / 2,
    adjoint: GroupElement = None,
    maximum_frequency: int = 2,
) -> GSpace3D:
    assert isinstance(n, int)
    assert n == -1 or n > 0

    sg_id = True, axis, n

    if adjoint is not None:
        sg_id += (adjoint,)

    return GSpace3D(sg_id, maximum_frequency=maximum_frequency)


def cylindricalOnR3(
    n: int = -1, adjoint: GroupElement = None, maximum_frequency: int = 2
) -> GSpace3D:
    assert isinstance(n, int)
    assert n == -1 or n > 0

    sg_id = True, False, n

    if adjoint is not None:
        sg_id += (adjoint,)

    return GSpace3D(sg_id, maximum_frequency=maximum_frequency)


def invOnR3() -> GSpace3D:
    r"""

    Describes the inversion symmetry of the space :math:`\R^3`.

    An inversion flips the sign of all coordinates, mapping a vector :math:`\bold{v} \in \R^3` to :math:`-\bold{v}`.

    """
    sg_id = True, False, 1
    return GSpace3D(sg_id, maximum_frequency=1)


def trivialOnR3() -> GSpace3D:
    r"""
    Describes the space :math:`\R^3` without considering any origin-preserving symmetry.
    This is modeled by choosing trivial fiber group :math:`\{e\}`.

    .. note ::
        This models the symmetries of conventional *Convolutional Neural Networks* which are not equivariant to
        origin preserving transformations such as rotations and reflections.

    """
    sg_id = False, False, 1
    return GSpace3D(sg_id, maximum_frequency=1)
