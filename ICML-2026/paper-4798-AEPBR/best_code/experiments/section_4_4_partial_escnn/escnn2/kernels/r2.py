from escnn.kernels.basis import KernelBasis, AdjointBasis
from escnn.kernels.steerable_basis import SteerableKernelBasis

# from escnn.kernels.wignereckart_solver import WignerEckartBasis
from .wignereckart_solver import (
    WignerEckartBasis,
    LearnableWignerEckartBasis,
    RestrictedWignerEckartBasis,
    LearnableRestrictedWignerEckartBasis,
)

from escnn.kernels.polar_basis import GaussianRadialProfile
from escnn.kernels.polar_basis import CircularShellsBasis

from escnn.group import *

import numpy as np

from typing import List, Union, Callable, Dict, Tuple
from functools import partial

__all__ = [
    "kernels_SO2_subgroup_act_R2",
    "kernels_O2_subgroup_act_R2",
    "kernels_SO2_act_R2",
    "kernels_O2_act_R2",
    "kernels_CN_act_R2",
    "kernels_DN_act_R2",
    "kernels_Flip_act_R2",
    "kernels_Trivial_act_R2",
]


def kernels_SO2_act_R2(
    in_repr: Representation,
    out_repr: Representation,
    radii: List[float],
    sigma: Union[List[float], float],
    maximum_frequency: int = None,
    filter: Callable[[Dict], bool] = None,
    learnable: bool = False,
    layer_id: bool = None,
    **mapping_kwargs
) -> KernelBasis:
    r"""

    Builds a basis for convolutional kernels equivariant to continuous rotations, modeled by the
    group :math:`SO(2)`.
    ``in_repr`` and ``out_repr`` need to be :class:`~escnn.group.Representation` s of :class:`~escnn.group.SO2`.

    Because the equivariance constraints allow any choice of radial profile, we use a
    :class:`~escnn.kernels.GaussianRadialProfile`.
    ``radii`` specifies the radial distances at which the rings are centered while ``sigma`` contains the width of each
    of these rings (see :class:`~escnn.kernels.GaussianRadialProfile`).

    Args:
        in_repr (Representation): the representation specifying the transformation of the input feature field
        out_repr (Representation): the representation specifying the transformation of the output feature field
        radii (list): radii of the rings defining the basis for the radial profile
        sigma (list or float): widths of the rings defining the basis for the radial profile

    """
    assert in_repr.group == out_repr.group

    group = in_repr.group

    assert isinstance(group, SO2)

    radial_profile = GaussianRadialProfile(radii, sigma)

    if maximum_frequency is None:
        max_in_freq = max(freq for freq, in in_repr.irreps)
        max_out_freq = max(freq for freq, in out_repr.irreps)
        maximum_frequency = max_in_freq + max_out_freq
    kwargs = dict(sg_id=(None, -1))
    if learnable:
        kwargs["layer_id"] = layer_id
        kwargs["mapping_kwargs"] = mapping_kwargs
    return SteerableKernelBasis(
        CircularShellsBasis(maximum_frequency, radial_profile, filter=filter),
        in_repr,
        out_repr,
        (
            LearnableRestrictedWignerEckartBasis
            if learnable
            else RestrictedWignerEckartBasis
        ),
        **kwargs,
    )


def kernels_O2_act_R2(
    in_repr: Representation,
    out_repr: Representation,
    radii: List[float],
    sigma: Union[List[float], float],
    maximum_frequency: int = None,
    axis: float = np.pi / 2,
    adjoint: np.ndarray = None,
    filter: Callable[[Dict], bool] = None,
    learnable: bool = False,
    layer_id: bool = None,
    **mapping_kwargs
) -> KernelBasis:
    r"""

    Builds a basis for convolutional kernels equivariant to reflections and continuous rotations, modeled by the
    group :math:`O(2)`.
    ``in_repr`` and ``out_repr`` need to be :class:`~escnn.group.Representation` s of :class:`~escnn.group.O2`.

    Because the equivariance constraints allow any choice of radial profile, we use a
    :class:`~escnn.kernels.GaussianRadialProfile`.
    ``radii`` specifies the radial distances at which the rings are centered while ``sigma`` contains the width of each
    of these rings (see :class:`~escnn.kernels.GaussianRadialProfile`).

    Because :math:`O(2)` contains all rotations, the reflection element of the group can be associated to any reflection
    axis. Reflections along other axes can be obtained by composition with rotations.
    However, a choice of this axis is required to align the basis with respect to the action of the group.

    Args:
        in_repr (Representation): the representation specifying the transformation of the input feature field
        out_repr (Representation): the representation specifying the transformation of the output feature field
        radii (list): radii of the rings defining the basis for the radial profile
        sigma (list or float): widths of the rings defining the basis for the radial profile
        axis (float, optional): angle of the axis of the reflection element
        adjoint (~numpy.ndarray, optional): instead of specifying a reflection axis, you can pass a 2x2 orthogonal
            matrix defining a change of basis on the base space

    """
    assert in_repr.group == out_repr.group

    group = in_repr.group
    assert isinstance(group, O2)
    radial_profile = GaussianRadialProfile(radii, sigma)
    if maximum_frequency is None:
        max_in_freq = max(freq for _, freq in in_repr.irreps)
        max_out_freq = max(freq for _, freq in out_repr.irreps)
        maximum_frequency = max_in_freq + max_out_freq

    kwargs = dict()
    if learnable:
        kwargs["layer_id"] = layer_id
        kwargs["mapping_kwargs"] = mapping_kwargs

    basis = SteerableKernelBasis(
        CircularShellsBasis(
            maximum_frequency, radial_profile, filter=filter, axis=axis
        ),
        in_repr,
        out_repr,
        LearnableWignerEckartBasis if learnable else WignerEckartBasis,
        **kwargs,
    )

    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (2, 2)
        basis = AdjointBasis(basis, adjoint)

    return basis


###### Automatic subgroups kernel bases


def kernels_O2_subgroup_act_R2(
    in_repr: Representation,
    out_repr: Representation,
    sg_id,
    radii: List[float],
    sigma: Union[List[float], float],
    maximum_frequency: int = 5,
    axis: float = np.pi / 2.0,
    adjoint: np.ndarray = None,
    filter: Callable[[Dict], bool] = None,
    learnable: bool = False,
    layer_id: bool = None,
    **mapping_kwargs
) -> KernelBasis:
    o2 = o2_group(maximum_frequency)

    group, _, _ = o2.subgroup(sg_id)
    assert in_repr.group == group
    assert out_repr.group == group

    radial_profile = GaussianRadialProfile(radii, sigma)

    kwargs = dict(sg_id=sg_id)
    if learnable:
        kwargs["layer_id"] = layer_id
        kwargs["mapping_kwargs"] = mapping_kwargs
    basis = SteerableKernelBasis(
        CircularShellsBasis(
            maximum_frequency, radial_profile, filter=filter, axis=axis
        ),
        in_repr,
        out_repr,
        (
            LearnableRestrictedWignerEckartBasis
            if learnable
            else RestrictedWignerEckartBasis
        ),
        **kwargs,
    )

    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (2, 2)
        basis = AdjointBasis(basis, adjoint)

    return basis


def kernels_SO2_subgroup_act_R2(
    in_repr: Representation,
    out_repr: Representation,
    sg_id,
    radii: List[float],
    sigma: Union[List[float], float],
    maximum_frequency: int = 5,
    adjoint: np.ndarray = None,
    filter: Callable[[Dict], bool] = None,
    learnable: bool = False,
    layer_id: bool = None,
    **mapping_kwargs
) -> KernelBasis:
    so2 = so2_group(maximum_frequency)

    group, _, _ = so2.subgroup(sg_id)
    assert in_repr.group == group
    assert out_repr.group == group

    o2 = o2_group(maximum_frequency)
    sg_id = o2._combine_subgroups((None, -1), sg_id)

    radial_profile = GaussianRadialProfile(radii, sigma)
    kwargs = dict(sg_id=sg_id)
    if learnable:
        kwargs["layer_id"] = layer_id
        kwargs["mapping_kwargs"] = mapping_kwargs
    basis = SteerableKernelBasis(
        CircularShellsBasis(maximum_frequency, radial_profile, filter=filter),
        in_repr,
        out_repr,
        (
            LearnableRestrictedWignerEckartBasis
            if learnable
            else RestrictedWignerEckartBasis
        ),
        **kwargs,
    )

    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (2, 2)
        basis = AdjointBasis(basis, adjoint)

    return basis


###### Discrete Symmetries


def kernels_CN_act_R2(
    in_repr: Representation,
    out_repr: Representation,
    radii: List[float],
    sigma: Union[List[float], float],
    maximum_frequency: int = None,
    max_offset: int = None,
    filter: Callable[[Dict], bool] = None,
    learnable: bool = False,
    layer_id: bool = None,
    **mapping_kwargs
) -> KernelBasis:
    r"""

    Builds a basis for convolutional kernels equivariant to :math:`N` discrete rotations, modeled by
    the group :math:`C_N`.
    ``in_repr`` and ``out_repr`` need to be :class:`~escnn.group.Representation` s of :class:`~escnn.group.CyclicGroup`.

    Because the equivariance constraints allow any choice of radial profile, we use a
    :class:`~escnn.kernels.GaussianRadialProfile`.
    ``radii`` specifies the radial distances at which the rings are centered while ``sigma`` contains the width of each
    of these rings (see :class:`~escnn.kernels.GaussianRadialProfile`).

    The analytical angular solutions of kernel constraints belong to an infinite dimensional space and so can be
    expressed in terms of infinitely many basis elements, each associated with one unique frequency. Because the kernels
    are then sampled on a finite number of points (e.g. the cells of a grid), only low-frequency solutions needs to be
    considered. This enables us to build a finite dimensional basis containing only a finite subset of all analytical
    solutions. ``maximum_frequency`` is an integer controlling the highest frequency sampled in the basis.

    Frequencies also appear in a basis with a period of :math:`N`, i.e. if the basis contains an element with frequency
    :math:`k`, then it also contains an element with frequency :math:`k + N`.
    In the analytical solutions shown in Table 11 `here <https://arxiv.org/abs/1911.08251>`_, each solution has a
    parameter :math:`t` or :math:`\hat{t}`.
    ``max_offset`` defines the maximum absolute value of these two numbers.

    Either ``maximum_frequency`` or ``max_offset`` must be specified.


    Args:
        in_repr (Representation): the representation specifying the transformation of the input feature field
        out_repr (Representation): the representation specifying the transformation of the output feature field
        radii (list): radii of the rings defining the basis for the radial profile
        sigma (list or float): widths of the rings defining the basis for the radial profile
        maximum_frequency (int): maximum frequency of the basis
        max_offset (int): maximum offset in the frequencies of the basis

    """
    assert in_repr.group == out_repr.group

    group = in_repr.group

    assert isinstance(group, CyclicGroup)

    prefilter = filter
    if max_offset is not None and prefilter is not None:
        filter = lambda attr, max_offset=max_offset, prefilter=prefilter: (
            attr["j"][0] - attr["_j"][0] <= max_offset
        ) and prefilter(attr)
    elif max_offset is not None and prefilter is None:
        filter = lambda attr, max_offset=max_offset: (
            attr["j"][0] - attr["_j"][0] <= max_offset
        )
    else:
        filter = prefilter

    sg_id = group.order()
    return kernels_SO2_subgroup_act_R2(
        in_repr,
        out_repr,
        sg_id,
        radii,
        sigma,
        maximum_frequency=maximum_frequency,
        adjoint=None,
        filter=filter,
        learnable=learnable,
        layer_id=layer_id,
        **mapping_kwargs,
    )


def kernels_DN_act_R2(
    in_repr: Representation,
    out_repr: Representation,
    radii: List[float],
    sigma: Union[List[float], float],
    axis: float = np.pi / 2,
    maximum_frequency: int = None,
    max_offset: int = None,
    adjoint: np.ndarray = None,
    filter: Callable[[Dict], bool] = None,
    learnable: bool = False,
    layer_id: bool = None,
    **mapping_kwargs
) -> KernelBasis:
    r"""

    Builds a basis for convolutional kernels equivariant to reflections and :math:`N` discrete rotations,
    modeled by the group :math:`D_N`.
    ``in_repr`` and ``out_repr`` need to be :class:`~escnn.group.Representation` s
    of :class:`~escnn.group.DihedralGroup`.

    Because the equivariance constraints allow any choice of radial profile, we use a
    :class:`~escnn.kernels.GaussianRadialProfile`.
    ``radii`` specifies the radial distances at which the rings are centered while ``sigma`` contains the width of each
    of these rings (see :class:`~escnn.kernels.GaussianRadialProfile`).

    The parameter ``axis`` is the angle in radians (with respect to the horizontal axis, rotating counter-clockwise)
    which defines the reflection axis for the reflection element of the group.

    Frequencies also appear in a basis with a period of :math:`N`, i.e. if the basis contains an element with frequency
    :math:`k`, then it also contains an element with frequency :math:`k + N`.
    In the analytical solutions shown in Table 12 `here <https://arxiv.org/abs/1911.08251>`_, each solution has a
    parameter :math:`t` or :math:`\hat{t}`.
    ``max_offset`` defines the maximum absolute value of these two numbers.

    Either ``maximum_frequency`` or ``max_offset`` must be specified.


    Args:
        in_repr (Representation): the representation specifying the transformation of the input feature field
        out_repr (Representation): the representation specifying the transformation of the output feature field
        radii (list): radii of the rings defining the basis for the radial profile
        sigma (list or float): widths of the rings defining the basis for the radial profile
        maximum_frequency (int): maximum frequency of the basis
        max_offset (int): maximum offset in the frequencies of the basis
        axis (float): angle defining the reflection axis
        adjoint (~numpy.ndarray, optional): instead of specifying a reflection axis, you can pass a 2x2 orthogonal
            matrix defining a change of basis on the base space


    """
    assert in_repr.group == out_repr.group

    group = in_repr.group

    assert isinstance(group, DihedralGroup)

    N = group.rotation_order

    prefilter = filter
    if max_offset is not None and prefilter is not None:
        filter = lambda attr, max_offset=max_offset, prefilter=prefilter: (
            attr["j"][1] - attr["_j"][1] <= max_offset
        ) and prefilter(attr)
    elif max_offset is not None and prefilter is None:
        filter = lambda attr, max_offset=max_offset: (
            attr["j"][1] - attr["_j"][1] <= max_offset
        )
    else:
        filter = prefilter

    sg_id = (0.0, N)
    return kernels_O2_subgroup_act_R2(
        in_repr,
        out_repr,
        sg_id,
        radii,
        sigma,
        axis=axis,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint,
        filter=filter,
        learnable=learnable,
        layer_id=layer_id,
        **mapping_kwargs,
    )


def kernels_Flip_act_R2(
    in_repr: Representation,
    out_repr: Representation,
    radii: List[float],
    sigma: Union[List[float], float],
    axis: float = np.pi / 2,
    maximum_frequency: int = None,
    adjoint: np.ndarray = None,
    filter: Callable[[Dict], bool] = None,
    learnable: bool = False,
    layer_id: bool = None,
    **mapping_kwargs
) -> KernelBasis:
    r"""

    Builds a basis for convolutional kernels equivariant to reflections.
    ``in_repr`` and ``out_repr`` need to be :class:`~escnn.group.Representation` s of :class:`~escnn.group.CyclicGroup`
    with ``N=2``.

    Because the equivariance constraints allow any choice of radial profile, we use a
    :class:`~escnn.kernels.GaussianRadialProfile`.
    ``radii`` specifies the radial distances at which the rings are centered while ``sigma`` contains the width of each
    of these rings (see :class:`~escnn.kernels.GaussianRadialProfile`).

    The parameter ``axis`` is the angle in radians (with respect to the horizontal axis, rotating counter-clockwise)
    which defines the reflection axis.

    The analytical angular solutions of kernel constraints belong to an infinite dimensional space and so can be
    expressed in terms of infinitely many basis elements. Only a finite subset can however be implemented.
    ``maximum_frequency`` defines the maximum frequency of a finite-dimensional bandlimited subspace and is therefore
    necessary to specify it.
    See :func:`~escnn.kernels.kernels_CN_act_R2` for more details.


    Args:
        in_repr (Representation): the representation specifying the transformation of the input feature field
        out_repr (Representation): the representation specifying the transformation of the output feature field
        radii (list): radii of the rings defining the basis for the radial profile
        sigma (list or float): widths of the rings defining the basis for the radial profile
        axis (float): angle defining the reflection axis
        maximum_frequency (int): maximum frequency of the basis
        adjoint (~numpy.ndarray, optional): instead of specifying a reflection axis, you can pass a 2x2 orthogonal
            matrix defining a change of basis on the base space

    """
    assert in_repr.group == out_repr.group
    group = in_repr.group
    assert isinstance(group, CyclicGroup) and group.order() == 2

    sg_id = (0.0, 1)
    return kernels_O2_subgroup_act_R2(
        in_repr,
        out_repr,
        sg_id,
        radii,
        sigma,
        axis=axis,
        maximum_frequency=maximum_frequency,
        adjoint=adjoint,
        filter=filter,
        learnable=learnable,
        layer_id=layer_id,
        **mapping_kwargs,
    )


def kernels_Trivial_act_R2(
    in_repr: Representation,
    out_repr: Representation,
    radii: List[float],
    sigma: Union[List[float], float],
    maximum_frequency: int = None,
    filter: Callable[[Dict], bool] = None,
    learnable: bool = False,
    layer_id: bool = None,
    **mapping_kwargs
) -> KernelBasis:
    r"""

    Builds a basis for unconstrained convolutional kernels.

    This is equivalent to use :func:`~escnn.kernels.kernels_CN_act_R2` with an instance of
    :class:`~escnn.group.CyclicGroup` with ``N=1`` (the trivial group :math:`C_1`).

    ``in_repr`` and ``out_repr`` need to be associated with an instance of :class:`~escnn.group.CyclicGroup` with
    ``N=1``.

    Because the equivariance constraints allow any choice of radial profile, we use a
    :class:`~escnn.kernels.GaussianRadialProfile`.
    ``radii`` specifies the radial distances at which the rings are centered while ``sigma`` contains the width of each
    of these rings (see :class:`~escnn.kernels.GaussianRadialProfile`).

    The analytical angular solutions of kernel constraints belong to an infinite dimensional space and so can be
    expressed in terms of infinitely many basis elements. Only a finite subset can however be implemented.
    ``maximum_frequency`` defines the maximum frequency of a finite-dimensinal bandlimited subspace and is therefore
    necessary to specify it.
    See :func:`~escnn.kernels.kernels_CN_act_R2` for more details.

    Args:
        in_repr (Representation): the representation specifying the transformation of the input feature field
        out_repr (Representation): the representation specifying the transformation of the output feature field
        radii (list): radii of the rings defining the basis for the radial profile
        sigma (list or float): widths of the rings defining the basis for the radial profile
        axis (float): angle defining the reflection axis
        maximum_frequency (int): maximum frequency of the basis

    """

    assert in_repr.group == out_repr.group

    group = in_repr.group
    assert isinstance(group, CyclicGroup) and group.order() == 1

    sg_id = 1
    return kernels_SO2_subgroup_act_R2(
        in_repr,
        out_repr,
        sg_id,
        radii,
        sigma,
        maximum_frequency=maximum_frequency,
        adjoint=None,
        filter=filter,
        learnable=learnable,
        layer_id=layer_id,
        **mapping_kwargs,
    )
