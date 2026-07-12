
from __future__ import annotations

import escnn.kernels
import escnn.group

from .utils import linear_transform_array_nd

from abc import ABC, abstractmethod
from typing import Tuple, Callable, List, Union

from collections import defaultdict

import numpy as np


__all__ = ["GSpace"]


class GSpace(ABC):
    
    def __init__(self, fibergroup: escnn.group.Group, dimensionality: int, name: str):
        r"""
        Abstract class for G-spaces.
        
        A ``GSpace`` describes the space where a signal lives (e.g. :math:`\R^2` for planar images) and its symmetries
        (e.g. rotations or reflections).
        As an `Euclidean` base space is assumed, a G-space is fully specified by the ``dimensionality`` of the space
        and a choice of origin-preserving symmetry group (``fibergroup``).
        
        .. seealso::
            
            :class:`~escnn.gspaces.GSpace0D`,
            :class:`~escnn.gspaces.GSpace2D`,
            :class:`~escnn.gspaces.GSpace3D`,
            or the factory methods
            :class:`~escnn.gspaces.flipRot3dOnR3`,
            :class:`~escnn.gspaces.rot3dOnR3`,
            :class:`~escnn.gspaces.rot2dOnR3`,
            :class:`~escnn.gspaces.flipRot2dOnR2`,
            :class:`~escnn.gspaces.rot2dOnR2`,
            :class:`~escnn.gspaces.flip2dOnR2`,
            :class:`~escnn.gspaces.trivialOnR2`
        
        .. note ::
        
            Mathematically, this class describes a *Principal Bundle*
            :math:`\pi : (\R^D, +) \rtimes G \to \mathbb{R}^D, tg \mapsto tG`,
            with the Euclidean space :math:`\mathbb{R}^D` (where :math:`D` is the ``dimensionality``) as `base space`
            and :math:`G` as `fiber group` (``fibergroup``).
            For more details on this interpretation we refer to
            `A General Theory of Equivariant CNNs On Homogeneous Spaces <https://papers.nips.cc/paper/9114-a-general-theory-of-equivariant-cnns-on-homogeneous-spaces.pdf>`_.
        
        
        Args:
            fibergroup (Group): the fiber group
            dimensionality (int): the dimensionality of the Euclidean space on which a signal is defined
            name (str): an identification name
        
        Attributes:
            ~.fibergroup (Group): the fiber group
            ~.dimensionality (int): the dimensionality of the Euclidean space on which a signal is defined
            ~.name (str): an identification name
            ~.basespace (str): the name of the space whose symmetries are modeled. It is an Euclidean space :math:`\R^D`.
        
        """
        
        # TODO move this sub-package to PyTorch

        self.name = name
        self.dimensionality = dimensionality
        self.fibergroup = fibergroup
        self.basespace = f"R^{self.dimensionality}"

        # in order to not recompute the basis for the same intertwiner as many times as it appears, we store the basis
        # in these dictionaries the first time we compute it

        # Store the computed intertwiners between irreps
        # - key = (filter size, sigma, rings)
        # - value = dictionary mapping (input_irrep, output_irrep) pairs to the corresponding basis
        self._irreps_intertwiners_basis_memory = defaultdict(lambda: dict())

        # Store the computed intertwiners between general representations
        # - key = (filter size, sigma, rings)
        # - value = dictionary mapping (input_repr, output_repr) pairs to the corresponding basis
        self._fields_intertwiners_basis_memory = defaultdict(dict)

        # Store the computed intertwiners between general representations
        # - key = (input_repr, output_repr)
        # - value = the corresponding basis
        self._fields_intertwiners_basis_memory_fiber_basis = dict()

    def type(self, *representations: escnn.group.Representation) -> escnn.nn.FieldType:
        r"""
            Shortcut to build a :class:`~escnn.nn.FieldType`.
            This is equivalent to ``FieldType(gspace, representations)``.
        """
        return escnn.nn.FieldType(self, representations)

    @abstractmethod
    def restrict(self, id) -> Tuple[GSpace, Callable, Callable]:
        r"""

        Build the :class:`~escnn.gspaces.GSpace` associated with the subgroup of the current fiber group identified by
        the input ``id``.
        This reduces the level of symmetries of the base space to be considered.

        Check the ``restrict`` method's documentation in the non-abstract subclass used for a description of the
        parameter ``id``.

        Args:
            id: id of the subgroup

        Returns:
            a tuple containing

                - **gspace**: the restricted gspace

                - **back_map**: a function mapping an element of the subgroup to itself in the fiber group of the original space

                - **subgroup_map**: a function mapping an element of the fiber group of the original space to itself in the subgroup (returns ``None`` if the element is not in the subgroup)

        """
        pass

    def featurefield_action(self, input: np.ndarray, repr: escnn.group.Representation, element: escnn.group.GroupElement, order: int =2) -> np.ndarray:
        r"""
        
        This method implements the action of the symmetry group on a feature field defined over the basespace of this
        G-space.
        It includes both an action over the basespace (e.g. a rotation of the points on the plane) and a transformation
        of the channels by left-multiplying them with a representation of the fiber group.

        The method takes as input a tensor (``input``), a representation (``repr``) and an ``element`` of the fiber
        group. The tensor ``input`` is the feature field to be transformed and needs to be compatible with this G-space
        and the representation (i.e. its number of channels equals the size of that representation).
        ``element`` needs to belong to the fiber group: check :meth:`escnn.group.Group.is_element`.
        This method returns a transformed tensor through the action of ``element``.

        More precisely, given an input tensor, interpreted as an :math:`c`-dimensional signal
        :math:`f: \R^D \to \mathbb{R}^c` defined over the base space :math:`\R^D`, a representation
        :math:`\rho: G \to \mathbb{R}^{c \times c}` of :math:`G` and an element :math:`g \in G` of the fiber group,
        the method returns the transformed signal :math:`f'` defined as:

        .. math::
            f'(x) := \rho(g) f(g^{-1} x)

        .. note ::

            Mathematically, this method transforms the input with the **induced representation** from the input ``repr``
            (:math:`\rho`) of the symmetry group (:math:`G`) to the *total space* (:math:`P`), i.e.
            with :math:`Ind_{G}^{P} \rho`.
            For more details on this, see
            `General E(2)-Equivariant Steerable CNNs <https://arxiv.org/abs/1911.08251>`_ or
            `A General Theory of Equivariant CNNs On Homogeneous Spaces <https://papers.nips.cc/paper/9114-a-general-theory-of-equivariant-cnns-on-homogeneous-spaces.pdf>`_.

        Args:
            input (~numpy.ndarray): input tensor
            repr (Representation): representation of the fiber group
            element (GroupElement): element of the fiber group

        Returns:
            the transformed tensor

        """
        
        from warnings import warn
        warn('Use `escnn.nn.FieldType.transform()` instead', DeprecationWarning)
        
        assert repr.group == self.fibergroup
        assert element.group == self.fibergroup

        rho = repr(element)
    
        output = np.einsum("oi,bi...->bo...", rho, input)
    
        return self._interpolate_transform_basespace(output, element, order=order)

    @property
    @abstractmethod
    def basespace_action(self) -> escnn.group.Representation:
        r"""

        Defines how the fiber group transforms the base space.

        More precisely, this method defines how an element :math:`g \in G` of the fiber group transforms a point
        :math:`x \in X \cong \R^d` of the base space.
        This action is defined as a :math:`d`-dimensional linear :class:`~escnn.group.Representation` of :math:`G`.
        
        """
        pass

    def _interpolate_transform_basespace(
            self,
            input: np.ndarray,
            element: escnn.group.GroupElement,
            order: int = 2,
    ) -> np.ndarray:
        r"""

        Defines how the fiber group transforms the base space.

        The methods takes a tensor compatible with this space (i.e. whose spatial dimensions are supported by the
        base space) and returns the transformed tensor.

        More precisely, given an input tensor, interpreted as an :math:`n`-dimensional signal
        :math:`f: X \to \mathbb{R}^n` defined over the base space :math:`X`, and an element :math:`g \in G` of the
        fiber group, the methods return the transformed signal :math:`f'` defined as:

        .. math::
            f'(x) := f(g^{-1} x)

        This method is specific of the particular GSpace and defines how :math:`g^{-1}` transforms a point
        :math:`x \in X` of the base space.


        Args:
            input (~numpy.ndarray): input tensor
            element (GroupElement): element of the fiber group

        Returns:
            the transformed tensor

        """
        assert element.group == self.fibergroup
        action = self.basespace_action
        trafo = action(element)
        return linear_transform_array_nd(input, trafo, order=order)

    @property
    def irreps(self) -> List[escnn.group.IrreducibleRepresentation]:
        r"""
        list containing all the already built irreducible representations of the fiber group of this space.

        .. seealso::

            See :attr:`escnn.group.Group.irreps` for more details

        """
        return self.fibergroup.irreps()

    @property
    def representations(self):
        r"""
        Dictionary containing all the already built representations of the fiber group of this space.

        .. seealso::

            See :attr:`escnn.group.Group.representations` for more details

        """
        return self.fibergroup.representations

    @property
    def trivial_repr(self) -> escnn.group.Representation:
        r"""
        The trivial representation of the fiber group of this space.

        .. seealso::

            :attr:`escnn.group.Group.trivial_representation`

        """
        return self.fibergroup.trivial_representation

    def irrep(self, *id) -> escnn.group.IrreducibleRepresentation:
        r"""
        Builds the irreducible representation (:class:`~escnn.group.IrreducibleRepresentation`) of the fiber group
        identified by the input arguments.

        .. seealso::

            This method is a wrapper for :meth:`escnn.group.Group.irrep`. See its documentation for more details.
            Check the documentation of :meth:`~escnn.group.Group.irrep` of the specific fiber group used for more
            information on the valid ``id``.


        Args:
            *id: parameters identifying the irrep.

        """
        return self.fibergroup.irrep(*id)

    @property
    def regular_repr(self) -> escnn.group.Representation:
        r"""
        The regular representation of the fiber group of this space.

        .. seealso::

            :attr:`escnn.group.Group.regular_representation`

        """
        return self.fibergroup.regular_representation

    def quotient_repr(self, subgroup_id) -> escnn.group.Representation:
        r"""
        Builds the quotient representation of the fiber group of this space with respect to the subgroup identified
        by ``subgroup_id``.
        
        Check the :meth:`~escnn.gspaces.GSpace.restrict` method's documentation in the non-abstract subclass used
        for a description of the parameter ``subgroup_id``.

        .. seealso::
            
            See :attr:`escnn.group.Group.quotient_representation` for more details on the representation.
        
        Args:
            subgroup_id: identifier of the subgroup

        """
        return self.fibergroup.quotient_representation(subgroup_id)

    def induced_repr(self, subgroup_id, repr: escnn.group.Representation) -> escnn.group.Representation:
        r"""
        Builds the induced representation of the fiber group of this space from the representation ``repr`` of
        the subgroup identified by ``subgroup_id``.

        Check the :meth:`~escnn.gspaces.GSpace.restrict` method's documentation in the non-abstract subclass used
        for a description of the parameter ``subgroup_id``.

        .. seealso::
            
            See :attr:`escnn.group.Group.induced_representation` for more details on the representation.
        
        Args:
            subgroup_id: identifier of the subgroup
            repr (Representation): the representation of the subgroup to induce

        """
        return self.fibergroup.induced_representation(subgroup_id, repr)

    @property
    def testing_elements(self):
        return self.fibergroup.testing_elements()

    def build_fiber_intertwiner_basis(self,
                                      in_repr: escnn.group.Representation,
                                      out_repr: escnn.group.Representation,
                                      ) -> escnn.kernels.KernelBasis:
        r"""


        Args:
            in_repr (Representation): the input representation
            out_repr (Representation): the output representation

        Returns:
            the analytical basis

        """
    
        assert isinstance(in_repr, escnn.group.Representation)
        assert isinstance(out_repr, escnn.group.Representation)
    
        assert in_repr.group == self.fibergroup
        assert out_repr.group == self.fibergroup
    
        if (in_repr.name, out_repr.name) not in self._fields_intertwiners_basis_memory_fiber_basis:
        
            basis = escnn.kernels.kernels_on_point(in_repr, out_repr)
        
            # store the basis in the dictionary
            self._fields_intertwiners_basis_memory_fiber_basis[(in_repr.name, out_repr.name)] = basis
    
        # return the dictionary with all the basis built
        return self._fields_intertwiners_basis_memory_fiber_basis[(in_repr.name, out_repr.name)]

    def build_kernel_basis(self,
                           in_repr: escnn.group.Representation,
                           out_repr: escnn.group.Representation,
                           sigma: Union[float, List[float]],
                           rings: List[float],
                           **kwargs) -> escnn.kernels.KernelBasis:
        r"""

        Builds a basis for the space of the equivariant kernels with respect to the symmetries described by this
        :class:`~escnn.gspaces.GSpace`.

        A kernel :math:`\kappa` equivariant to a group :math:`G` needs to satisfy the following equivariance constraint:

        .. math::
            \kappa(gx) = \rho_\text{out}(g) \kappa(x) \rho_\text{in}(g)^{-1}  \qquad \forall g \in G, x \in \R^D

        where :math:`\rho_\text{in}` is ``in_repr`` while :math:`\rho_\text{out}` is ``out_repr``.


        Because the equivariance constraints only restrict the angular part of the kernels, any radial profile is
        permitted.
        The basis for the radial profile used here contains rings with different radii (``rings``)
        associated with (possibly different) widths (``sigma``).
        A ring is implemented as a Gaussian function over the radial component, centered at one radius
        (see also :class:`~escnn.kernels.GaussianRadialProfile`).

        .. note ::
            This method is a wrapper for the functions building the bases which are defined in :doc:`escnn.kernels`:

            - :meth:`escnn.kernels.kernels_O2_act_R2`,

            - :meth:`escnn.kernels.kernels_SO2_act_R2`,

            - :meth:`escnn.kernels.kernels_DN_act_R2`,

            - :meth:`escnn.kernels.kernels_CN_act_R2`,

            - :meth:`escnn.kernels.kernels_Flip_act_R2`,

            - :meth:`escnn.kernels.kernels_Trivial_act_R2`


        Args:
            in_repr (Representation): the input representation
            out_repr (Representation): the output representation
            sigma (list or float): parameters controlling the width of each ring of the radial profile.
                    If only one scalar is passed, it is used for all rings
            rings (list): radii of the rings defining the radial profile
            **kwargs: Group-specific keywords arguments for ``_basis_generator`` method

        Returns:
            an instance of :class:`~escnn.kernels.KernelBasis` representing the analytical basis

        """
        
        # TODO - move sigma and rings in kwargs
        
        # TODO - solve incompatibility with subclasses' method signature
        # now different subgroups have different keys, so no point is sharing this piece of code
        # maybe use an 'parse_args(**kwargs)'  abstract method????
        # same for _basis_generator
    
        assert isinstance(in_repr, escnn.group.Representation)
        assert isinstance(out_repr, escnn.group.Representation)
    
        assert in_repr.group == self.fibergroup
        assert out_repr.group == self.fibergroup
    
        if isinstance(sigma, float):
            sigma = [sigma] * len(rings)
    
        assert all([s > 0. for s in sigma])
        assert len(sigma) == len(rings)
    
        # build the key
        key = dict(**kwargs)
        key["sigma"] = tuple(sigma)
        key["rings"] = tuple(rings)
        key = tuple(sorted(key.items()))
    
        if (in_repr.name, out_repr.name) not in self._fields_intertwiners_basis_memory[key]:
            # TODO - we could use a flag in the args to choose whether to store it or not
        
            basis = self._basis_generator(in_repr, out_repr, rings, sigma, **kwargs)
        
            # store the basis in the dictionary
            self._fields_intertwiners_basis_memory[key][(in_repr.name, out_repr.name)] = basis
    
        # return the dictionary with all the basis built for this filter size
        return self._fields_intertwiners_basis_memory[key][(in_repr.name, out_repr.name)]

    @abstractmethod
    def _basis_generator(self,
                         in_repr: escnn.group.Representation,
                         out_repr: escnn.group.Representation,
                         rings: List[float],
                         sigma: List[float],
                         **kwargs):
        pass

    def __repr__(self):
        return self.name
