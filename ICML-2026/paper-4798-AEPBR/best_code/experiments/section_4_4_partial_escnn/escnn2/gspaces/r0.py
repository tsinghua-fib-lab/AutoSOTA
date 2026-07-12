
from __future__ import annotations

import escnn.kernels
import escnn.group
from escnn.group import Group
from .gspace import GSpace

from typing import Tuple, Callable

import numpy as np


__all__ = [
    "GSpace0D",
    "no_base_space",
]


class GSpace0D(GSpace):
    
    def __init__(self, G: escnn.group.Group):
        # TODO: is GSpace0D a good name? Maybe RepresentationSpace is better?
        super(GSpace0D, self).__init__(G, 0, G.name)

    def restrict(self, id) -> Tuple[GSpace, Callable, Callable]:
        subgroup, parent, child = self.fibergroup.subgroup(id)
        return GSpace0D(subgroup), parent, child

    @property
    def basespace_action(self) -> escnn.group.Representation:
        return None
    
    def _interpolate_transform_basespace(
            self,
            input: np.ndarray,
            element: escnn.group.GroupElement,
            order: int = 2,
    ) -> np.ndarray:
        assert element.group == self.fibergroup
        return input

    def build_kernel_basis(self,
                           in_repr: escnn.group.Representation,
                           out_repr: escnn.group.Representation,
                           **kwargs) -> escnn.kernels.KernelBasis:
        r"""

        Args:
            in_repr (Representation): the input representation
            out_repr (Representation): the output representation
            **kwargs: Group-specific keywords arguments for ``_basis_generator`` method

        Returns:
            the analytical basis

        """
        assert isinstance(in_repr, escnn.group.Representation)
        assert isinstance(out_repr, escnn.group.Representation)

        assert in_repr.group == self.fibergroup
        assert out_repr.group == self.fibergroup

        # build the key
        key = ()

        if (in_repr.name, out_repr.name) not in self._fields_intertwiners_basis_memory[key]:
    
            basis = self._basis_generator(in_repr, out_repr)
    
            # store the basis in the dictionary
            self._fields_intertwiners_basis_memory[key][(in_repr.name, out_repr.name)] = basis

        # return the dictionary with the basis built
        return self._fields_intertwiners_basis_memory[key][(in_repr.name, out_repr.name)]

    def _basis_generator(self,
                         in_repr: escnn.group.Representation,
                         out_repr: escnn.group.Representation,
                         **kwargs):
        return escnn.kernels.kernels_on_point(in_repr, out_repr)

    def __eq__(self, other):
        if isinstance(other, GSpace0D):
            return self.fibergroup == other.fibergroup
        else:
            return False

    def __hash__(self):
        return hash(self.fibergroup)


def no_base_space(G: Group) -> GSpace0D:
    r"""
    
    Build the :class:`~escnn.gspaces.GSpace` of the input group ``G`` acting on a single point space.
    
    This simple gspace can be useful to describe the features of a ``G``-equivariant MLP.
    
    Args:
        G (Group): a group

    """
    return GSpace0D(G)