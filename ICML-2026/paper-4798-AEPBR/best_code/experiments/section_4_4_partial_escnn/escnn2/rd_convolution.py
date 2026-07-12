from abc import ABC, abstractmethod

from escnn.nn import FieldType
from escnn.nn import GeometricTensor
from escnn.group import Representation
from escnn.kernels import KernelBasis
from escnn.gspaces import *

from escnn.nn.modules.equivariant_module import EquivariantModule

from escnn.nn.modules.basismanager import BasisManager
from escnn.nn.modules.basismanager import BlocksBasisExpansion

from .basissampler_blocks import BlocksBasisSampler

from typing import Callable, Union, Tuple, List

import torch
from torch.nn import Parameter
import numpy as np
import math

from functools import partial


__all__ = ["_RdConv"]


class _RdConv(EquivariantModule, ABC):
    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        d: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        padding_mode: str = "zeros",
        groups: int = 1,
        bias: bool = True,
        basis_filter: Callable[[dict], bool] = None,
        recompute: bool = False,
        learnable_eq=False,
        normalise_basis=True,
        layer_id=None,
        **mapping_kwargs
    ):
        r"""

        Abstract class which implements a general G-steerable convolution, mapping between the input and output
        :class:`~escnn.nn.FieldType` s specified by the parameters ``in_type`` and ``out_type``.
        This operation is equivariant under the action of :math:`\R^d\rtimes G` where :math:`G` is the
        :attr:`escnn.nn.FieldType.fibergroup` of ``in_type`` and ``out_type``.
        
        Specifically, let :math:`\rho_\text{in}: G \to \GL{\R^{c_\text{in}}}` and
        :math:`\rho_\text{out}: G \to \GL{\R^{c_\text{out}}}` be the representations specified by the input and output
        field types.
        Then :class:`~escnn.nn._RdConv` guarantees an equivariant mapping
        
        .. math::
            \kappa \star [\mathcal{T}^\text{in}_{g,u} . f] = \mathcal{T}^\text{out}_{g,u} . [\kappa \star f] \qquad\qquad \forall g \in G, u \in \R^d
            
        where the transformation of the input and output fields are given by
 
        .. math::
            [\mathcal{T}^\text{in}_{g,u} . f](x) &= \rho_\text{in}(g)f(g^{-1} (x - u)) \\
            [\mathcal{T}^\text{out}_{g,u} . f](x) &= \rho_\text{out}(g)f(g^{-1} (x - u)) \\

        The equivariance of G-steerable convolutions is guaranteed by restricting the space of convolution kernels to an
        equivariant subspace.
        As proven in `3D Steerable CNNs <https://arxiv.org/abs/1807.02547>`_, this parametrizes the *most general
        equivariant convolutional map* between the input and output fields.

        .. warning ::

            This class implements a *discretized* convolution operator over a discrete grid.
            This means that equivariance to continuous symmetries is *not* perfect.
            In practice, by using sufficiently band-limited filters, the equivariance error introduced by the
            discretization of the filters and the features is contained, but some design choices may have a negative
            effect on the overall equivariance of the architecture.

        During training, in each forward pass the module expands the basis of G-steerable kernels with learned weights
        before performing the convolution.
        When :meth:`~torch.nn.Module.eval()` is called, the filter is built with the current trained weights and stored
        for future reuse such that no overhead of expanding the kernel remains.
        
        .. warning ::
            
            When :meth:`~torch.nn.Module.train()` is called, the attributes :attr:`~escnn.nn.R2Conv.filter` and
            :attr:`~escnn.nn.R2Conv.expanded_bias` are discarded to avoid situations of mismatch with the
            learnable expansion coefficients.
            See also :meth:`escnn.nn._RdConv.train`.
            
            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of the class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.
 
        
        Args:
            in_type (FieldType): the type of the input field, specifying its transformation law
            out_type (FieldType): the type of the output field, specifying its transformation law
            d (int): dimensionality of the base space (2 for images, 3 for volumes)
            kernel_size (int): the size of the (square) filter
            padding(int, optional): implicit zero paddings on both sides of the input. Default: ``0``
            padding_mode(str, optional): ``zeros``, ``reflect``, ``replicate`` or ``circular``. Default: ``zeros``
            stride(int, optional): the stride of the kernel. Default: ``1``
            dilation(int, optional): the spacing between kernel elements. Default: ``1``
            groups (int, optional): number of blocked connections from input channels to output channels.
                                    It allows depthwise convolution. When used, the input and output types need to be
                                    divisible in ``groups`` groups, all equal to each other.
                                    Default: ``1``.
            bias (bool, optional): Whether to add a bias to the output (only to fields which contain a
                    trivial irrep) or not. Default ``True``
            basis_filter (callable, optional): filter for the basis elements. Should take a dictionary containing an
                                               element's attributes and return whether to keep it or not.
            recompute (bool, optional): if ``True``, recomputes a new basis for the equivariant kernels.
                    By Default (``False``), it  caches the basis built or reuse a cached one, if it is found.
            learnable_eq (bool, optional): Allows for learnable degree of equivariance (https://arxiv.org/abs/2406.03946).
                    Default: ``False``
            normalise_basis (bool, optional): Normalise the basis expansion to improve stability. Only applicable for
                    learnable equivariance. Default: ``True``.
            layer_id (int | None): identification of the learnable degree of equivariance. Layers with the same id share a single
                    degree of equivariance. If set to None then a unique id is assigned automatically. Default: ``None``,
             **mapping_kwargs: Additional arguments for the partially equivariant mapping.
                    
        Attributes:
            
            ~.weights (torch.Tensor): the learnable parameters which are used to expand the kernel
            ~.filter (torch.Tensor): the convolutional kernel obtained by expanding the parameters
                                    in :attr:`~escnn.nn.R2Conv.weights`
            ~.bias (torch.Tensor): the learnable parameters which are used to expand the bias, if ``bias=True``
            ~.expanded_bias (torch.Tensor): the equivariant bias which is summed to the output, obtained by expanding
                                    the parameters in :attr:`~escnn.nn.R2Conv.bias`
        
        """
        assert in_type.gspace == out_type.gspace
        assert isinstance(in_type.gspace, GSpace)
        assert d >= in_type.gspace.dimensionality

        super(_RdConv, self).__init__()

        self.d = d
        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = out_type

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.groups = groups
        if layer_id is None:
            if self._layer_ids:
                layer_id = max(self._layer_ids) + 1
            else:
                layer_id = 0

        self.layer_id = layer_id
        self._layer_ids.add(layer_id)

        if isinstance(padding, tuple) and len(padding) == self.d:
            _padding = padding
        elif isinstance(padding, int):
            _padding = (padding,) * self.d
        else:
            raise ValueError(
                "padding needs to be either an integer or a tuple containing {} integers but {} found".format(
                    self.d, padding
                )
            )

        padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in padding_modes:
            raise ValueError(
                "padding_mode must be one of [{}], but got padding_mode='{}'".format(
                    padding_modes, padding_mode
                )
            )
        self._reversed_padding_repeated_twice = tuple(
            x for x in reversed(_padding) for _ in range(2)
        )

        if groups > 1:
            # Check the input and output classes can be split in `groups` groups, all equal to each other
            # first, check that the number of fields is divisible by `groups`
            assert len(in_type) % groups == 0
            assert len(out_type) % groups == 0
            in_size = len(in_type) // groups
            out_size = len(out_type) // groups

            # then, check that all groups are equal to each other, i.e. have the same types in the same order
            assert all(
                in_type.representations[i] == in_type.representations[i % in_size]
                for i in range(len(in_type))
            )
            assert all(
                out_type.representations[i] == out_type.representations[i % out_size]
                for i in range(len(out_type))
            )

            # finally, retrieve the type associated to a single group in input.
            # this type will be used to build a smaller kernel basis and a smaller filter
            # as in PyTorch, to build a filter for grouped convolution, we build a filter which maps from one input
            # group to all output groups. Then, PyTorch's standard convolution routine interpret this filter as `groups`
            # different filters, each mapping an input group to an output group.
            in_type = in_type.index_select(list(range(in_size)))

        if bias:
            # bias can be applied only to trivial irreps inside the representation
            # to apply bias to a field we learn a bias for each trivial irreps it contains
            # and, then, we transform it with the change of basis matrix to be able to apply it to the whole field
            # this is equivalent to transform the field to its irreps through the inverse change of basis,
            # sum the bias only to the trivial irrep and then map it back with the change of basis

            # count the number of trivial irreps
            trivials = 0
            for r in self.out_type:
                for irr in r.irreps:
                    if self.out_type.fibergroup.irrep(*irr).is_trivial():
                        trivials += 1

            # if there is at least 1 trivial irrep
            if trivials > 0:
                # matrix containing the columns of the change of basis which map from the trivial irreps to the
                # field representations. This matrix allows us to map the bias defined only over the trivial irreps
                # to a bias for the whole field more efficiently
                bias_expansion = torch.zeros(self.out_type.size, trivials)

                p, c = 0, 0
                for r in self.out_type:
                    pi = 0
                    for irr in r.irreps:
                        irr = self.out_type.fibergroup.irrep(*irr)
                        if irr.is_trivial():
                            bias_expansion[p : p + r.size, c] = torch.tensor(
                                r.change_of_basis[:, pi]
                            )
                            c += 1
                        pi += irr.size
                    p += r.size

                self.register_buffer("bias_expansion", bias_expansion)
                self.bias = Parameter(torch.zeros(trivials), requires_grad=True)
                self.register_buffer("expanded_bias", torch.zeros(out_type.size))
            else:
                self.bias = None
                self.expanded_bias = None
        else:
            self.bias = None
            self.expanded_bias = None

        # compute the coordinates of the centers of the cells in the grid where the filter is sampled
        self.grid = get_grid_coords(d, kernel_size, dilation)

        # note that `in_type` is used instead of `self.in_type` such that it works also when `groups > 1`

        # BlocksBasisExpansion: submodule which takes care of building the filter
        if not learnable_eq:
            self._basisexpansion = BlocksBasisExpansion(
                in_type.representations,
                out_type.representations,
                self._build_kernel_basis,
                self.grid,
                basis_filter=basis_filter,
                recompute=recompute,
            )
        # Learn equivariance
        else:
            self.grid = torch.from_numpy(self.grid).type(torch.float32)
            self._basisexpansion = BlocksBasisSampler(
                in_type.representations,
                out_type.representations,
                partial(
                    self._build_kernel_basis,
                    learnable=True,
                    layer_id=self.layer_id,
                    **mapping_kwargs
                ),
                basis_filter=basis_filter,
                recompute=True,
            )
            # Store normalization factors for basissamplers
            if normalise_basis:
                self._basisexpansion.normalize(self.grid)

        if self.basisexpansion.dimension() == 0:
            raise ValueError(
                """
                The basis for the steerable filter is empty!
                Tune the `frequencies_cutoff`, `kernel_size`, `rings`, `sigma` or `basis_filter` parameters to allow
                for a larger basis.
            """
            )

        self.weights = Parameter(
            torch.zeros(self.basisexpansion.dimension()), requires_grad=True
        )

        filter_size = (out_type.size, in_type.size) + (kernel_size,) * d
        self.register_buffer("filter", torch.zeros(*filter_size))

    _layer_ids = set()

    @abstractmethod
    def _build_kernel_basis(
        self, in_repr: Representation, out_repr: Representation
    ) -> KernelBasis:
        raise NotImplementedError

    @property
    def basisexpansion(self) -> BlocksBasisExpansion:
        r"""
        Submodule which takes care of building the filter.

        It uses the learnt ``weights`` to expand a basis and returns a filter in the usual form used by conventional
        convolutional modules.
        It uses the learned ``weights`` to expand the kernel in the G-steerable basis and returns it in the shape
        :math:`(c_\text{out}, c_\text{in}, s^d)`, where :math:`s` is the ``kernel_size`` and :math:`d` is the
        dimensionality of the base space.

        """
        return self._basisexpansion

    def expand_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Expand the filter in terms of the :attr:`~escnn.nn._RdConv.weights` and the
        expanded bias in terms of :class:`~escnn.nn._RdConv.bias`.

        Returns:
            the expanded filter and bias

        """
        if isinstance(self.basisexpansion, BlocksBasisExpansion):
            _filter = self.basisexpansion(self.weights)
            _filter = _filter.reshape(
                _filter.shape[0], _filter.shape[1], *(self.kernel_size,) * self.d
            )

        elif isinstance(self.basisexpansion, BlocksBasisSampler):
            self.grid = self.grid.type(self.weights.dtype).to(self.weights.device)
            _filter = self.basisexpansion(self.weights, self.grid).permute(1, 2, 0)

            _filter = _filter.reshape(
                _filter.shape[0], _filter.shape[1], *(self.kernel_size,) * self.d
            )

        if self.bias is None:
            _bias = None
        else:
            _bias = self.bias_expansion @ self.bias

        return _filter, _bias

    @abstractmethod
    def forward(self, input: GeometricTensor):
        r"""
        Convolve the input with the expanded filter and bias.

        Args:
            input (GeometricTensor): input feature field transforming according to ``in_type``

        Returns:
            output feature field transforming according to ``out_type``

        """
        pass

    def train(self, mode=True):
        r"""

        If ``mode=True``, the method sets the module in training mode and discards the :attr:`~escnn.nn._RdConv.filter`
        and :attr:`~escnn.nn._RdConv.expanded_bias` attributes.

        If ``mode=False``, it sets the module in evaluation mode. Moreover, the method builds the filter and the bias
        using the current values of the trainable parameters and store them in :attr:`~escnn.nn._RdConv.filter` and
        :attr:`~escnn.nn._RdConv.expanded_bias` such that they are not recomputed at each forward pass.

        .. warning ::

            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of this class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.

        Args:
            mode (bool, optional): whether to set training mode (``True``) or evaluation mode (``False``).
                                   Default: ``True``.

        """

        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
            if hasattr(self, "expanded_bias"):
                del self.expanded_bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`

            _filter, _bias = self.expand_parameters()

            self.register_buffer("filter", _filter)
            if _bias is not None:
                self.register_buffer("expanded_bias", _bias)
            else:
                self.expanded_bias = None

        return super(_RdConv, self).train(mode)

    def evaluate_output_shape(self, input_shape: Tuple) -> Tuple:
        assert len(input_shape) == 2 + self.d
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        w = input_shape[2:]

        wo = [None] * self.d
        for i in range(self.d):
            wo[i] = math.floor(
                (w[i] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                / self.stride
                + 1
            )

        return (b, self.out_type.size) + tuple(wo)

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        main_str = self._get_name() + "("
        if len(extra_lines) == 1:
            main_str += extra_lines[0]
        else:
            main_str += "\n  " + "\n  ".join(extra_lines) + "\n"

        main_str += ")"
        return main_str

    def extra_repr(self):
        s = "{in_type}, {out_type}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != 0 and self.padding != (0,) * self.d:
            s += ", padding={padding}"
        if self.dilation != 1 and self.dilation != (1,) * self.d:
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


def get_grid_coords(d: int, kernel_size: int, dilation: int = 1) -> np.ndarray:
    actual_size = dilation * (kernel_size - 1) + 1

    origin = actual_size / 2 - 0.5

    points = np.empty((kernel_size**d, d))

    for i in range(kernel_size**d):
        for j in range(d):
            points[i, j] = (i // (kernel_size**j)) % kernel_size
            points[i, j] *= dilation

            # center the origin
            points[i, j] -= origin

            if j >= 1:
                # invert Y and Z coordinates
                # TODO : should this hold also for other coordinates in R^d, d > 3?
                points[i, j] *= -1

    return points
