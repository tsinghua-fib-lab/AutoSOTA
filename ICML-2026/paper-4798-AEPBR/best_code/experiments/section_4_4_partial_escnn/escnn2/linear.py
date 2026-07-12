from escnn.gspaces import GSpace0D
from escnn.nn import init
from escnn.nn import FieldType
from escnn.nn import GeometricTensor
from escnn.nn.modules import EquivariantModule

from escnn.nn.modules.basismanager import BasisManager
from escnn.nn.modules.basismanager import BlocksBasisExpansion  # , BlocksBasisSampler

from .basissampler_blocks import BlocksBasisSampler

from .kernels_on_point import kernels_on_point

from torch.nn import Parameter
from torch.nn.functional import linear
import torch
import numpy as np
from functools import partial

from typing import Tuple


__all__ = ["Linear"]


class Linear(EquivariantModule):
    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        bias: bool = True,
        basisexpansion: str = "blocks",
        recompute: bool = False,
        initialize: bool = True,
        learnable_eq=False,
        prelim=False,
        normalise_basis=True,
        layer_id=None,
        **mapping_kwargs
    ):
        r"""


        G-equivariant linear transformation mapping between the input and output :class:`~escnn.nn.FieldType` s
        specified by the parameters ``in_type`` and ``out_type``.
        This operation is equivariant under the action of the :attr:`escnn.nn.FieldType.fibergroup` :math:`G` of
        ``in_type`` and ``out_type``.

        Specifically, let :math:`\rho_\text{in}: G \to \GL{\R^{c_\text{in}}}` and
        :math:`\rho_\text{out}: G \to \GL{\R^{c_\text{out}}}` be the representations specified by the input and output
        field types.
        Then :class:`~escnn.nn.Linear` guarantees an equivariant mapping

        .. math::
            W \rho_\text{in}(g) v = \rho_\text{out}(g) W v \qquad\qquad \forall g \in G, u \in \R^{c_\text{in}}

        where :math:`\rho_\text{in}` and :math:`\rho_\text{out}` are the :math:`G`-representations associated with
        ``in_type`` and ``out_type``.
        The equivariance of a G-equivariant linear layer is guaranteed by restricting the space of weight matrices to an
        equivariant subspace.
        During training, in each forward pass the module expands the basis of G-equivariant matrices with learned weights
        before performing the linear trasformation.
        When :meth:`~torch.nn.Module.eval()` is called, the matrix is built with the current trained weights and stored
        for future reuse such that no overhead of expanding the matrix remains.

        .. warning ::

            When :meth:`~torch.nn.Module.train()` is called, the attributes :attr:`~escnn.nn.Linear.matrix` and
            :attr:`~escnn.nn.Linear.expanded_bias` are discarded to avoid situations of mismatch with the
            learnable expansion coefficients.
            See also :meth:`escnn.nn.Linear.train`.

            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of the class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.

        .. warning ::
            To ensure compatibility with both :class:`torch.nn.Linear` and :class:`~escnn.nn.GeometricTensor`, this
            module supports only input tensors with two dimensions ``(batch_size, number_features)``.


        The learnable expansion coefficients of the this module can be initialized with the methods in
        :mod:`escnn.nn.init`.
        By default, the weights are initialized in the constructors using :func:`~escnn.nn.init.generalized_he_init`.

        .. warning ::

            This initialization procedure can be extremely slow for wide layers.
            In case initializing the model is not required (e.g. before loading the state dict of a pre-trained model)
            or another initialization method is preferred (e.g. :func:`~escnn.nn.init.deltaorthonormal_init`), the
            parameter ``initialize`` can be set to ``False`` to avoid unnecessary overhead.


        Args:
            in_type (FieldType): the type of the input field, specifying its transformation law
            out_type (FieldType): the type of the output field, specifying its transformation law
            bias (bool, optional): Whether to add a bias to the output (only to fields which contain a
                    trivial irrep) or not. Default ``True``
            basisexpansion (str, optional): the basis expansion algorithm to use. You can ignore this attribute.
            recompute (bool, optional): if ``True``, recomputes a new basis for the equivariant kernels.
                    By Default (``False``), it  caches the basis built or reuse a cached one, if it is found.
            initialize (bool, optional): initialize the weights of the model. Default: ``True``
            learnable_eq (bool, optional): Allows for learnable degree of equivariance (https://arxiv.org/abs/2406.03946).
                    Default: ``False``
            prelim (bool, optional): Uses a preliminary approach (Appendix F.2, https://arxiv.org/abs/2406.03946) if True.
                    Default: ``False``,
            normalise_basis (bool, optional): Normalise the basis expansion to improve stability. Only applicable for
                    learnable equivariance. Default: ``True``.
            layer_id (int | None): identification of the learnable degree of equivariance. Layers with the same id share a single
                    degree of equivariance. If set to None then a unique id is assigned automatically. Default: ``None``,
            **mapping_kwargs: Additional arguments for the partially equivariant mapping.

        Attributes:

            ~.weights (torch.Tensor): the learnable parameters which are used to expand the matrix
            ~.matrix (torch.Tensor): the matrix obtained by expanding the parameters in :attr:`~escnn.nn.Linear.weights`
            ~.bias (torch.Tensor): the learnable parameters which are used to expand the bias, if ``bias=True``
            ~.expanded_bias (torch.Tensor): the equivariant bias which is summed to the output, obtained by expanding
                                    the parameters in :attr:`~escnn.nn.Linear.bias`

        """

        # only GSpace0D allowed since Linear acts on the last dimension of a tensor
        assert isinstance(in_type.gspace, GSpace0D)

        assert in_type.gspace == out_type.gspace

        super(Linear, self).__init__()

        self.in_type = in_type
        self.out_type = out_type
        self.space = in_type.gspace
        # self.layer_id = layer_id
        if layer_id is None:
            if self._layer_ids:
                layer_id = max(self._layer_ids) + 1
            else:
                layer_id = 0

        self.layer_id = layer_id
        self._layer_ids.add(layer_id)

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

        # BasisExpansion: submodule which takes care of building the matrix
        self._basisexpansion = None

        if basisexpansion == "blocks":
            if not learnable_eq:
                self._basisexpansion = BlocksBasisExpansion(
                    in_type.representations,
                    out_type.representations,
                    self.space.build_fiber_intertwiner_basis,
                    np.zeros((1, 1)),
                    recompute=recompute,
                )
            else:
                self._basisexpansion = BlocksBasisSampler(
                    in_type.representations,
                    out_type.representations,
                    partial(
                        kernels_on_point,
                        prelim=prelim,
                        layer_id=self.layer_id,
                        **mapping_kwargs
                    ),
                    recompute=True,
                )
                if normalise_basis:
                    self._basisexpansion.normalize(torch.zeros(1, 1))

        else:
            raise ValueError(
                'Basis Expansion algorithm "%s" not recognized' % basisexpansion
            )

        self.weights = Parameter(
            torch.zeros(self.basisexpansion.dimension()), requires_grad=True
        )

        filter_size = (out_type.size, in_type.size)
        self.register_buffer("matrix", torch.zeros(*filter_size))
        if initialize:
            # by default, the weights are initialized with a generalized form of He's weight initialization
            init.generalized_he_init(self.weights.data, self.basisexpansion)

    _layer_ids = set()

    def forward(self, input: GeometricTensor):
        r"""
        Convolve the input with the expanded matrix and bias.

        Args:
            input (GeometricTensor): input feature field transforming according to ``in_type``
        Returns:
            output feature field transforming according to ``out_type``

        """

        assert input.type == self.in_type
        # only GSpace0D allowed in practice
        assert len(input.shape) == 2

        if not self.training:
            _matrix = self.matrix
            _bias = self.expanded_bias
        else:
            # retrieve the matrix and the bias
            _matrix, _bias = self.expand_parameters()

        output = linear(input.tensor, _matrix, bias=_bias)

        return GeometricTensor(output, self.out_type, input.coords)

    @property
    def basisexpansion(self) -> BasisManager:
        r"""
        Submodule which takes care of building the matrix.
        It uses the learnt ``weights`` to expand a basis and returns a matrix in the usual form used by conventional
        linear modules.
        It uses the learned ``weights`` to expand the kernel in the G-steerable basis and returns it in the shape
        :math:`(c_\text{out}, c_\text{in})`.
        """
        return self._basisexpansion

    def expand_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Expand the matrix in terms of the :attr:`escnn.nn.Linear.weights` and the
        expanded bias in terms of :class:`escnn.nn.Linear.bias`.
        Returns:
            the expanded matrix and bias
        """
        if isinstance(self.basisexpansion, BlocksBasisExpansion):
            _matrix = self.basisexpansion(self.weights)
            _matrix = _matrix.reshape(_matrix.shape[0], _matrix.shape[1])
        elif isinstance(self.basisexpansion, BlocksBasisSampler):
            _matrix = self.basisexpansion(
                self.weights,
                torch.zeros(1, 1, device=self.weights.device, dtype=self.weights.dtype),
            ).permute(1, 2, 0)
            _matrix = _matrix.reshape(_matrix.shape[0], _matrix.shape[1])

        if self.bias is None:
            _bias = None
        else:
            _bias = self.bias_expansion @ self.bias

        return _matrix, _bias

    def train(self, mode=True):
        r"""
        If ``mode=True``, the method sets the module in training mode and discards the :attr:`~escnn.nn.Linear.matrix`
        and :attr:`~escnn.nn.Linear.expanded_bias` attributes.
        If ``mode=False``, it sets the module in evaluation mode. Moreover, the method builds the matrix and the bias
        using the current values of the trainable parameters and store them in :attr:`~escnn.nn.Linear.matrix` and
        :attr:`~escnn.nn.Linear.expanded_bias` such that they are not recomputed at each forward pass.
        .. warning ::
            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of this class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.
        Args:
            mode (bool, optional): whether to set training mode (``True``) or evaluation mode (``False``).
                                   Default: ``True``.
        """

        if not mode:
            _matrix, _bias = self.expand_parameters()

            self.register_buffer("matrix", _matrix)
            if _bias is not None:
                self.register_buffer("expanded_bias", _bias)
            else:
                self.expanded_bias = None

        else:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "matrix"):
                del self.matrix
            if hasattr(self, "expanded_bias"):
                del self.expanded_bias

        return super(Linear, self).train(mode)

    def evaluate_output_shape(self, input_shape: Tuple) -> Tuple:
        assert len(input_shape) == 2
        assert input_shape[1] == self.in_type.size

        return (input_shape[0], self.out_type.size)

    def export(self) -> torch.nn.Linear:
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.Linear` module and set to "eval" mode.
        """
        # set to eval mode so the matrix and the bias are updated with the current
        # values of the weights

        self.eval()
        _matrix = self.matrix
        _bias = self.expanded_bias

        has_bias = self.bias is not None

        # build the PyTorch module
        linear = torch.nn.Linear(self.in_type.size, self.out_type.size, bias=has_bias)
        linear.weight.data[:] = self

        # set the weights and the bias
        linear.weight.data = _matrix.data
        if has_bias:
            linear.bias.data = _bias.data

        return linear

    def check_equivariance(
        self,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        assertion: bool = True,
        verbose: bool = True,
    ):
        x = torch.randn(10, self.in_type.size)
        x = GeometricTensor(x, self.in_type)

        errors = []

        for el in self.space.testing_elements:
            out1 = self(x).transform_fibers(el).tensor.detach().numpy()
            out2 = self(x.transform_fibers(el)).tensor.detach().numpy()

            errs = np.abs(out1 - out2)

            esum = np.maximum(np.abs(out1), np.abs(out2))
            esum[esum == 0.0] = 1

            relerr = errs / esum

            if verbose:
                print(
                    el,
                    relerr.max(),
                    relerr.mean(),
                    relerr.var(),
                    errs.max(),
                    errs.mean(),
                    errs.var(),
                )

            tol = rtol * esum + atol

            if np.any(errs > tol) and verbose:
                print(out1[errs > tol])
                print(out2[errs > tol])
                print(tol[errs > tol])

            if assertion:
                assert np.all(
                    errs < tol
                ), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(
                    el, errs.max(), errs.mean(), errs.var()
                )

            errors.append((el, errs.mean()))

        return errors


if __name__ == "__main__":
    from escnn import group
    from escnn import gspaces
    from escnn import nn

    G = group.o2_group(maximum_frequency=12)
    act = gspaces.no_base_space(G)
    f_in = nn.FieldType(act, [G.standard_representation()] * 12)
    f_out = nn.FieldType(act, [act.irrep(*irr) for irr in G.bl_irreps(4)] * 24)
    # f_out = nn.FieldType(act, [G.standard_representation()] * 24)
    l = Linear(f_in, f_out, basisexpansion="learn_eq")
    thing = f_in(torch.randn(1, 24))
    out = l(thing)

    # print(l._basisexpansion)
    # l._basisexpansion = BlocksBasisExpansion(
    #     l.in_type.representations,
    #     l.out_type.representations,s
    #     l.space.build_fiber_intertwiner_basis,
    #     np.zeros((1, 1)),
    #     recompute=False,
    # )

    # l.check_equivariance()
