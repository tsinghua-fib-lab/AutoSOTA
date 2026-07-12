from torch.nn.functional import conv2d, pad

from escnn.nn import FieldType
from escnn.nn import GeometricTensor
import escnn.nn

from escnn.group import Representation, Group
from escnn.kernels import KernelBasis
from .gspaces import GSpace2D

from .rd_convolution import _RdConv

from typing import Callable, Union, List

import torch
import numpy as np
import math


__all__ = ["R2Conv"]


class R2Conv(_RdConv):
    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        padding_mode: str = "zeros",
        groups: int = 1,
        bias: bool = True,
        sigma: Union[List[float], float] = None,
        frequencies_cutoff: Union[float, Callable[[float], int]] = None,
        rings: List[float] = None,
        maximum_offset: int = None,
        recompute: bool = False,
        basis_filter: Callable[[dict], bool] = None,
        initialize: bool = True,
        learnable_eq=False,
        normalise_basis=True,
        layer_id=None,
        **mapping_kwargs
    ):
        r"""
        
        
        G-steerable planar convolution mapping between the input and output :class:`~escnn.nn.FieldType` s specified by
        the parameters ``in_type`` and ``out_type``.
        This operation is equivariant under the action of :math:`\R^2\rtimes G` where :math:`G` is the
        :attr:`escnn.nn.FieldType.fibergroup` of ``in_type`` and ``out_type``.
        
        Specifically, let :math:`\rho_\text{in}: G \to \GL{\R^{c_\text{in}}}` and
        :math:`\rho_\text{out}: G \to \GL{\R^{c_\text{out}}}` be the representations specified by the input and output
        field types.
        Then :class:`~escnn.nn.R2Conv` guarantees an equivariant mapping
        
        .. math::
            \kappa \star [\mathcal{T}^\text{in}_{g,u} . f] = \mathcal{T}^\text{out}_{g,u} . [\kappa \star f] \qquad\qquad \forall g \in G, u \in \R^2
            
        where the transformation of the input and output fields are given by
 
        .. math::
            [\mathcal{T}^\text{in}_{g,u} . f](x) &= \rho_\text{in}(g)f(g^{-1} (x - u)) \\
            [\mathcal{T}^\text{out}_{g,u} . f](x) &= \rho_\text{out}(g)f(g^{-1} (x - u)) \\

        The equivariance of G-steerable convolutions is guaranteed by restricting the space of convolution kernels to an
        equivariant subspace.
        As proven in `3D Steerable CNNs <https://arxiv.org/abs/1807.02547>`_, this parametrizes the *most general
        equivariant convolutional map* between the input and output fields.
        For feature fields on :math:`\R^2` (e.g. images), the complete G-steerable kernel spaces for :math:`G \leq \O2`
        is derived in `General E(2)-Equivariant Steerable CNNs <https://arxiv.org/abs/1911.08251>`_.

        .. warning ::

            This class implements a *discretized* convolution operator over a discrete grid.
            This means that equivariance to continuous symmetries is *not* perfect.
            In practice, by using sufficiently band-limited filters, the equivariance error introduced by the
            discretization of the filters and the features is contained, but some design choices may have a negative
            effect on the overall equivariance of the architecture.

        During training, in each forward pass the module expands the basis of G-steerable kernels with learned weights
        before calling :func:`torch.nn.functional.conv2d`.
        When :meth:`~torch.nn.Module.eval()` is called, the filter is built with the current trained weights and stored
        for future reuse such that no overhead of expanding the kernel remains.
        
        .. warning ::
            
            When :meth:`~torch.nn.Module.train()` is called, the attributes :attr:`~escnn.nn.R2Conv.filter` and
            :attr:`~escnn.nn.R2Conv.expanded_bias` are discarded to avoid situations of mismatch with the
            learnable expansion coefficients.
            See also :meth:`escnn.nn.R2Conv.train`.
            
            This behaviour can cause problems when storing the :meth:`~torch.nn.Module.state_dict` of a model while in
            a mode and lately loading it in a model with a different mode, as the attributes of the class change.
            To avoid this issue, we recommend converting the model to eval mode before storing or loading the state
            dictionary.
 
 
        The learnable expansion coefficients of the this module can be initialized with the methods in
        :mod:`escnn.nn.init`.
        By default, the weights are initialized in the constructors using :func:`~escnn.nn.init.generalized_he_init`.
        
        .. warning ::
            
            This initialization procedure can be extremely slow for wide layers.
            In case initializing the model is not required (e.g. before loading the state dict of a pre-trained model)
            or another initialization method is preferred (e.g. :func:`~escnn.nn.init.deltaorthonormal_init`), the
            parameter ``initialize`` can be set to ``False`` to avoid unnecessary overhead.
        
        
        The parameters ``basisexpansion``, ``sigma``, ``frequencies_cutoff``, ``rings`` and ``maximum_offset`` are
        optional parameters used to control how the basis for the filters is built, how it is sampled on the filter
        grid and how it is expanded to build the filter. We suggest to keep these default values.
        
        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.
        
        Args:
            in_type (FieldType): the type of the input field, specifying its transformation law
            out_type (FieldType): the type of the output field, specifying its transformation law
            kernel_size (int): the size of the (square) filter
            padding(int, optional): implicit zero paddings on both sides of the input. Default: ``0``
            stride(int, optional): the stride of the kernel. Default: ``1``
            dilation(int, optional): the spacing between kernel elements. Default: ``1``
            padding_mode(str, optional): ``zeros``, ``reflect``, ``replicate`` or ``circular``. Default: ``zeros``
            groups (int, optional): number of blocked connections from input channels to output channels.
                                    It allows depthwise convolution. When used, the input and output types need to be
                                    divisible in ``groups`` groups, all equal to each other.
                                    Default: ``1``.
            bias (bool, optional): Whether to add a bias to the output (only to fields which contain a
                    trivial irrep) or not. Default ``True``
            sigma (list or float, optional): width of each ring where the bases are sampled. If only one scalar
                    is passed, it is used for all rings.
            frequencies_cutoff (callable or float, optional): function mapping the radii of the basis elements to the
                    maximum frequency accepted. If a float values is passed, the maximum frequency is equal to the
                    radius times this factor. By default (``None``), a more complex policy is used.
            rings (list, optional): radii of the rings where to sample the bases
            maximum_offset (int, optional): number of additional (aliased) frequencies in the intertwiners for finite
                    groups. By default (``None``), all additional frequencies allowed by the frequencies cut-off
                    are used.
            recompute (bool, optional): if ``True``, recomputes a new basis for the equivariant kernels.
                    By Default (``False``), it  caches the basis built or reuse a cached one, if it is found.
            basis_filter (callable, optional): function which takes as input a descriptor of a basis element
                    (as a dictionary) and returns a boolean value: whether to preserve (``True``) or discard (``False``)
                    the basis element. By default (``None``), no filtering is applied.
            initialize (bool, optional): initialize the weights of the model. Default: ``True``
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

        assert isinstance(in_type.gspace, GSpace2D)
        assert isinstance(out_type.gspace, GSpace2D)

        (
            basis_filter,
            self._rings,
            self._sigma,
            self._maximum_frequency,
        ) = compute_basis_params(
            kernel_size, frequencies_cutoff, rings, sigma, dilation, basis_filter
        )

        super(R2Conv, self).__init__(
            in_type,
            out_type,
            2,
            kernel_size,
            padding,
            stride,
            dilation,
            padding_mode,
            groups,
            bias,
            basis_filter,
            recompute,
            learnable_eq=learnable_eq,
            normalise_basis=normalise_basis,
            layer_id=layer_id,
            **mapping_kwargs
        )
        # self.layer_id = len(self._layer_ids)
        # self._layer_ids.append(self.layer_id)
        if initialize:
            # by default, the weights are initialized with a generalized form of He's weight initialization
            escnn.nn.init.generalized_he_init(self.weights.data, self.basisexpansion)

    # _layer_ids = []

    def _build_kernel_basis(
        self,
        in_repr: Representation,
        out_repr: Representation,
        learnable: bool = False,
        layer_id: int = None,
        **mapping_kwargs
    ) -> KernelBasis:
        return self.space.build_kernel_basis(
            in_repr,
            out_repr,
            self._sigma,
            self._rings,
            maximum_frequency=self._maximum_frequency,
            learnable=learnable,
            layer_id=layer_id,
            **mapping_kwargs
        )

    def forward(self, input: GeometricTensor):
        r"""
        Convolve the input with the expanded filter and bias.

        Args:
            input (GeometricTensor): input feature field transforming according to ``in_type``

        Returns:
            output feature field transforming according to ``out_type``

        """

        assert input.type == self.in_type
        if not self.training:
            _filter = self.filter
            _bias = self.expanded_bias
        else:
            # retrieve the filter and the bias
            _filter, _bias = self.expand_parameters()

        # use it for convolution and return the result
        if self.padding_mode == "zeros":
            output = conv2d(
                input.tensor,
                _filter,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=_bias,
            )
        else:
            output = conv2d(
                pad(
                    input.tensor,
                    self._reversed_padding_repeated_twice,
                    self.padding_mode,
                ),
                _filter,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
                bias=_bias,
            )

        return GeometricTensor(output, self.out_type, coords=None)

    def check_equivariance(
        self,
        atol: float = 0.1,
        rtol: float = 0.1,
        assertion: bool = True,
        verbose: bool = True,
    ):
        # np.set_printoptions(precision=5, threshold=30 *self.in_type.size**2, suppress=False, linewidth=30 *self.in_type.size**2)

        feature_map_size = 33
        last_downsampling = 5
        first_downsampling = 5

        initial_size = (
            feature_map_size * last_downsampling - 1 + self.kernel_size
        ) * first_downsampling

        c = self.in_type.size

        import matplotlib.image as mpimg
        from skimage.measure import block_reduce
        from skimage.transform import resize

        # x = mpimg.imread('../group/testimage.jpeg').transpose((2, 0, 1))[np.newaxis, 0:c, :, :]
        import scipy

        try:
            x = scipy.datasets.face().transpose((2, 0, 1))[np.newaxis, 0:c, :, :]
        except AttributeError:
            x = scipy.misc.face().transpose((2, 0, 1))[np.newaxis, 0:c, :, :]

        x = resize(
            x, (x.shape[0], x.shape[1], initial_size, initial_size), anti_aliasing=True
        )

        x = x / 255.0 - 0.5

        if x.shape[1] < c:
            to_stack = [x for i in range(c // x.shape[1])]
            if c % x.shape[1] > 0:
                to_stack += [x[:, : (c % x.shape[1]), ...]]

            x = np.concatenate(to_stack, axis=1)

        x = GeometricTensor(torch.FloatTensor(x), self.in_type)

        def shrink(t: GeometricTensor, s) -> GeometricTensor:
            return GeometricTensor(
                torch.FloatTensor(
                    block_reduce(t.tensor.detach().numpy(), s, func=np.mean)
                ),
                t.type,
            )

        errors = []

        for el in self.space.testing_elements:
            out1 = self(shrink(x, (1, 1, 5, 5))).transform(el).tensor.detach().numpy()
            out2 = self(shrink(x.transform(el), (1, 1, 5, 5))).tensor.detach().numpy()

            out1 = block_reduce(out1, (1, 1, 5, 5), func=np.mean)
            out2 = block_reduce(out2, (1, 1, 5, 5), func=np.mean)

            b, c, h, w = out2.shape

            center_mask = np.zeros((2, h, w))
            center_mask[1, :, :] = np.arange(0, w) - w / 2
            center_mask[0, :, :] = np.arange(0, h) - h / 2
            center_mask[0, :, :] = center_mask[0, :, :].T
            center_mask = (
                center_mask[0, :, :] ** 2 + center_mask[1, :, :] ** 2 < (h / 4) ** 2
            )

            out1 = out1[..., center_mask]
            out2 = out2[..., center_mask]

            out1 = out1.reshape(-1)
            out2 = out2.reshape(-1)

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

        # init.deltaorthonormal_init(self.weights.data, self.basisexpansion)
        # filter = self.basisexpansion()
        # center = self.s // 2
        # filter = filter[..., center, center]
        # assert torch.allclose(torch.eye(filter.shape[1]), filter.t() @ filter, atol=3e-7)

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.Conv2d` module and set to "eval" mode.

        """

        # set to eval mode so the filter and the bias are updated with the current
        # values of the weights
        self.eval()
        _filter = self.filter
        _bias = self.expanded_bias

        # build the PyTorch Conv2d module
        has_bias = self.bias is not None
        conv = torch.nn.Conv2d(
            self.in_type.size,
            self.out_type.size,
            self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            bias=has_bias,
        )

        # set the filter and the bias
        conv.weight.data = _filter.data
        if has_bias:
            conv.bias.data = _bias.data

        return conv


def bandlimiting_filter(
    frequency_cutoff: Union[float, Callable[[float], float]]
) -> Callable[[dict], bool]:
    r"""

    Returns a method which takes as input the attributes (as a dictionary) of a basis element and returns a boolean
    value: whether to preserve that element (true) or not (false)

    If the parameter ``frequency_cutoff`` is a scalar value, the maximum frequency allowed at a certain radius is
    proportional to the radius itself. in thi case, the parameter ``frequency_cutoff`` is the factor controlling this
    proportionality relation.

    If the parameter ``frequency_cutoff`` is a callable, it needs to take as input a radius (a scalar value) and return
    the maximum frequency which can be sampled at that radius.

    args:
        frequency_cutoff (float): factor controlling the bandlimiting

    returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """

    if isinstance(frequency_cutoff, float):
        frequency_cutoff = lambda r, fco=frequency_cutoff: r * frequency_cutoff

    def bl_filter(attributes: dict) -> bool:
        return math.fabs(attributes["irrep:frequency"]) <= frequency_cutoff(
            attributes["radius"]
        )

    return bl_filter


def compute_basis_params(
    kernel_size: int,
    frequencies_cutoff: Union[float, Callable[[float], float]] = None,
    rings: List[float] = None,
    sigma: List[float] = None,
    dilation: int = 1,
    custom_basis_filter: Callable[[dict], bool] = None,
):
    width = dilation * (kernel_size - 1) / 2
    max_radius = width * np.sqrt(2)

    # by default, the number of rings equals half of the filter size
    if rings is None:
        n_rings = math.ceil(kernel_size / 2)
        rings = torch.linspace(0, (kernel_size - 1) // 2, n_rings) * dilation
        rings = rings.tolist()

    assert all([max_radius >= r >= 0 for r in rings])

    if sigma is None:
        sigma = [0.6] * (len(rings) - 1) + [0.4]
        for i, r in enumerate(rings):
            if r == 0.0:
                sigma[i] = 0.005

    elif isinstance(sigma, float):
        sigma = [sigma] * len(rings)

    if frequencies_cutoff is None:
        frequencies_cutoff = -1.0

    if isinstance(frequencies_cutoff, float):
        if frequencies_cutoff == -3:
            frequencies_cutoff = _manual_fco3(kernel_size // 2)
        elif frequencies_cutoff == -2:
            frequencies_cutoff = _manual_fco2(kernel_size // 2)
        elif frequencies_cutoff == -1:
            frequencies_cutoff = _manual_fco1(kernel_size // 2)
        else:
            frequencies_cutoff = lambda r, fco=frequencies_cutoff: fco * r

    # check if the object is a callable function
    assert callable(frequencies_cutoff)

    maximum_frequency = int(max(frequencies_cutoff(r) for r in rings))

    fco_filter = bandlimiting_filter(frequencies_cutoff)

    if custom_basis_filter is not None:
        basis_filter = (
            lambda d, custom_basis_filter=custom_basis_filter, fco_filter=fco_filter: (
                custom_basis_filter(d) and fco_filter(d)
            )
        )
    else:
        basis_filter = fco_filter

    return basis_filter, rings, sigma, maximum_frequency


def _manual_fco3(max_radius: float) -> Callable[[float], float]:
    r"""

    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.

    Args:
        max_radius (float): radius of the last ring touching the border of the grid

    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """

    def bl_filter(r: float) -> float:
        max_freq = 0 if r == 0.0 else 1 if r == max_radius else 2
        return max_freq

    return bl_filter


def _manual_fco2(max_radius: float) -> Callable[[float], float]:
    r"""

    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.

    Args:
        max_radius (float): radius of the last ring touching the border of the grid

    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """

    def bl_filter(r: float) -> float:
        max_freq = (
            0 if r == 0.0 else min(2 * r, 1 if r == max_radius else 2 * r - (r + 1) % 2)
        )
        return max_freq

    return bl_filter


def _manual_fco1(max_radius: float) -> Callable[[float], float]:
    r"""

    Returns a method which takes as input the radius of a ring and returns the maximum frequency which can be sampled
    on that ring.

    Args:
        max_radius (float): radius of the last ring touching the border of the grid

    Returns:
        a function which checks the attributes of individual basis elements and chooses whether to discard them or not

    """

    def bl_filter(r: float) -> float:
        max_freq = (
            0 if r == 0.0 else min(2 * r, 2 if r == max_radius else 2 * r - (r + 1) % 2)
        )
        return max_freq

    return bl_filter
