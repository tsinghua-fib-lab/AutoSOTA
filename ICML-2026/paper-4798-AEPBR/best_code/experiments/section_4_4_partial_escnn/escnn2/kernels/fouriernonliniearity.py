from escnn.group import *
from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor

from escnn.nn.modules.equivariant_module import EquivariantModule

import torch
import torch.nn.functional as F

from typing import List, Tuple, Any

import numpy as np
from functools import partial

__all__ = ["FourierPointwise", "FourierELU"]


def _build_kernel(G: Group, irrep: List[tuple]):
    kernel = []

    for irr in irrep:
        irr = G.irrep(*irr)

        c = int(irr.size // irr.sum_of_squares_constituents)
        k = irr(G.identity)[:, :c] * np.sqrt(irr.size)
        kernel.append(k.T.reshape(-1))

    kernel = np.concatenate(kernel)
    return kernel


class InvFourier(EquivariantModule):
    """Samples fourier coefficients using Inverse Fourier transform.


    Args:
           gspace (GSpace):  the gspace describing the symmetries of the data. The Fourier transform is
                             performed over the group ```gspace.fibergroup```
           channels (int): number of independent fields in the input `FieldType`
           irreps (list): list of irreps' ids to construct the band-limited representation
           grid (List[GroupElement], default=None): Optional pre-determined grid of points on the group.
           *grid_args: parameters used to construct the discretization grid
           **grid_kwargs: keyword parameters used to construct the discretization grid
    """

    def __init__(
        self,
        gspace: GSpace,
        channels: int,
        irreps: List,
        grid: List[GroupElement] = None,
        *grid_args,
        **grid_kwargs
    ):
        assert isinstance(gspace, GSpace)

        super(InvFourier, self).__init__()
        self.space = gspace

        G: Group = gspace.fibergroup

        self.rho = G.spectral_regular_representation(*irreps, name=None)

        self.in_type = FieldType(self.space, [self.rho] * channels)
        kernel = _build_kernel(G, irreps)
        assert kernel.shape[0] == self.rho.size
        kernel = kernel.reshape(-1, 1)
        if grid is None:
            try:
                grid = G.grid(*grid_args, **grid_kwargs)
            except (NotImplementedError, AssertionError):
                grid = G.elements
        A = np.concatenate([self.rho(g) @ kernel for g in grid], axis=1).T
        self.register_buffer("A", torch.tensor(A, dtype=torch.get_default_dtype()))
        self.grid = grid

    def forward(self, input: GeometricTensor) -> torch.tensor:
        assert input.type == self.in_type

        shape = input.shape
        x_hat = input.tensor.view(
            shape[0], len(self.in_type), self.rho.size, *shape[2:]
        )
        x_hat = x_hat.to(self.A.dtype)
        x_hat = x_hat.to(self.A.device)
        x = torch.einsum("bcf...,gf->bcg...", x_hat, self.A)

        return x

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        pass


class FTSoftmaxNormalisation(EquivariantModule):
    def __init__(
        self,
        gspace: GSpace,
        channels: int,
        irreps: List,
        *grid_args,
        out_irreps: List = None,
        identity_grid: bool = False,
        **grid_kwargs
    ):
        r"""

        Normalises the Fourier coefficients to likelihood distribution over the group under the Haar measure using a
        pointwise softmax normalisation.

        Applies a Inverse Fourier Transform to sample the input features, apply the sofmtax non-linearity in the
        group domain (Dirac-delta basis) and, finally, computes the Fourier Transform to obtain irreps coefficients.
        Also stores variables that are relevant for the computation of KL-divergence and alignment loss.

        .. warning::
            This operation is only *approximately* equivariant and its equivariance depends on the sampling grid and the
            non-linear activation used, as well as the original band-limitation of the input features.

        The same function is applied to every channel independently.
        By default, the input representation is preserved by this operation and, therefore, it equals the output
        representation.
        Optionally, the output can have a different band-limit by using the argument ``out_irreps``.

        The class first constructs a band-limited regular representation of ```gspace.fibergroup``` using
        :meth:`escnn.group.Group.spectral_regular_representation`.
        The band-limitation of this representation is specified by ```irreps``` which should be a list containing
        a list of ids identifying irreps of ```gspace.fibergroup``` (see :attr:`escnn.group.IrreducibleRepresentation.id`).
        This representation is used to define the input and output field types, each containing ```channels``` copies
        of a feature field transforming according to this representation.
        A feature vector transforming according to such representation is interpreted as a vector of coefficients
        parameterizing a function over the group using a band-limited Fourier basis.

        .. note::
            Instead of building the list ``irreps`` manually, most groups implement a method ``bl_irreps()`` which can be
            used to generate this list with through a simpler interface. Check each group's documentation.

        To approximate the Fourier transform, this module uses a finite number of samples from the group.
        The set of samples used is specified by the ```grid_args``` and ```grid_kwargs``` which are forwarded to
        the method :meth:`~escnn.group.Group.grid`.

        Args:
            gspace (GSpace):  the gspace describing the symmetries of the data. The Fourier transform is
                              performed over the group ```gspace.fibergroup```
            channels (int): number of independent fields in the input `FieldType`
            irreps (list): list of irreps' ids to construct the band-limited representation
            *grid_args: parameters used to construct the discretization grid
            out_irreps (list, optional): optionally, one can specify a different band-limiting in output
            identity_grid (bool, default=False): determines if an additional grid is sampled around the identity element
                                                to ensure a more stable alignment loss.
            **grid_kwargs: keyword parameters used to construct the discretization grid

        """

        assert isinstance(gspace, GSpace)

        super(FTSoftmaxNormalisation, self).__init__()

        self.space = gspace

        G: Group = gspace.fibergroup

        self.rho = G.spectral_regular_representation(*irreps, name=None)

        self.in_type = FieldType(self.space, [self.rho] * channels)

        if out_irreps is None:
            # the representation in input is preserved
            self.out_type = self.in_type
            self.rho_out = self.rho
        else:
            self.rho_out = G.spectral_regular_representation(*out_irreps, name=None)
            self.out_type = FieldType(self.space, [self.rho_out] * channels)

        self._function = self.softmax

        kernel = _build_kernel(G, irreps)
        assert kernel.shape[0] == self.rho.size
        kernel = kernel.reshape(-1, 1)

        try:
            grid = G.grid(*grid_args, **grid_kwargs)
        except NotImplementedError:
            grid = G.elements

        # Finds identity element, otherwise adds it to the grid.
        try:
            self.identiy_ind = grid.index(G.identity)
        except ValueError:
            grid = [G.identity] + grid
            self.identiy_ind = 0

        # Samples an additional grid around the identity.
        if identity_grid:
            if isinstance(G, (O3, SO3)):
                identity_Q = G.identity.to("Q")
                pre = None
                if isinstance(identity_Q, tuple):
                    pre, identity_Q = identity_Q
                quats = np.random.normal(0, 0.1, (100, 4)) + identity_Q
                quats /= np.linalg.norm(quats, axis=1)[:, None]
                identity_grid = [
                    G.element(quat, "Q") if pre is None else G.element((pre, quat), "Q")
                    for quat in quats
                ]
                self.identity_grid = identity_grid

                A_identity = np.concatenate(
                    [self.rho(g) @ kernel for g in identity_grid], axis=1
                ).T

                self.register_buffer(
                    "A_identity",
                    torch.tensor(A_identity, dtype=torch.get_default_dtype()),
                )
            elif isinstance(G, (SO2, O2)):
                angles = np.random.normal(0, 0.2, 100)
                self.identity_grid = [
                    (G.element(angle) if isinstance(G, SO2) else G.element((0, angle)))
                    for angle in angles
                ]

                A_identity = np.concatenate(
                    [self.rho(g) @ kernel for g in self.identity_grid], axis=1
                ).T

                self.register_buffer(
                    "A_identity",
                    torch.tensor(A_identity, dtype=torch.get_default_dtype()),
                )

        A = np.concatenate([self.rho(g) @ kernel for g in grid], axis=1).T

        if out_irreps is not None:
            _missing_input_irreps = list(set(irreps).difference(set(out_irreps)))
            rho_out_extended = G.spectral_regular_representation(
                *out_irreps, *_missing_input_irreps, name=None
            )
            kernel_out = _build_kernel(G, out_irreps + _missing_input_irreps)
            assert kernel_out.shape[0] == rho_out_extended.size

            kernel_out = kernel_out.reshape(-1, 1)

            A_out = np.concatenate(
                [rho_out_extended(g) @ kernel_out for g in grid], axis=1
            ).T
        else:
            A_out = A
            _missing_input_irreps = []
            rho_out_extended = self.rho_out

        eps = 1e-8
        Ainv = (
            np.linalg.inv(A_out.T @ A_out + eps * np.eye(rho_out_extended.size))
            @ A_out.T
        )

        if out_irreps is not None:
            Ainv = Ainv[: self.rho_out.size, :]

        self.register_buffer("A", torch.tensor(A, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "Ainv", torch.tensor(Ainv, dtype=torch.get_default_dtype())
        )

    def update_identity_val(self, f: torch.Tensor, total_max: torch.Tensor):
        r"""Updates the deviation of the identity from the maximum likelihood
        (useful for shift/alignment error).

        Args:
            f (torch.Tensor): The sampled likelihood
            total_max (torch.tensor): the maximum likelihood"""
        f = f - total_max
        self.identity_val = (f)[:, :, self.identiy_ind]

    def softmax(self, logits: torch.Tensor, logits_max: torch.Tensor) -> torch.Tensor:
        r"""Applies softmax non-linearity on the logits to compute the true likelihood.
        Also saves the z-term for future KL-divergence computation.

        Args:
            logits (torch.Tensor): The logits obtained from the parameterised Fourier coefficients
            x_max (torch.Tensor):

        Returns
            f (torch.Tensor): Normalised likelihood distribution
        """
        logits_exp = torch.exp(logits)
        logits_exp_sum = torch.mean(logits_exp)
        f = logits_exp / logits_exp_sum
        self.z = torch.log(logits_exp_sum) + logits_max
        return f

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Applies the pointwise activation function on the input fields

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map after the non-linearities have been applied

        """

        assert input.type == self.in_type

        shape = input.shape
        x_hat = input.tensor.view(
            shape[0], len(self.in_type), self.rho.size, *shape[2:]
        )

        logits = torch.einsum("bcf...,gf->bcg...", x_hat, self.A)
        logits_max = torch.max(logits)
        x_logits = logits - torch.max(logits_max)

        y = self.softmax(x_logits, logits_max)

        total_logits_max = logits_max
        if hasattr(self, "A_identity"):
            logits_id = torch.einsum("bcf...,gf->bcg...", x_hat, self.A_identity)
            total_logits_max = max(logits_max, torch.max(logits_id))

        self.update_identity_val(logits, total_logits_max)

        y_hat = torch.einsum("bcg...,fg->bcf...", y, self.Ainv)

        y_hat = y_hat.reshape(shape[0], self.out_type.size, *shape[2:])

        return GeometricTensor(y_hat, self.out_type, input.coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(
        self, atol: float = 1e-5, rtol: float = 2e-2, assert_raise: bool = True
    ) -> List[Tuple[Any, float]]:
        c = self.in_type.size
        B = 128
        x = torch.randn(B, c, *[3] * self.space.dimensionality)

        # since we mostly use non-linearities like relu or elu, we make sure the average value of the features is
        # positive, such that, when we test inputs with only frequency 0 (or only low frequencies), the output is not
        # zero everywhere
        x = x.view(
            B, len(self.in_type), self.rho.size, *[3] * self.space.dimensionality
        )
        p = 0
        for irr in self.rho.irreps:
            irr = self.space.irrep(*irr)
            if irr.is_trivial():
                x[:, :, p] = x[:, :, p].abs()
            p += irr.size

        x = x.view(B, self.in_type.size, *[3] * self.space.dimensionality)

        errors = []

        # for el in self.space.testing_elements:
        for _ in range(100):
            el = self.space.fibergroup.sample()

            x1 = GeometricTensor(x.clone(), self.in_type)
            x2 = GeometricTensor(x.clone(), self.in_type).transform_fibers(el)

            out1 = self(x1).transform_fibers(el)
            out2 = self(x2)

            out1 = (
                out1.tensor.view(
                    B, len(self.out_type), self.rho_out.size, *out1.shape[2:]
                )
                .detach()
                .numpy()
            )
            out2 = (
                out2.tensor.view(
                    B, len(self.out_type), self.rho_out.size, *out2.shape[2:]
                )
                .detach()
                .numpy()
            )

            errs = np.linalg.norm(out1 - out2, axis=2).reshape(-1)
            errs[errs < atol] = 0.0
            norm = np.sqrt(
                np.linalg.norm(out1, axis=2).reshape(-1)
                * np.linalg.norm(out2, axis=2).reshape(-1)
            )

            relerr = errs / norm

            # print(el, errs.max(), errs.mean(), relerr.max(), relerr.min())

            if assert_raise:
                assert (
                    relerr.mean() + relerr.std() < rtol
                ), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {}, std ={}'.format(
                    el, relerr.max(), relerr.mean(), relerr.std()
                )

            # errors.append((el, errs.mean()))
            errors.append(relerr)

        # return errors
        return np.concatenate(errors).reshape(-1)
