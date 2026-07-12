from escnn.kernels import SteerableKernelBasis, EmptyBasisException
from escnn.nn.modules.basismanager import BasisManager
from escnn.nn.modules.basismanager.basisexpansion_singleblock import normalize_basis

from typing import Callable, Dict, Iterable, Union, Tuple

import torch
import numpy as np

__all__ = ["SingleBlockBasisSampler", "block_basissampler"]


class SingleBlockBasisSampler(torch.nn.Module, BasisManager):
    def __init__(self, basis: SteerableKernelBasis, mask: np.ndarray = None):
        r"""

        Basis expansion method for a single contiguous block, i.e. for kernels whose input type and output type contain
        only fields of one type.

        Args:
            basis (SteerableKernelBasis): analytical basis to sample
            mask (np.ndarray, optional): binary mask to select only a subset of the basis elements.
                                         By default (``None``), all elements are kept.

        """

        super(SingleBlockBasisSampler, self).__init__()

        self.basis = basis

        if mask is None:
            mask = np.ones(len(basis), dtype=bool)

        assert mask.shape == (len(basis),) and mask.dtype == bool

        if not mask.any():
            raise EmptyBasisException

        self._mask = mask
        self.basis = basis

        # we need to know the real output size of the basis elements (i.e. without the change of basis and the padding)
        # to perform the normalization
        sizes = []
        for attr in basis:
            sizes.append(attr["shape"][0])

        sizes = torch.tensor(sizes, dtype=torch.float32).reshape(1, 1, 1, -1)

        # to normalize the basis
        self.register_buffer("sizes", sizes)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        assert len(points.shape) == 2
        basis = self.basis.sample(points)

        norms = getattr(self, "norms", None)
        # basis has shape (p, k, o, i)
        # permute to (p, o, i, k)
        basis = basis.permute(1, 2, 3, 0)
        mask = self._mask
        # Multiply with inverse norms if normalize() has been called.
        if norms is not None:
            basis = basis[mask] * norms[mask]
        else:
            basis = basis[mask] * self.sizes.permute(3, 0, 1, 2)[mask]

        basis = basis.permute((3, 1, 2, 0))

        return basis

    def get_element_info(self, id: int) -> Dict:
        idx = 0
        for i, attr in enumerate(self.basis):
            if self._mask[i]:
                if idx == id:
                    attr["id"] = idx
                    return attr
                else:
                    idx += 1

    def get_basis_info(self) -> Iterable[Dict]:
        idx = 0
        for i, attr in enumerate(self.basis):
            if self._mask[i]:
                attr["id"] = idx
                idx += 1
                yield attr

    def dimension(self) -> int:
        return self._mask.astype(int).sum()

    def normalize(self, points: torch.Tensor):
        """
        Compute the norms of the basis sampled with points and store as buffer.
        """
        with torch.no_grad():
            sampled_basis = self.basis.sample(points).permute(1, 2, 3, 0)

        b = sampled_basis.shape[0]

        norms = torch.einsum(
            "bop...,bpq...->boq...", (sampled_basis, sampled_basis.transpose(1, 2))
        ).detach()

        norms = torch.einsum("bii...->b", norms)
        norms /= self.sizes.flatten()
        norms[norms < 1e-15] = 0
        norms = torch.sqrt(norms)

        norms[norms < 1e-6] = 1
        norms[norms != norms] = 1

        mask = norms**2 < 1e-2

        norms = 1 / norms

        norms[mask] = 0

        norms = norms.view(b, *([1] * (len(sampled_basis.shape) - 1)))

        self.register_buffer("norms", norms)

    def __eq__(self, other):
        if isinstance(other, SingleBlockBasisSampler):
            return self.basis == other.basis and np.all(self._mask == other._mask)
        else:
            return False

    def __hash__(self):
        return 10000 * hash(self.basis) + hash(self._mask.tobytes())


# dictionary storing references to already built basis samplers
# when a new filter tensor is built, it is also stored here
# when the same basis is built again (eg. in another layer), the already existing filter tensor is retrieved
_stored_filters = {}


def block_basissampler(
    basis: SteerableKernelBasis,
    basis_filter: Callable[[dict], bool] = None,
    recompute: bool = False,
) -> SingleBlockBasisSampler:
    r"""


    Args:
        basis (SteerableKernelBasis): basis defining the space of kernels
        basis_filter (callable, optional): filter for the basis elements. Should take a dictionary containing an
                                           element's attributes and return whether to keep it or not.
        recompute (bool, optional): whether to recompute new bases or reuse, if possible, already built tensors.

    """

    if basis_filter is not None:
        mask = np.zeros(len(basis), dtype=bool)
        for b, attr in enumerate(basis):
            mask[b] = basis_filter(attr)
    else:
        mask = np.ones(len(basis), dtype=bool)

    if not recompute:
        # compute the mask of the sampled basis containing only the elements allowed by the filter
        key = (basis, mask.tobytes())
        if key not in _stored_filters:
            _stored_filters[key] = SingleBlockBasisSampler(basis, mask)

        return _stored_filters[key]

    else:
        return SingleBlockBasisSampler(basis, mask)
