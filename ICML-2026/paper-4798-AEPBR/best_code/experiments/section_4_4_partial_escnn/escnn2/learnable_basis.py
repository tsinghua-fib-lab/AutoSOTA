import numpy as np

from escnn.kernels.steerable_basis import IrrepBasis
from escnn.kernels.steerable_filters_basis import SteerableFiltersBasis
from escnn.group import *

import torch
from typing import Union, Tuple, Dict, Iterable, List
from collections import defaultdict
from itertools import chain


class PrelimLearnableWignerEckartBasis(IrrepBasis):
    def __init__(
        self,
        basis: SteerableFiltersBasis,
        in_irrep: Union[str, IrreducibleRepresentation, int],
        out_irrep: Union[str, IrreducibleRepresentation, int],
        layer_id: int,
    ):
        """Prelminary approach to learning the degree of equivariance. Allows for the
        learning of non-zero mapping between non-matchin basis and CG-decomposition irreps.


        Args:
            basis (SteerableFiltersBasis): a `G`-steerable basis for scalar functions over the base space
            in_repr (IrreducibleRepresentation): the input irrep
            out_repr (IrreducibleRepresentation): the output irrep
            layer_id (int, optional, default=None): Identifier of the layer ID. Layers with the same ID share the degree of
                                                    equivariance."""

        self.layer_id = layer_id
        group = basis.group
        in_irrep = group.irrep(*group.get_irrep_id(in_irrep))
        out_irrep = group.irrep(*group.get_irrep_id(out_irrep))

        assert in_irrep.group == group
        assert out_irrep.group == group
        assert in_irrep.group == out_irrep.group
        self.dim = 1
        self.m = in_irrep.id
        self.n = out_irrep.id
        _js = group._tensor_product_irreps(self.m, self.n)
        _js = basis.js

        _js = [(j, jJl) for j, jJl in _js if basis.multiplicity(j) > 0]
        dim = 0
        self._dim_harmonics = {}
        self._jJl = {}
        for j, jJl in _js:
            self._dim_harmonics[j] = (
                basis.multiplicity(j)
                * jJl
                * group.irrep(*j).sum_of_squares_constituents
            )
            self._jJl[j] = jJl
            dim += self._dim_harmonics[j]

        super(PrelimLearnableWignerEckartBasis, self).__init__(
            basis, in_irrep, out_irrep, dim, harmonics=[_j for _j, _ in _js]
        )
        # Identity mapping if irreps are equal, otherwise *near* zero mapping (required for gradients)
        if in_irrep != out_irrep:
            sampling = torch.ones(out_irrep.size, in_irrep.size) * 1e-20
        else:
            sampling = torch.eye(out_irrep.size, in_irrep.size)
        self.sampling = torch.nn.Parameter(sampling, requires_grad=True)

    def sample(self, points: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            s1, s2 = self.io_pair.shape
            out = torch.zeros(1, 1, s1, s2)
        out += self.sampling.view(1, 1, s1, s2)
        return out

    def sample_harmonics(
        self, points: Dict[Tuple, torch.Tensor], out: Dict[Tuple, torch.Tensor] = None
    ) -> Dict[Tuple, torch.Tensor]:
        for j in self.js:
            if j in out:
                assert out[j].shape == (
                    points[j].shape[0],
                    self.dim_harmonic(j),
                    self.shape[0],
                    self.shape[1],
                )
        for j in points:
            out[j] += self.sampling.reshape(
                1, 1, self.sampling.shape[0], self.sampling.shape[1]
            )
        return out

    _cached_instances = {}

    def __hash__(self):
        return (
            hash(self.basis)
            + hash(self.in_irrep)
            + hash(self.out_irrep)
            + hash(tuple(self.js))
            + hash(self.layer_id)
        )

    def __eq__(self, other):
        if not isinstance(other, PrelimLearnableWignerEckartBasis):
            return False
        elif (
            self.basis != other.basis
            or self.in_irrep != other.in_irrep
            or self.out_irrep != other.out_irrep
            or self.layer_id != other.layer_id
        ):
            # TODO check isomorphism too!
            return False
        elif len(self.js) != len(other.js):
            return False
        else:
            # for b, (j, i) in enumerate(zip(self.js, other.js)):
            #     if j != i or not torch.allclose(self.coeff(b), other.coeff(b)):
            #         return False
            return True

    def __getitem__(self, idx):
        assert 0 <= idx < self.dim

        i = idx
        for j in self.js:
            dim = self.dim_harmonic(j)
            if i < dim:
                break
            else:
                i -= dim
        return self.attrs_j(j, i)

    def attrs_j(self, j: Tuple, idx) -> Dict:
        assert 0 <= idx < self.dim_harmonic(j)

        full_idx = self._start_index[j] + idx

        dim = self.basis.multiplicity(j)

        attr = {"irrep:" + k: v for k, v in self.group.irrep(*j).attributes.items()}

        i = idx % dim
        attr_i = self.basis.steerable_attrs_j(j, i)

        attr.update(**attr_i)
        attr["idx"] = full_idx
        attr["j"] = j
        attr["i"] = i
        attr["s"] = (idx // dim) % self._jJl[j]
        attr["k"] = idx // (dim * self._jJl[j])

        return attr

    def dim_harmonic(self, j: Tuple) -> int:
        if j in self._dim_harmonics:
            return self._dim_harmonics[j]
        else:
            return 0

    def attrs_j_iter(self, j: Tuple) -> Iterable:
        if self.dim_harmonic(j) == 0:
            return

        idx = self._start_index[j]

        j_attr = {"irrep:" + k: v for k, v in self.group.irrep(*j).attributes.items()}
        steerable_basis_j_attr = list(self.basis.steerable_attrs_j_iter(j))

        for k in range(self.group.irrep(*j).sum_of_squares_constituents):
            for s in range(self._jJl[j]):
                for i, attr_i in enumerate(steerable_basis_j_attr):
                    attr = j_attr.copy()
                    attr.update(**attr_i)
                    attr["idx"] = idx
                    attr["j"] = j
                    attr["i"] = i
                    attr["s"] = s
                    attr["k"] = k
                    idx += 1

                    yield attr

    @classmethod
    def _generator(
        cls,
        basis: SteerableFiltersBasis,
        psi_in: Union[IrreducibleRepresentation, Tuple],
        psi_out: Union[IrreducibleRepresentation, Tuple],
        **kwargs,
    ) -> "IrrepBasis":
        assert len(kwargs) == 1
        layer_id = kwargs["layer_id"]

        psi_in = basis.group.irrep(*basis.group.get_irrep_id(psi_in))
        psi_out = basis.group.irrep(*basis.group.get_irrep_id(psi_out))

        key = (basis, psi_in.id, psi_out.id, layer_id)
        if key not in cls._cached_instances:
            cls._cached_instances[key] = PrelimLearnableWignerEckartBasis(
                basis, in_irrep=psi_in, out_irrep=psi_out, layer_id=layer_id
            )
        return cls._cached_instances[key]
