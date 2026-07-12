import numpy as np

from escnn.kernels.steerable_basis import IrrepBasis
from escnn.kernels.steerable_filters_basis import SteerableFiltersBasis

from .irreps_mapping import (
    IrrepsMap,
    IrrepsMapFourier,
    IrrepsMapFourierBL,
    IrrepsMapFourierBLact,
)

from escnn.group import *

import torch

from typing import Union, Tuple, Dict, Iterable, List
from collections import defaultdict
from itertools import chain

import time

__all__ = [
    "WignerEckartBasis",
    "RestrictedWignerEckartBasis",
    "LearnableWignerEckartBasis",
    "LearnableRestictedWignerEckartBasis",
]


class LearnableWignerEckartBasis(IrrepBasis):
    def __init__(
        self,
        basis: SteerableFiltersBasis,
        in_irrep: Union[str, IrreducibleRepresentation, int],
        out_irrep: Union[str, IrreducibleRepresentation, int],
        layer_id: int = None,
        **mapping_kwargs,
    ):
        r"""
        Learnable Wigner Eckart-based basis.
        Initially, essentially solves the kernel constraint for a pair of input and output :math:`G`-irreps by using the Wigner-Eckart theorem
        described in Theorem 2.1 of
        `A Program to Build E(N)-Equivariant Steerable CNNs <https://openreview.net/forum?id=WE4qe9xlnQw>`_
        (see also
        `A Wigner-Eckart Theorem for Group Equivariant Convolution Kernels <https://arxiv.org/abs/2010.10952>`_
        ).

        However, implements it as described in <https://arxiv.org/abs/2406.03946>, parameterising the equivariance as a
        learnable likelihood distribution, allowing breaking of equivariance.

        The method relies on a :math:`G`-Steerable basis of scalar functions over the base space.

        Args:
            basis (SteerableFiltersBasis): a `G`-steerable basis for scalar functions over the base space
            in_repr (IrreducibleRepresentation): the input irrep
            out_repr (IrreducibleRepresentation): the output irrep
            layer_id (int, optional, default=None): Identifier of the layer ID. Layers with the same ID share the degree of
                                                    equivariance.
            **mapping_kwargs: Additional kwargs for the irrep mapping.

        """

        self.layer_id = layer_id
        group = basis.group

        in_irrep = group.irrep(*group.get_irrep_id(in_irrep))
        out_irrep = group.irrep(*group.get_irrep_id(out_irrep))

        assert in_irrep.group == group
        assert out_irrep.group == group
        assert in_irrep.group == out_irrep.group

        # print(in_irrep, out_irrep)

        self.m = in_irrep.id
        self.n = out_irrep.id

        self._js_cg = group._tensor_product_irreps(self.m, self.n)

        _js = basis.js

        _js = [j for j, _ in _js if basis.multiplicity(j) > 0]

        self._basis_to_cg = defaultdict(list)

        irrepmap = IrrepsMapFourierBLact._generator(
            self.layer_id, group, **mapping_kwargs
        )

        dim = 0
        self._dim_harmonics = {}
        self._jJl = {}
        with torch.no_grad():
            for j in _js:
                jJl = len(self._js_cg)
                # jJl = len(self._basis_to_cg[j])
                self._dim_harmonics[j] = (
                    basis.multiplicity(j)
                    # * jJl
                    # * 1  # group.irrep(*j).sum_of_squares_constituents
                    * sum(irrepmap.dim(j, _j) * _jJl for _j, _jJl in self._js_cg)
                )

                dim += self._dim_harmonics[j]

        for j, jJl in self._js_cg:
            self._jJl[j] = jJl

        super(LearnableWignerEckartBasis, self).__init__(
            basis, in_irrep, out_irrep, dim, harmonics=_js
        )
        # SteerableFiltersBasis: a `G`-steerable basis for scalar functions over the base space
        self.basis = basis
        self._irrepmap = irrepmap

        _cg = [
            torch.tensor(
                group._clebsh_gordan_coeff(self.n, self.m, j), dtype=torch.float32
            )
            for j, _ in self._js_cg
        ]

        self._irrepmap = irrepmap
        self._empty_c = set()
        with torch.no_grad():
            for j_basis in self.js:
                for j_cg, _ in self._js_cg:
                    c = self._irrepmap(j_basis, j_cg)
                    if not all(c.shape):
                        self._empty_c.add((j_basis, j_cg))

        for b, j in enumerate(self._js_cg):
            self.register_buffer(f"clebsch_gordan_{b}", _cg[b])

    def coeff(self, idx_basis: int, idx_cg_j: int) -> torch.Tensor:
        r"""Uses the irrepmapping to compute the coefficients for the basis using basis irrep and irrep from
         clebsch-gordan decomposition.
        Args:
            idx_baxis (int): index of the basis irrep.
            idx_cg_j (int): index of clebsch gordan irrep
        Returns:
            coeff (torch.Tensor): basis coefficients
        """
        clebsch = getattr(self, f"clebsch_gordan_{idx_cg_j}")
        j_basis, (j_cg, _) = self.js[idx_basis], self._js_cg[idx_cg_j]
        c = self._irrepmap(j_basis, j_cg)
        return torch.einsum("nmsi,koi->ksnmo", clebsch, c)

    def sample_harmonics(
        self, points: Dict[Tuple, torch.Tensor], out: Dict[Tuple, torch.Tensor] = None
    ) -> Dict[Tuple, torch.Tensor]:
        if out is None:
            out = {
                j: torch.zeros(
                    (
                        points[j].shape[0],
                        self.dim_harmonic(j),
                        self.shape[0],
                        self.shape[1],
                    ),
                    device=points[j].device,
                    dtype=points[j].dtype,
                )
                for j in self.js
            }

        for j in self.js:
            if j in out:
                assert out[j].shape == (
                    points[j].shape[0],
                    self.dim_harmonic(j),
                    self.shape[0],
                    self.shape[1],
                ), (
                    out[j].shape,
                    points[j].shape[0],
                    self.dim_harmonic(j),
                    self.shape[0],
                    self.shape[1],
                )
        for b_basis, j_basis in enumerate(self.js):
            if j_basis not in out:
                continue

            Ys = points[j_basis]
            out_j_basis = out[j_basis].view(
                (
                    Ys.shape[0],
                    self.dim_harmonic(j_basis) // Ys.shape[1],
                    Ys.shape[1],
                    self.out_irrep.size,
                    self.in_irrep.size,
                )
            )
            blocks = []

            p = 0
            for b_cg, (j_cg, _) in enumerate(self._js_cg):
                if (j_basis, j_cg) in self._empty_c:
                    continue
                coeff = self.coeff(b_basis, b_cg)
                k, s, m, n, o = coeff.shape

                Ys = points[j_basis]

                d = np.prod(coeff.shape[:2])
                update = torch.einsum(
                    "lmno,qio->qlimn",
                    coeff.view(k * s, m, n, o),
                    Ys,
                )
                blocks.append(update)
                p += d

            if blocks:
                out[j_basis] = (out_j_basis + torch.cat(blocks, dim=1)).reshape_as(
                    out[j_basis]
                )
            else:
                out[j_basis] = out_j_basis.reshape_as(out[j_basis])
        return out

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
        # nr_c_mats = 1  # self.group.irrep(*j).sum_of_squares_constituents
        # for k in range(nr_c_mats):
        for _j, _ in self._js_cg:
            for k in range(self._irrepmap.dim(j, _j)):
                for s in range(self._jJl[_j]):
                    for i, attr_i in enumerate(steerable_basis_j_attr):
                        attr = j_attr.copy()
                        attr.update(**attr_i)
                        attr["idx"] = idx
                        attr["j"] = j
                        attr["i"] = i
                        attr["s"] = s
                        attr["k"] = k
                        attr["_j"] = _j
                        idx += 1

                        yield attr

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
        _k = idx // (dim * self._jJl[j])

        p = 0
        for _j, _ in self._js_cg:
            if _k < p + self._irrepmap.dim(j, _j):
                _k = _k - p
                break
            p += self._irrepmap.dim(j, _j)

        attr["k"] = _k
        attr["_j"] = _j

        return attr

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

    def __iter__(self):
        for j in self.js:
            for attr in self.attrs_j_iter(j):
                yield attr
        # return chain(self.attrs_j_iter(j) for j in self.js)

    def __eq__(self, other):
        if not isinstance(other, LearnableWignerEckartBasis):
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
            for b_basis, (j, i) in enumerate(zip(self.js, other.js)):
                for b_cg, (j_cg, i_cg) in enumerate(zip(self._js_cg, other._js_cg)):
                    if (
                        j != i
                        or j_cg != i_cg
                        or not torch.allclose(
                            self.coeff(b_basis, b_cg), other.coeff(b_basis, b_cg)
                        )
                    ):
                        return False
            return True

    def __hash__(self):
        return (
            hash(self.basis)
            + hash(self.in_irrep)
            + hash(self.out_irrep)
            + hash(tuple(self.js))
            + hash(self.layer_id)
        )

    _cached_instances = {}

    @classmethod
    def _generator(
        cls,
        basis: SteerableFiltersBasis,
        psi_in: Union[IrreducibleRepresentation, Tuple],
        psi_out: Union[IrreducibleRepresentation, Tuple],
        **kwargs,
    ) -> "IrrepBasis":
        assert len(kwargs) == 2
        assert "layer_id" in kwargs
        assert "mapping_kwargs" in kwargs
        layer_id = kwargs["layer_id"]
        mapping_kwargs = kwargs["mapping_kwargs"]

        psi_in = basis.group.irrep(*basis.group.get_irrep_id(psi_in))
        psi_out = basis.group.irrep(*basis.group.get_irrep_id(psi_out))

        key = (basis, psi_in.id, psi_out.id, layer_id)

        if key not in cls._cached_instances:
            cls._cached_instances[key] = LearnableWignerEckartBasis(
                basis,
                in_irrep=psi_in,
                out_irrep=psi_out,
                layer_id=layer_id,
                **mapping_kwargs,
            )
        return cls._cached_instances[key]


class LearnableRestrictedWignerEckartBasis(IrrepBasis):
    def __init__(
        self,
        basis: SteerableFiltersBasis,
        sg_id: Tuple,
        in_irrep: Union[str, IrreducibleRepresentation, int],
        out_irrep: Union[str, IrreducibleRepresentation, int],
        layer_id: int = None,
        **mapping_kwargs,
    ):
        r"""

        Learnable Wigner Eckart-based basis (learnable degree of equivariance).
        Initially, essentially solves the kernel constraint for a pair of input and output :math:`G`-irreps by using the Wigner-Eckart theorem
        described in Theorem 2.1 of
        `A Program to Build E(N)-Equivariant Steerable CNNs <https://openreview.net/forum?id=WE4qe9xlnQw>`_
        (see also
        `A Wigner-Eckart Theorem for Group Equivariant Convolution Kernels <https://arxiv.org/abs/2010.10952>`_
        ).

        However, implements it as described in <https://arxiv.org/abs/2406.03946>, parameterising the equivariance as a
        learnable likelihood distribution, allowing breaking of equivariance.

        This method implicitly constructs the required :math:`G`-steerable basis for scalar functions on the base space
        from a :math:`G'`-steerable basis, with :math:`G' > G` a larger group, according to Equation 5 from the same
        paper.


        The equivariance group :math:`G < G'` is identified by the input id ``sg_id``.

        .. warning::
            Note that the group :math:`G'` associated with ``basis`` is generally not the same as the group :math:`G`
            associated with ``in_irrep`` and ``out_irrep`` and which the resulting kernel basis is equivariant to.

        Args:
            basis (SteerableFiltersBasis): :math:`G'`-steerable basis for scalar filters
            sg_id (tuple): id of :math:`G` as a subgroup of :math:`G'`.
            in_repr (IrreducibleRepresentation): the input `G`-irrep
            out_repr (IrreducibleRepresentation): the output `G`-irrep
            layer_id (int, optional, default=None): Identifier of the layer ID. Layers with the same ID share the degree of
                                                    equivariance.
            **mapping_kwargs: Additional kwargs for the irrep mapping.

        """
        self.layer_id = layer_id

        # the larger group G'
        _G = basis.group

        G = _G.subgroup(sg_id)[0]
        # Group: the smaller equivariance group G
        self.group = G
        self.sg_id = sg_id

        in_irrep = G.irrep(*G.get_irrep_id(in_irrep))
        out_irrep = G.irrep(*G.get_irrep_id(out_irrep))

        assert in_irrep.group == G
        assert out_irrep.group == G
        assert in_irrep.group == out_irrep.group

        self.m = in_irrep.id
        self.n = out_irrep.id

        # irreps of G in the decomposition of the tensor product of in_irrep and out_irrep
        self._js_G = G._tensor_product_irreps(self.m, self.n)

        _js = set()
        _js_restriction = defaultdict(list)

        # for each harmonic j' to consider
        for _j in set(_j for _j, _ in basis.js):
            if basis.multiplicity(_j) == 0:
                continue

            # restrict the corresponding G' irrep j' to G
            _j_G = _G.irrep(*_j).restrict(sg_id)

            # for each G-irrep j in the tensor product decomposition of in_irrep and out_irrep
            for j, _ in self._js_G:
                # irrep-decomposition coefficients of j in j'
                id_coeff_l = []
                p = 0
                # for each G-irrep i in the restriction of j' to G
                for i in _j_G.irreps:
                    size = G.irrep(*i).size
                    # if the restricted irrep contains one of the irreps in the tensor product
                    # if i == j:
                    id_coeff_l.append(_j_G.change_of_basis_inv[p : p + size, :])
                    id_coeff = _j_G.change_of_basis_inv[p : p + size, :][None, :]

                    p += size

                    # if the G irrep j appears in the restriction of the G'-irrep j',
                    # store its irrep-decomposition coefficients
                    # if len(id_coeff) > 0:
                    # id_coeff = np.stack(id_coeff, axis=-1)
                    _js.add(_j)
                    _js_restriction[_j].append((i, id_coeff))

        _js = sorted(list(_js))

        irrepmap = IrrepsMapFourierBLact._generator(self.layer_id, G, **mapping_kwargs)

        _cg = []
        _id_coeffs = []
        self._dim_harmonics = {}
        self._js_restriction = {}
        dim = 0

        for j_cg, _ in self._js_G:
            _cg.append(
                torch.tensor(
                    G._clebsh_gordan_coeff(self.n, self.m, j_cg),
                    dtype=torch.float32,
                )
            )

        for _j in _js:
            self._js_restriction[_j] = [
                (j, id_coeff.shape[0]) for j, id_coeff in _js_restriction[_j]
            ]
            with torch.no_grad():
                self._dim_harmonics[_j] = sum(
                    irrepmap.dim(j_tilde, j) * _jJl * t
                    for j, _jJl in self._js_G
                    for j_tilde, t in self._js_restriction[_j]
                )
                dim += self._dim_harmonics[_j] * basis.multiplicity(_j)

        self._jJl = {}
        for j, jJl in self._js_G:
            self._jJl[j] = jJl

        super(LearnableRestrictedWignerEckartBasis, self).__init__(
            basis, in_irrep, out_irrep, dim, harmonics=_js
        )

        for _j in self.js:
            _id_coeffs.append(
                [
                    torch.tensor(id_coeff, dtype=torch.float32)
                    for _, id_coeff in _js_restriction[_j]
                ]
            )

        self._irrepmap = irrepmap
        self._empty_c = set()
        with torch.no_grad():
            for j_basis in self.js:
                for j_cg, _ in self._js_G:
                    for j_restrict, _ in self._js_restriction[j_basis]:
                        with torch.no_grad():
                            c = self._irrepmap(j_restrict, j_cg)
                            if not all(c.shape):
                                self._empty_c.add((j_restrict, j_cg))

        # SteerableFiltersBasis: a `G'`-steerable basis for scalar functions over the base space, for the larger
        # group `G' > G`
        self.basis = basis

        for b, _j in enumerate(self.js):
            # setattr(self, f"nr_restrictions_{b}", len(_clebsch_id_coeff[b]))
            setattr(self, f"nr_restrictions_{b}", len(_js_restriction[_j]))
            # self.register_buffer(f"coeff_{b}", _coeffs[_j])
            for i in range(len(_js_restriction[_j])):

                self.register_buffer(f"id_coeffs_{b}_{i}", _id_coeffs[b][i])

        for b, (j, _) in enumerate(self._js_G):
            self.register_buffer(f"clebsch_gordan_{b}", _cg[b])

    def coeff(self, idx: int, idx_cg_j: int) -> torch.Tensor:
        _j = self.js[idx]
        _j_cg, _ = self._js_G[idx_cg_j]
        Y_size = self.basis.group.irrep(*_j).size
        # coeff_OLD = [
        #     torch.einsum(
        #         # 'nmsi,kji,jyt->nmksty',
        #         "nmsi,kji,jyt->kstnmy",
        #         getattr(self, f"clebsch_gordan_{idx}_{i}"),
        #         self._c[f"equivariance layer {self.layer_id} restrict {(_j, i)}"],
        #         # self._irrepmap(self._js_restriction[_j][i][0], _j_cg),
        #         getattr(self, f"id_coeffs_{idx}_{i}"),
        #     ).reshape((-1, self.out_irrep.size, self.in_irrep.size, Y_size))
        #     for i in range(getattr(self, f"nr_restrictions_{idx}"))
        # ]
        coeff = [
            torch.einsum(
                # 'nmsi,kji,jyt->nmksty',
                "nmsi,kji,tjy->kstnmy",
                getattr(self, f"clebsch_gordan_{idx_cg_j}"),
                # self._c[f"equivariance layer {self.layer_id} restrict {(_j, i)}"],
                self._irrepmap(self._js_restriction[_j][i][0], _j_cg),
                getattr(self, f"id_coeffs_{idx}_{i}"),
            ).reshape((-1, self.out_irrep.size, self.in_irrep.size, Y_size))
            for i in range(getattr(self, f"nr_restrictions_{idx}"))
        ]
        # coeff_OLD = torch.cat(coeff_OLD, dim=0)
        coeff = torch.cat(coeff, dim=0)
        return coeff.view(-1, *coeff.shape[-3:])

    def sample_harmonics(
        self, points: Dict[Tuple, torch.Tensor], out: Dict[Tuple, torch.Tensor] = None
    ) -> Dict[Tuple, torch.Tensor]:
        if out is None:
            out = {
                j: torch.zeros(
                    (
                        points[j].shape[0],
                        self.dim_harmonic(j),
                        self.shape[0],
                        self.shape[1],
                    ),
                    device=points[j].device,
                    dtype=points[j].dtype,
                )
                for j in self.js
            }

        for j in self.js:
            if j in out:
                assert out[j].shape == (
                    points[j].shape[0],
                    self.dim_harmonic(j),
                    self.shape[0],
                    self.shape[1],
                )
        for b_basis, j_basis in enumerate(self.js):
            if j_basis not in out:
                continue
            Ys = points[j_basis]
            out_j_basis = out[j_basis].view(
                (
                    Ys.shape[0],
                    self.dim_harmonic(j_basis) // Ys.shape[1],
                    Ys.shape[1],
                    self.out_irrep.size,
                    self.in_irrep.size,
                )
            )
            blocks = []
            for b_cg, (j_cg, _) in enumerate(self._js_G):
                coeff = self.coeff(b_basis, b_cg)

                blocks.append(torch.einsum("lmno,qio->qlimn", coeff, Ys))
            if blocks:
                out[j_basis] = (out_j_basis + torch.cat(blocks, dim=1)).reshape_as(
                    out[j_basis]
                )
            else:
                out[j_basis] = out_j_basis.reshape_as(out[j_basis])
        return out

    def dim_harmonic(self, j: Tuple) -> int:
        if j in self._dim_harmonics:
            return self.basis.multiplicity(j) * self._dim_harmonics[j]
        else:
            return 0

    def attrs_j_iter(self, _j: Tuple) -> Iterable:
        if self.dim_harmonic(_j) == 0:
            return

        idx = self._start_index[_j]

        steerable_basis_j_attr = list(self.basis.steerable_attrs_j_iter(_j))

        j_attr = {
            "irrep:" + k: v for k, v in self.basis.group.irrep(*_j).attributes.items()
        }

        count = 0
        for j_tilde, _jj in self._js_restriction[_j]:
            for _j_cg, _ in self._js_G:
                for k in range(self._irrepmap.dim(j_tilde, _j_cg)):
                    for s in range(self._jJl[_j_cg]):
                        for t in range(_jj):
                            for i, attr_i in enumerate(steerable_basis_j_attr):
                                attr = j_attr.copy()
                                attr.update(**attr_i)
                                attr["idx"] = idx
                                attr["j"] = _j
                                attr["_j"] = _jj
                                attr["i"] = i
                                attr["t"] = t
                                attr["s"] = s
                                attr["k"] = k

                                assert idx < self.dim
                                assert count < self.dim_harmonic(_j), (
                                    count,
                                    self.dim_harmonic(_j),
                                )

                                idx += 1
                                count += 1

                                yield attr

        assert count == self.dim_harmonic(_j), (count, self.dim_harmonic(_j))

    def attrs_j(self, j: Tuple, idx) -> Dict:
        assert 0 <= idx < self.dim_harmonic(j)

        full_idx = self._start_index[j] + idx

        dim = self.basis.multiplicity(j)

        for _j, _jj in self._js_restriction[j]:
            _jJl = self.group._clebsh_gordan_coeff(self.n, self.m, _j).shape[2]
            K = 1  # self.group.irrep(*_j).sum_of_squares_constituents

            d = _jj * _jJl * K * dim

            if idx >= d:
                idx -= d
            else:
                break

        i = idx % dim
        attr_i = self.basis.steerable_attrs_j(j, i)

        attr = {
            "irrep:" + k: v for k, v in self.basis.group.irrep(*j).attributes.items()
        }
        attr.update(**attr_i)
        attr["idx"] = full_idx
        attr["j"] = j
        attr["_j"] = _j
        attr["i"] = i
        attr["t"] = (idx // dim) % _jj
        attr["s"] = (idx // (dim * _jj)) % _jJl
        attr["k"] = idx // (dim * _jj * _jJl)
        return attr

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

    def __iter__(self) -> Iterable:
        for j in self.js:
            for attr in self.attrs_j_iter(j):
                yield attr
        # return chain(self.attrs_j_iter(j) for j in self.js)

    def __eq__(self, other):
        if not isinstance(other, RestrictedWignerEckartBasis):
            return False
        elif (
            self.basis != other.basis
            or self.sg_id != other.sg_id
            or self.in_irrep != other.in_irrep
            or self.out_irrep != other.out_irrep
            or self.layer_id != other.layer_id
        ):
            # TODO check isomorphism too!
            return False
        elif len(self.js) != len(other.js):
            return False
        else:
            for b, (j, i) in enumerate(zip(self.js, other.js)):
                if j != i or not torch.allclose(self.coeff(b), other.coeff(b)):
                    return False
            return True

    def __hash__(self):
        return (
            hash(self.basis)
            + hash(self.sg_id)
            + hash(self.in_irrep)
            + hash(self.out_irrep)
            + hash(tuple(self.js))
            + hash(self.layer_id)
        )

    _cached_instances = {}

    @classmethod
    def _generator(
        cls,
        basis: SteerableFiltersBasis,
        psi_in: Union[IrreducibleRepresentation, Tuple],
        psi_out: Union[IrreducibleRepresentation, Tuple],
        **kwargs,
    ) -> "IrrepBasis":
        assert len(kwargs) == 3
        assert "sg_id" in kwargs
        assert "layer_id" in kwargs
        assert "mapping_kwargs" in kwargs
        sg_id = kwargs["sg_id"]
        layer_id = kwargs["layer_id"]
        mapping_kwargs = kwargs["mapping_kwargs"]

        G, _, _ = basis.group.subgroup(kwargs["sg_id"])
        psi_in = G.irrep(*G.get_irrep_id(psi_in))
        psi_out = G.irrep(*G.get_irrep_id(psi_out))

        key = (basis, psi_in.id, psi_out.id, kwargs["sg_id"], layer_id)

        if key not in cls._cached_instances:
            cls._cached_instances[key] = LearnableRestrictedWignerEckartBasis(
                basis,
                sg_id=sg_id,
                in_irrep=psi_in,
                out_irrep=psi_out,
                layer_id=layer_id,
                **mapping_kwargs,
            )

        return cls._cached_instances[key]


class LearnableRestrictedWignerEckartBasisOLD(IrrepBasis):
    def __init__(
        self,
        basis: SteerableFiltersBasis,
        sg_id: Tuple,
        in_irrep: Union[str, IrreducibleRepresentation, int],
        out_irrep: Union[str, IrreducibleRepresentation, int],
        layer_id: int = None,
    ):
        r"""

        Solves the kernel constraint for a pair of input and output :math:`G`-irreps by using the Wigner-Eckart theorem
        described in Theorem 2.1 of
        `A Program to Build E(N)-Equivariant Steerable CNNs  <https://openreview.net/forum?id=WE4qe9xlnQw>`_.
        This method implicitly constructs the required :math:`G`-steerable basis for scalar functions on the base space
        from a :math:`G'`-steerable basis, with :math:`G' > G` a larger group, according to Equation 5 from the same
        paper.

        The equivariance group :math:`G < G'` is identified by the input id ``sg_id``.

        .. warning::
            Note that the group :math:`G'` associated with ``basis`` is generally not the same as the group :math:`G`
            associated with ``in_irrep`` and ``out_irrep`` and which the resulting kernel basis is equivariant to.

        Args:
            basis (SteerableFiltersBasis): :math:`G'`-steerable basis for scalar filters
            sg_id (tuple): id of :math:`G` as a subgroup of :math:`G'`.
            in_repr (IrreducibleRepresentation): the input `G`-irrep
            out_repr (IrreducibleRepresentation): the output `G`-irrep

        """
        self.layer_id = layer_id

        # the larger group G'
        _G = basis.group

        G = _G.subgroup(sg_id)[0]
        # Group: the smaller equivariance group G
        self.group = G
        self.sg_id = sg_id

        in_irrep = G.irrep(*G.get_irrep_id(in_irrep))
        out_irrep = G.irrep(*G.get_irrep_id(out_irrep))

        assert in_irrep.group == G
        assert out_irrep.group == G
        assert in_irrep.group == out_irrep.group

        self.m = in_irrep.id
        self.n = out_irrep.id

        # irreps of G in the decomposition of the tensor product of in_irrep and out_irrep
        _js_G = [j for j, _ in G._tensor_product_irreps(self.m, self.n)]

        _js = set()
        _js_restriction = defaultdict(list)

        # for each harmonic j' to consider
        for _j in set(_j for _j, _ in basis.js):
            if basis.multiplicity(_j) == 0:
                continue

            # restrict the corresponding G' irrep j' to G
            _j_G = _G.irrep(*_j).restrict(sg_id)

            # for each G-irrep j in the tensor product decomposition of in_irrep and out_irrep
            for j in _js_G:
                # irrep-decomposition coefficients of j in j'
                id_coeff = []
                p = 0
                # for each G-irrep i in the restriction of j' to G
                for i in _j_G.irreps:
                    size = G.irrep(*i).size
                    # if the restricted irrep contains one of the irreps in the tensor product
                    if i == j:
                        id_coeff.append(_j_G.change_of_basis_inv[p : p + size, :])

                    p += size

                # if the G irrep j appears in the restriction of the G'-irrep j',
                # store its irrep-decomposition coefficients
                if len(id_coeff) > 0:
                    id_coeff = np.stack(id_coeff, axis=-1)
                    _js.add(_j)
                    _js_restriction[_j].append((j, id_coeff))

        _js = sorted(list(_js))

        # self._js_restriction = {}
        # self._dim_harmonics = {}
        # _coeffs = {}
        # dim = 0
        # for _j in _js:
        #     Y_size = _G.irrep(*_j).size
        #     coeff = [
        #         torch.einsum(
        #             # 'nmsi,kji,jyt->nmksty',
        #             "nmsi,kji,jyt->kstnmy",
        #             torch.tensor(
        #                 G._clebsh_gordan_coeff(self.n, self.m, j), dtype=torch.float32
        #             ),
        #             torch.tensor(G.irrep(*j).endomorphism_basis(), dtype=torch.float32),
        #             torch.tensor(id_coeff, dtype=torch.float32),
        #         ).reshape((-1, out_irrep.size, in_irrep.size, Y_size))
        #         for j, id_coeff in _js_restriction[_j]
        #     ]
        #     _coeffs[_j] = torch.cat(coeff, dim=0)
        # self._js_restriction[_j] = [
        #     (j, id_coeff.shape[2]) for j, id_coeff in _js_restriction[_j]
        # ]
        # self._dim_harmonics[_j] = _coeffs[_j].shape[0]
        # dim += self._dim_harmonics[_j] * basis.multiplicity(_j)

        # _cg = []
        _c = []
        # _id_coeffs = []
        _clebsch_id_coeff = []
        self._dim_harmonics = {}
        self._js_restriction = {}
        dim = 0
        for _j in _js:
            Y_size = _G.irrep(*_j).size
            _clebsch_id_coeff.append(
                [
                    torch.einsum(
                        "nmsi,jyt->snmityj",
                        torch.tensor(
                            G._clebsh_gordan_coeff(self.n, self.m, j),
                            dtype=torch.float32,
                        ),
                        torch.tensor(id_coeff, dtype=torch.float32),
                    )
                    for j, id_coeff in _js_restriction[_j]
                ]
            )
            # _cg.append(
            #     [
            #         torch.tensor(
            #             G._clebsh_gordan_coeff(self.n, self.m, j),
            #             dtype=torch.float32,
            #         )
            #         for j, _ in _js_restriction[_j]
            #     ]
            # )

            # _id_coeffs.append(
            #     [
            #         torch.tensor(id_coeff, dtype=torch.float32)
            #         for _, id_coeff in _js_restriction[_j]
            #     ]
            # )

            _c.append(
                [
                    torch.nn.Parameter(
                        torch.tensor(
                            G.irrep(*j).endomorphism_basis(), dtype=torch.float32
                        )
                    )
                    for j, _ in _js_restriction[_j]
                ]
            )

            self._js_restriction[_j] = [
                (j, id_coeff.shape[2]) for j, id_coeff in _js_restriction[_j]
            ]
            s, n, m, i, t, y, j = _clebsch_id_coeff[-1][-1].shape
            # n, m, s, i = _cg[-1][-1].shape
            # j, y, t = _id_coeffs[-1][-1].shape
            k, j, i = _c[-1][-1].shape
            out = int(k * s * t * n * m * y / (out_irrep.size * in_irrep.size * Y_size))
            self._dim_harmonics[_j] = len(_js_restriction[_j]) * out
            dim += self._dim_harmonics[_j] * basis.multiplicity(_j)

        super(LearnableRestrictedWignerEckartBasis, self).__init__(
            basis, in_irrep, out_irrep, dim, harmonics=_js
        )

        self._c = torch.nn.ParameterDict(
            {
                f"equivariance layer {self.layer_id} restrict {(_j, i)}": _c[k][i]
                for k, _j in enumerate(_js)
                for i in range(len(_js_restriction[_j]))
            }
        )

        # SteerableFiltersBasis: a `G'`-steerable basis for scalar functions over the base space, for the larger
        # group `G' > G`
        self.basis = basis

        for b, _j in enumerate(self.js):
            setattr(self, f"nr_restrictions_{b}", len(_clebsch_id_coeff[b]))
            # setattr(self, f"nr_restrictions_{b}", len(_id_coeffs[b]))
            # self.register_buffer(f"coeff_{b}", _coeffs[_j])
            for i in range(len(_js_restriction[_j])):
                self.register_buffer(
                    f"clebsch_id_coeff_{b}_{i}", _clebsch_id_coeff[b][i]
                )

                # self.register_buffer(f"id_coeffs_{b}_{i}", _id_coeffs[b][i])
                # self.register_buffer(f"clebsch_gordan_{b}_{i}", _cg[b][i])

    def coeff(self, idx: int) -> torch.Tensor:
        _j = self.js[idx]
        Y_size = self.basis.group.irrep(*_j).size

        coeff = [
            torch.einsum(
                "snmityj, kji -> kstnmy",
                getattr(self, f"clebsch_id_coeff_{idx}_{i}"),
                self._c[f"equivariance layer {self.layer_id} restrict {(_j, i)}"],
            ).reshape((-1, self.out_irrep.size, self.in_irrep.size, Y_size))
            for i in range(getattr(self, f"nr_restrictions_{idx}"))
        ]

        # coeff = [
        #     torch.einsum(
        #         # 'nmsi,kji,jyt->nmksty',
        #         "nmsi,kji,jyt->kstnmy",
        #         getattr(self, f"clebsch_gordan_{idx}_{i}"),
        #         self._c[str((_j, i))],
        #         getattr(self, f"id_coeffs_{idx}_{i}"),
        #     ).reshape((-1, self.out_irrep.size, self.in_irrep.size, Y_size))
        #     for i in range(getattr(self, f"nr_restrictions_{idx}"))
        # ]
        coeff = torch.cat(coeff, dim=0)
        return coeff
        # return getattr(self, f"coeff_{idx}")

    def sample_harmonics(
        self, points: Dict[Tuple, torch.Tensor], out: Dict[Tuple, torch.Tensor] = None
    ) -> Dict[Tuple, torch.Tensor]:
        if out is None:
            out = {
                j: torch.zeros(
                    (
                        points[j].shape[0],
                        self.dim_harmonic(j),
                        self.shape[0],
                        self.shape[1],
                    ),
                    device=points[j].device,
                    dtype=points[j].dtype,
                )
                for j in self.js
            }

        for j in self.js:
            if j in out:
                assert out[j].shape == (
                    points[j].shape[0],
                    self.dim_harmonic(j),
                    self.shape[0],
                    self.shape[1],
                )

        for b, j in enumerate(self.js):
            if j not in out:
                continue
            coeff = self.coeff(b)

            Ys = points[j]
            out[j].view(
                (
                    Ys.shape[0],
                    coeff.shape[0],
                    Ys.shape[1],
                    self.out_irrep.size,
                    self.in_irrep.size,
                )
            )[:] = torch.einsum(
                "dpnm,sim->sdipn",
                coeff,
                Ys,
            )

        return out

    def dim_harmonic(self, j: Tuple) -> int:
        if j in self._dim_harmonics:
            return self.basis.multiplicity(j) * self._dim_harmonics[j]
        else:
            return 0

    def attrs_j_iter(self, j: Tuple) -> Iterable:
        if self.dim_harmonic(j) == 0:
            return

        idx = self._start_index[j]

        steerable_basis_j_attr = list(self.basis.steerable_attrs_j_iter(j))

        j_attr = {
            "irrep:" + k: v for k, v in self.basis.group.irrep(*j).attributes.items()
        }

        count = 0
        for _j, _jj in self._js_restriction[j]:
            _jJl = self.group._clebsh_gordan_coeff(self.n, self.m, _j).shape[2]
            K = self.group.irrep(*_j).sum_of_squares_constituents

            for k in range(K):
                for s in range(_jJl):
                    for t in range(_jj):
                        for i, attr_i in enumerate(steerable_basis_j_attr):
                            attr = j_attr.copy()
                            attr.update(**attr_i)
                            attr["idx"] = idx
                            attr["j"] = j
                            attr["_j"] = _j
                            attr["i"] = i
                            attr["t"] = t
                            attr["s"] = s
                            attr["k"] = k

                            assert idx < self.dim
                            assert count < self.dim_harmonic(j), (
                                count,
                                self.dim_harmonic(j),
                            )

                            idx += 1
                            count += 1

                            yield attr

        assert count == self.dim_harmonic(j), (count, self.dim_harmonic(j))

    def attrs_j(self, j: Tuple, idx) -> Dict:
        assert 0 <= idx < self.dim_harmonic(j)

        full_idx = self._start_index[j] + idx

        dim = self.basis.multiplicity(j)

        for _j, _jj in self._js_restriction[j]:
            _jJl = self.group._clebsh_gordan_coeff(self.n, self.m, _j).shape[2]
            K = 1  # self.group.irrep(*_j).sum_of_squares_constituents

            d = _jj * _jJl * K * dim

            if idx >= d:
                idx -= d
            else:
                break

        i = idx % dim
        attr_i = self.basis.steerable_attrs_j(j, i)

        attr = {
            "irrep:" + k: v for k, v in self.basis.group.irrep(*j).attributes.items()
        }
        attr.update(**attr_i)
        attr["idx"] = full_idx
        attr["j"] = j
        attr["_j"] = _j
        attr["i"] = i
        attr["t"] = (idx // dim) % _jj
        attr["s"] = (idx // (dim * _jj)) % _jJl
        attr["k"] = idx // (dim * _jj * _jJl)
        return attr

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

    def __iter__(self) -> Iterable:
        for j in self.js:
            for attr in self.attrs_j_iter(j):
                yield attr
        # return chain(self.attrs_j_iter(j) for j in self.js)

    def __eq__(self, other):
        if not isinstance(other, RestrictedWignerEckartBasis):
            return False
        elif (
            self.basis != other.basis
            or self.sg_id != other.sg_id
            or self.in_irrep != other.in_irrep
            or self.out_irrep != other.out_irrep
            or self.layer_id != other.layer_id
        ):
            # TODO check isomorphism too!
            return False
        elif len(self.js) != len(other.js):
            return False
        else:
            for b, (j, i) in enumerate(zip(self.js, other.js)):
                if j != i or not torch.allclose(self.coeff(b), other.coeff(b)):
                    return False
            return True

    def __hash__(self):
        print(hash(self.sg_id))
        return (
            hash(self.basis)
            + hash(self.sg_id)
            + hash(self.in_irrep)
            + hash(self.out_irrep)
            + hash(tuple(self.js))
            + hash(self.layer_id)
        )

    _cached_instances = {}

    @classmethod
    def _generator(
        cls,
        basis: SteerableFiltersBasis,
        psi_in: Union[IrreducibleRepresentation, Tuple],
        psi_out: Union[IrreducibleRepresentation, Tuple],
        **kwargs,
    ) -> "IrrepBasis":
        assert len(kwargs) == 2
        assert "sg_id" in kwargs
        assert "layer_id" in kwargs
        layer_id = kwargs["layer_id"]

        G, _, _ = basis.group.subgroup(kwargs["sg_id"])
        psi_in = G.irrep(*G.get_irrep_id(psi_in))
        psi_out = G.irrep(*G.get_irrep_id(psi_out))

        key = (basis, psi_in.id, psi_out.id, kwargs["sg_id"], layer_id)

        if key not in cls._cached_instances:
            cls._cached_instances[key] = LearnableRestrictedWignerEckartBasis(
                basis,
                sg_id=kwargs["sg_id"],
                in_irrep=psi_in,
                out_irrep=psi_out,
                layer_id=layer_id,
            )

        return cls._cached_instances[key]


class WignerEckartBasis(IrrepBasis):
    def __init__(
        self,
        basis: SteerableFiltersBasis,
        in_irrep: Union[str, IrreducibleRepresentation, int],
        out_irrep: Union[str, IrreducibleRepresentation, int],
    ):
        r"""

        Solves the kernel constraint for a pair of input and output :math:`G`-irreps by using the Wigner-Eckart theorem
        described in Theorem 2.1 of
        `A Program to Build E(N)-Equivariant Steerable CNNs <https://openreview.net/forum?id=WE4qe9xlnQw>`_
        (see also
        `A Wigner-Eckart Theorem for Group Equivariant Convolution Kernels <https://arxiv.org/abs/2010.10952>`_
        ).
        The method relies on a :math:`G`-Steerable basis of scalar functions over the base space.

        Args:
            basis (SteerableFiltersBasis): a `G`-steerable basis for scalar functions over the base space
            in_repr (IrreducibleRepresentation): the input irrep
            out_repr (IrreducibleRepresentation): the output irrep

        """

        group = basis.group

        in_irrep = group.irrep(*group.get_irrep_id(in_irrep))
        out_irrep = group.irrep(*group.get_irrep_id(out_irrep))

        assert in_irrep.group == group
        assert out_irrep.group == group
        assert in_irrep.group == out_irrep.group

        self.m = in_irrep.id
        self.n = out_irrep.id

        _js = group._tensor_product_irreps(self.m, self.n)

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

        super(WignerEckartBasis, self).__init__(
            basis, in_irrep, out_irrep, dim, harmonics=[_j for _j, _ in _js]
        )

        # SteerableFiltersBasis: a `G`-steerable basis for scalar functions over the base space
        self.basis = basis

        _coeff = [
            torch.einsum(
                # 'mnsi,koi->mnkso',
                "mnsi,koi->ksmno",
                torch.tensor(
                    group._clebsh_gordan_coeff(self.n, self.m, j), dtype=torch.float32
                ),
                torch.tensor(group.irrep(*j).endomorphism_basis(), dtype=torch.float32),
            )
            for j in self.js
        ]

        for b, j in enumerate(self.js):
            coeff = _coeff[b]
            assert self._jJl[j] == coeff.shape[1]
            self.register_buffer(f"coeff_{b}", coeff)

    def coeff(self, idx: int) -> torch.Tensor:
        return getattr(self, f"coeff_{idx}")

    def sample_harmonics(
        self, points: Dict[Tuple, torch.Tensor], out: Dict[Tuple, torch.Tensor] = None
    ) -> Dict[Tuple, torch.Tensor]:
        if out is None:
            out = {
                j: torch.zeros(
                    (
                        points[j].shape[0],
                        self.dim_harmonic(j),
                        self.shape[0],
                        self.shape[1],
                    ),
                    device=points[j].device,
                    dtype=points[j].dtype,
                )
                for j in self.js
            }

        for j in self.js:
            if j in out:
                assert out[j].shape == (
                    points[j].shape[0],
                    self.dim_harmonic(j),
                    self.shape[0],
                    self.shape[1],
                ), (
                    out[j].shape,
                    points[j].shape[0],
                    self.dim_harmonic(j),
                    self.shape[0],
                    self.shape[1],
                )

        for b, j in enumerate(self.js):
            if j not in out:
                continue
            coeff = self.coeff(b)

            jJl = coeff.shape[1]

            Ys = points[j]

            out[j].view(
                (
                    Ys.shape[0],
                    self.group.irrep(*j).sum_of_squares_constituents,
                    jJl,
                    Ys.shape[1],
                    self.out_irrep.size,
                    self.in_irrep.size,
                )
            )[:] = torch.einsum(
                # 'Nnksm,miS->NnksiS',
                "kspnm,qim->qksipn",
                coeff,
                Ys,
            )

        return out

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

    def __iter__(self):
        for j in self.js:
            for attr in self.attrs_j_iter(j):
                yield attr
        # return chain(self.attrs_j_iter(j) for j in self.js)

    def __eq__(self, other):
        if not isinstance(other, WignerEckartBasis):
            return False
        elif (
            self.basis != other.basis
            or self.in_irrep != other.in_irrep
            or self.out_irrep != other.out_irrep
        ):
            # TODO check isomorphism too!
            return False
        elif len(self.js) != len(other.js):
            return False
        else:
            for b, (j, i) in enumerate(zip(self.js, other.js)):
                if j != i or not torch.allclose(self.coeff(b), other.coeff(b)):
                    return False
            return True

    def __hash__(self):
        return (
            hash(self.basis)
            + hash(self.in_irrep)
            + hash(self.out_irrep)
            + hash(tuple(self.js))
        )

    _cached_instances = {}

    @classmethod
    def _generator(
        cls,
        basis: SteerableFiltersBasis,
        psi_in: Union[IrreducibleRepresentation, Tuple],
        psi_out: Union[IrreducibleRepresentation, Tuple],
        **kwargs,
    ) -> "IrrepBasis":
        assert len(kwargs) == 0

        psi_in = basis.group.irrep(*basis.group.get_irrep_id(psi_in))
        psi_out = basis.group.irrep(*basis.group.get_irrep_id(psi_out))

        key = (basis, psi_in.id, psi_out.id)

        if key not in cls._cached_instances:
            cls._cached_instances[key] = WignerEckartBasis(
                basis, in_irrep=psi_in, out_irrep=psi_out
            )
        return cls._cached_instances[key]


class RestrictedWignerEckartBasis(IrrepBasis):
    def __init__(
        self,
        basis: SteerableFiltersBasis,
        sg_id: Tuple,
        in_irrep: Union[str, IrreducibleRepresentation, int],
        out_irrep: Union[str, IrreducibleRepresentation, int],
    ):
        r"""

        Solves the kernel constraint for a pair of input and output :math:`G`-irreps by using the Wigner-Eckart theorem
        described in Theorem 2.1 of
        `A Program to Build E(N)-Equivariant Steerable CNNs  <https://openreview.net/forum?id=WE4qe9xlnQw>`_.
        This method implicitly constructs the required :math:`G`-steerable basis for scalar functions on the base space
        from a :math:`G'`-steerable basis, with :math:`G' > G` a larger group, according to Equation 5 from the same
        paper.

        The equivariance group :math:`G < G'` is identified by the input id ``sg_id``.

        .. warning::
            Note that the group :math:`G'` associated with ``basis`` is generally not the same as the group :math:`G`
            associated with ``in_irrep`` and ``out_irrep`` and which the resulting kernel basis is equivariant to.

        Args:
            basis (SteerableFiltersBasis): :math:`G'`-steerable basis for scalar filters
            sg_id (tuple): id of :math:`G` as a subgroup of :math:`G'`.
            in_repr (IrreducibleRepresentation): the input `G`-irrep
            out_repr (IrreducibleRepresentation): the output `G`-irrep

        """

        # the larger group G'
        _G = basis.group
        self.basis_group = _G

        G = _G.subgroup(sg_id)[0]
        # Group: the smaller equivariance group G
        self.group = G
        self.sg_id = sg_id

        in_irrep = G.irrep(*G.get_irrep_id(in_irrep))
        out_irrep = G.irrep(*G.get_irrep_id(out_irrep))

        assert in_irrep.group == G
        assert out_irrep.group == G
        assert in_irrep.group == out_irrep.group

        self.m = in_irrep.id
        self.n = out_irrep.id

        # irreps of G in the decomposition of the tensor product of in_irrep and out_irrep
        _js_G = [j for j, _ in G._tensor_product_irreps(self.m, self.n)]

        _js = set()
        _js_restriction = defaultdict(list)

        # for each harmonic j' to consider
        for _j in set(_j for _j, _ in basis.js):
            if basis.multiplicity(_j) == 0:
                continue

            # restrict the corresponding G' irrep j' to G
            _j_G = _G.irrep(*_j).restrict(sg_id)

            # for each G-irrep j in the tensor product decomposition of in_irrep and out_irrep
            for j in _js_G:
                # irrep-decomposition coefficients of j in j'
                id_coeff = []
                p = 0
                # for each G-irrep i in the restriction of j' to G
                for i in _j_G.irreps:
                    size = G.irrep(*i).size
                    # if the restricted irrep contains one of the irreps in the tensor product
                    if i == j:
                        id_coeff.append(_j_G.change_of_basis_inv[p : p + size, :])

                    p += size

                # if the G irrep j appears in the restriction of the G'-irrep j',
                # store its irrep-decomposition coefficients
                if len(id_coeff) > 0:
                    id_coeff = np.stack(id_coeff, axis=-1)
                    _js.add(_j)
                    _js_restriction[_j].append((j, id_coeff))

        _js = sorted(list(_js))

        self._js_restriction = {}
        self._dim_harmonics = {}
        _coeffs = {}
        dim = 0
        for _j in _js:
            Y_size = _G.irrep(*_j).size
            coeff = [
                torch.einsum(
                    # 'nmsi,kji,jyt->nmksty',
                    "nmsi,kji,jyt->kstnmy",
                    torch.tensor(
                        G._clebsh_gordan_coeff(self.n, self.m, j), dtype=torch.float32
                    ),
                    torch.tensor(G.irrep(*j).endomorphism_basis(), dtype=torch.float32),
                    torch.tensor(id_coeff, dtype=torch.float32),
                ).reshape((-1, out_irrep.size, in_irrep.size, Y_size))
                for j, id_coeff in _js_restriction[_j]
            ]
            _coeffs[_j] = torch.cat(coeff, dim=0)
            self._js_restriction[_j] = [
                (j, id_coeff.shape[2]) for j, id_coeff in _js_restriction[_j]
            ]
            self._dim_harmonics[_j] = _coeffs[_j].shape[0]
            dim += self._dim_harmonics[_j] * basis.multiplicity(_j)

        # self._cg = []
        # _c = []
        # self._id_coeffs = []
        # for _j in _js:
        #     Y_size = _G.irrep(*_j).size
        #     self._cg.append(
        #         [
        #             torch.tensor(
        #                 G._clebsh_gordan_coeff(self.n, self.m, j), dtype=torch.float32
        #             )
        #             for j, _ in _js_restriction[_j]
        #         ]
        #     )
        #     _c.append(
        #         [
        #             torch.nn.Parameter(
        #                 torch.tensor(
        #                     G.irrep(*j).endomorphism_basis(), dtype=torch.float32
        #                 )
        #             )
        #             for j, _ in _js_restriction[_j]
        #         ]
        #     )
        #     self._id_coeffs.append(
        #         [
        #             torch.tensor(id_coeff, dtype=torch.float32)
        #             for _, id_coeff in _js_restriction[_j]
        #         ]
        #     )

        #     self._js_restriction[_j] = [
        #         (j, id_coeff.shape[2]) for j, id_coeff in _js_restriction[_j]
        #     ]
        #     n, m, s, i = self._cg[-1][-1].shape
        #     k, j, i = _c[-1][-1].shape
        #     j, y, t = self._id_coeffs[-1][-1].shape
        #     out = int(k * s * t * n * m * y / (out_irrep.size * in_irrep.size * Y_size))
        #     self._dim_harmonics[_j] = len(_js_restriction[_j]) * out

        super(RestrictedWignerEckartBasis, self).__init__(
            basis, in_irrep, out_irrep, dim, harmonics=_js
        )

        # self._c = torch.nn.ParameterDict(
        #     {
        #         str((_j, i)): _c[k][i]
        #         for k, _j in enumerate(_js)
        #         for i in range(len(_js_restriction[_j]))
        #     }
        # )

        # SteerableFiltersBasis: a `G'`-steerable basis for scalar functions over the base space, for the larger
        # group `G' > G`
        self.basis = basis

        for b, _j in enumerate(self.js):
            self.register_buffer(f"coeff_{b}", _coeffs[_j])
            # for i in range(len(_js_restriction[_j])):
            #     self.register_buffer(f"id_coeffs_{b}_{i}", _id_coeffs[b][i])
            #     self.register_buffer(f"clebsch_gordan_{b}_{i}", _id_coeffs[b][i])

    def coeff(self, idx: int) -> torch.Tensor:
        # _j = self.js[idx]
        # Y_size = self.basis_group.irrep(*_j).size
        # coeff = [
        #     torch.einsum(
        #         # 'nmsi,kji,jyt->nmksty',
        #         "nmsi,kji,jyt->kstnmy",
        #         cg,
        #         self._c[str((_j, i))],
        #         coeff_id,
        #     ).reshape((-1, self.out_irrep.size, self.in_irrep.size, Y_size))
        #     for i, (cg, coeff_id) in enumerate(zip(self._cg[idx], self._id_coeffs[idx]))
        # ]
        # coeff = torch.cat(coeff, dim=0)
        # coeff2 =
        return getattr(self, f"coeff_{idx}")

    def sample_harmonics(
        self, points: Dict[Tuple, torch.Tensor], out: Dict[Tuple, torch.Tensor] = None
    ) -> Dict[Tuple, torch.Tensor]:
        if out is None:
            out = {
                j: torch.zeros(
                    (
                        points[j].shape[0],
                        self.dim_harmonic(j),
                        self.shape[0],
                        self.shape[1],
                    ),
                    device=points[j].device,
                    dtype=points[j].dtype,
                )
                for j in self.js
            }

        for j in self.js:
            if j in out:
                assert out[j].shape == (
                    points[j].shape[0],
                    self.dim_harmonic(j),
                    self.shape[0],
                    self.shape[1],
                )

        for b, j in enumerate(self.js):
            if j not in out:
                continue
            coeff = self.coeff(b)

            Ys = points[j]
            out[j].view(
                (
                    Ys.shape[0],
                    coeff.shape[0],
                    Ys.shape[1],
                    self.out_irrep.size,
                    self.in_irrep.size,
                )
            )[:] = torch.einsum(
                "dpnm,sim->sdipn",
                coeff,
                Ys,
            )

        return out

    def dim_harmonic(self, j: Tuple) -> int:
        if j in self._dim_harmonics:
            return self.basis.multiplicity(j) * self._dim_harmonics[j]
        else:
            return 0

    def attrs_j_iter(self, j: Tuple) -> Iterable:
        if self.dim_harmonic(j) == 0:
            return

        idx = self._start_index[j]

        steerable_basis_j_attr = list(self.basis.steerable_attrs_j_iter(j))

        j_attr = {
            "irrep:" + k: v for k, v in self.basis.group.irrep(*j).attributes.items()
        }

        count = 0
        for _j, _jj in self._js_restriction[j]:
            _jJl = self.group._clebsh_gordan_coeff(self.n, self.m, _j).shape[2]
            K = self.group.irrep(*_j).sum_of_squares_constituents

            for k in range(K):
                for s in range(_jJl):
                    for t in range(_jj):
                        for i, attr_i in enumerate(steerable_basis_j_attr):
                            attr = j_attr.copy()
                            attr.update(**attr_i)
                            attr["idx"] = idx
                            attr["j"] = j
                            attr["_j"] = _j
                            attr["i"] = i
                            attr["t"] = t
                            attr["s"] = s
                            attr["k"] = k

                            assert idx < self.dim
                            assert count < self.dim_harmonic(j), (
                                count,
                                self.dim_harmonic(j),
                            )

                            idx += 1
                            count += 1

                            yield attr

        assert count == self.dim_harmonic(j), (count, self.dim_harmonic(j))

    def attrs_j(self, j: Tuple, idx) -> Dict:
        assert 0 <= idx < self.dim_harmonic(j)

        full_idx = self._start_index[j] + idx

        dim = self.basis.multiplicity(j)

        for _j, _jj in self._js_restriction[j]:
            _jJl = self.group._clebsh_gordan_coeff(self.n, self.m, _j).shape[2]
            K = self.group.irrep(*_j).sum_of_squares_constituents

            d = _jj * _jJl * K * dim

            if idx >= d:
                idx -= d
            else:
                break

        i = idx % dim
        attr_i = self.basis.steerable_attrs_j(j, i)

        attr = {
            "irrep:" + k: v for k, v in self.basis.group.irrep(*j).attributes.items()
        }
        attr.update(**attr_i)
        attr["idx"] = full_idx
        attr["j"] = j
        attr["_j"] = _j
        attr["i"] = i
        attr["t"] = (idx // dim) % _jj
        attr["s"] = (idx // (dim * _jj)) % _jJl
        attr["k"] = idx // (dim * _jj * _jJl)
        return attr

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

    def __iter__(self) -> Iterable:
        for j in self.js:
            for attr in self.attrs_j_iter(j):
                yield attr
        # return chain(self.attrs_j_iter(j) for j in self.js)

    def __eq__(self, other):
        if not isinstance(other, RestrictedWignerEckartBasis):
            return False
        elif (
            self.basis != other.basis
            or self.sg_id != other.sg_id
            or self.in_irrep != other.in_irrep
            or self.out_irrep != other.out_irrep
        ):
            # TODO check isomorphism too!
            return False
        elif len(self.js) != len(other.js):
            return False
        else:
            for b, (j, i) in enumerate(zip(self.js, other.js)):
                if j != i or not torch.allclose(self.coeff(b), other.coeff(b)):
                    return False
            return True

    def __hash__(self):
        return (
            hash(self.basis)
            + hash(self.sg_id)
            + hash(self.in_irrep)
            + hash(self.out_irrep)
            + hash(tuple(self.js))
        )

    _cached_instances = {}

    @classmethod
    def _generator(
        cls,
        basis: SteerableFiltersBasis,
        psi_in: Union[IrreducibleRepresentation, Tuple],
        psi_out: Union[IrreducibleRepresentation, Tuple],
        **kwargs,
    ) -> "IrrepBasis":
        assert len(kwargs) == 1
        assert "sg_id" in kwargs

        G, _, _ = basis.group.subgroup(kwargs["sg_id"])
        psi_in = G.irrep(*G.get_irrep_id(psi_in))
        psi_out = G.irrep(*G.get_irrep_id(psi_out))

        key = (basis, psi_in.id, psi_out.id, kwargs["sg_id"])

        if key not in cls._cached_instances:
            cls._cached_instances[key] = RestrictedWignerEckartBasis(
                basis,
                sg_id=kwargs["sg_id"],
                in_irrep=psi_in,
                out_irrep=psi_out,
            )

        return cls._cached_instances[key]


class LearnableWignerEckartBasisOLDOLD(IrrepBasis):
    def __init__(
        self,
        basis: SteerableFiltersBasis,
        in_irrep: Union[str, IrreducibleRepresentation, int],
        out_irrep: Union[str, IrreducibleRepresentation, int],
        layer_id: int = None,
    ):
        r"""
        Solves the kernel constraint for a pair of input and output :math:`G`-irreps by using the Wigner-Eckart theorem
        described in Theorem 2.1 of
        `A Program to Build E(N)-Equivariant Steerable CNNs <https://openreview.net/forum?id=WE4qe9xlnQw>`_
        (see also
        `A Wigner-Eckart Theorem for Group Equivariant Convolution Kernels <https://arxiv.org/abs/2010.10952>`_
        ).
        The method relies on a :math:`G`-Steerable basis of scalar functions over the base space.
        Args:
            basis (SteerableFiltersBasis): a `G`-steerable basis for scalar functions over the base space
            in_repr (IrreducibleRepresentation): the input irrep
            out_repr (IrreducibleRepresentation): the output irrep
        """
        self.layer_id = layer_id
        group = basis.group

        in_irrep = group.irrep(*group.get_irrep_id(in_irrep))
        out_irrep = group.irrep(*group.get_irrep_id(out_irrep))

        assert in_irrep.group == group
        assert out_irrep.group == group
        assert in_irrep.group == out_irrep.group

        self.m = in_irrep.id
        self.n = out_irrep.id

        # print(self.m)
        # print(self.n)
        _js = group._tensor_product_irreps(self.m, self.n)
        # print(self.m, self.n, _js)

        _js = [(j, jJl) for j, jJl in _js if basis.multiplicity(j) > 0]

        dim = 0
        self._dim_harmonics = {}
        self._jJl = {}
        for j, jJl in _js:
            self._dim_harmonics[j] = (
                basis.multiplicity(j)
                * jJl
                * 1  # group.irrep(*j).sum_of_squares_constituents
            )
            self._jJl[j] = jJl
            dim += self._dim_harmonics[j]

        super(LearnableWignerEckartBasis, self).__init__(
            basis, in_irrep, out_irrep, dim, harmonics=[_j for _j, _ in _js]
        )
        # SteerableFiltersBasis: a `G`-steerable basis for scalar functions over the base space
        self.basis = basis

        # _coeff = [
        #     torch.einsum(
        #         # 'mnsi,koi->mnkso',
        #         "mnsi,koi->ksmno",
        #         torch.tensor(
        #             group._clebsh_gordan_coeff(self.n, self.m, j), dtype=torch.float32
        #         ),
        #         torch.tensor(group.irrep(*j).endomorphism_basis(), dtype=torch.float32),
        #     )
        #     for j in self.js
        # ]

        _cg = [
            torch.tensor(
                group._clebsh_gordan_coeff(self.n, self.m, j), dtype=torch.float32
            )
            for j in self.js
        ]

        _c = dict()
        for j in self.js:
            for basis_j, jJl in self.basis.js:
                size = group.irrep(*j).size
                param = (
                    torch.tensor(
                        group.irrep(*j).endomorphism_basis(), dtype=torch.float32
                    )
                    if j == basis_j
                    else torch.zeros((1, size, size))
                )
                _c[f"equivariance layer {self.layer_id} {(j, basis_j)}"] = (
                    torch.nn.Parameter(param)
                )

        self._c2 = torch.nn.ParameterDict(_c)

        # for i, _ in enumerate(_cg):
        #     print(_cg[i], _cg[i].shape)
        #     # _cg[i] = torch.ones(1, 1, 1, 1)

        self._c = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.tensor(
                        group.irrep(*j).endomorphism_basis(), dtype=torch.float32
                    )
                )
                for j in self.js
            ]
        )

        # for j in self.js:
        #     c = group.irrep(*j).endomorphism_basis()
        #     cg = group._clebsh_gordan_coeff(self.n, self.m, j)
        #     print(self.m, self.n, j)
        #     print(c, c.shape)
        #     print(cg, cg.shape)
        #     print()

        # print("stuff and things")
        # for i, j in enumerate(self.js):
        #     print(
        #         self.m,
        #         self.n,
        #         j,
        #         "\n",
        #         group._clebsh_gordan_coeff(self.n, self.m, j),
        #         "\n",
        #         self._c[i],
        #     )

        # print("\n")

        # for b, j in enumerate(self.js):
        #     coeff = _coeff[b]
        #     assert self._jJl[j] == coeff.shape[1]
        #     self.register_buffer(f"coeff_{b}", coeff)

        for b, j in enumerate(self.js):
            self.register_buffer(f"clebsch_gordan_{b}", _cg[b])

    def coeff(self, idx: int) -> torch.Tensor:
        # return getattr(self, f"coeff_{idx}")
        clebsch = getattr(self, f"clebsch_gordan_{idx}")
        return torch.einsum("mnsi,koi->ksmno", clebsch, self._c[idx])

    def sample_harmonics(
        self, points: Dict[Tuple, torch.Tensor], out: Dict[Tuple, torch.Tensor] = None
    ) -> Dict[Tuple, torch.Tensor]:
        if out is None:
            out = {
                j: torch.zeros(
                    (
                        points[j].shape[0],
                        self.dim_harmonic(j),
                        self.shape[0],
                        self.shape[1],
                    ),
                    device=points[j].device,
                    dtype=points[j].dtype,
                )
                for j in self.js
            }

        for j in self.js:
            if j in out:
                assert out[j].shape == (
                    points[j].shape[0],
                    self.dim_harmonic(j),
                    self.shape[0],
                    self.shape[1],
                ), (
                    out[j].shape,
                    points[j].shape[0],
                    self.dim_harmonic(j),
                    self.shape[0],
                    self.shape[1],
                )

        for b, j in enumerate(self.js):
            if j not in out:
                continue
            coeff = self.coeff(b)

            jJl = coeff.shape[1]

            Ys = points[j]

            out[j].view(
                (
                    Ys.shape[0],
                    1,  # self.group.irrep(*j).sum_of_squares_constituents,
                    jJl,
                    Ys.shape[1],
                    self.out_irrep.size,
                    self.in_irrep.size,
                )
            )[:] = torch.einsum(
                # 'Nnksm,miS->NnksiS',
                "kspnm,qim->qksipn",
                coeff,
                Ys,
            )
        return out

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
        nr_c_mats = 1  # self.group.irrep(*j).sum_of_squares_constituents
        for k in range(nr_c_mats):
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

    def __iter__(self):
        for j in self.js:
            for attr in self.attrs_j_iter(j):
                yield attr
        # return chain(self.attrs_j_iter(j) for j in self.js)

    def __eq__(self, other):
        if not isinstance(other, LearnableWignerEckartBasis):
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
            for b, (j, i) in enumerate(zip(self.js, other.js)):
                if j != i or not torch.allclose(self.coeff(b), other.coeff(b)):
                    return False
            return True

    def __hash__(self):
        return (
            hash(self.basis)
            + hash(self.in_irrep)
            + hash(self.out_irrep)
            + hash(tuple(self.js))
            + hash(self.layer_id)
        )

    _cached_instances = {}

    @classmethod
    def _generator(
        cls,
        basis: SteerableFiltersBasis,
        psi_in: Union[IrreducibleRepresentation, Tuple],
        psi_out: Union[IrreducibleRepresentation, Tuple],
        **kwargs,
    ) -> "IrrepBasis":
        assert len(kwargs) == 1
        assert "layer_id" in kwargs
        layer_id = kwargs["layer_id"]

        psi_in = basis.group.irrep(*basis.group.get_irrep_id(psi_in))
        psi_out = basis.group.irrep(*basis.group.get_irrep_id(psi_out))

        key = (basis, psi_in.id, psi_out.id, layer_id)

        if key not in cls._cached_instances:
            cls._cached_instances[key] = LearnableWignerEckartBasis(
                basis, in_irrep=psi_in, out_irrep=psi_out, layer_id=layer_id
            )
        return cls._cached_instances[key]


class LearnableWignerEckartBasisOLD(IrrepBasis):
    def __init__(
        self,
        basis: SteerableFiltersBasis,
        in_irrep: Union[str, IrreducibleRepresentation, int],
        out_irrep: Union[str, IrreducibleRepresentation, int],
        layer_id: int = None,
    ):
        r"""
        Solves the kernel constraint for a pair of input and output :math:`G`-irreps by using the Wigner-Eckart theorem
        described in Theorem 2.1 of
        `A Program to Build E(N)-Equivariant Steerable CNNs <https://openreview.net/forum?id=WE4qe9xlnQw>`_
        (see also
        `A Wigner-Eckart Theorem for Group Equivariant Convolution Kernels <https://arxiv.org/abs/2010.10952>`_
        ).
        The method relies on a :math:`G`-Steerable basis of scalar functions over the base space.
        Args:
            basis (SteerableFiltersBasis): a `G`-steerable basis for scalar functions over the base space
            in_repr (IrreducibleRepresentation): the input irrep
            out_repr (IrreducibleRepresentation): the output irrep
        """
        self.layer_id = layer_id
        group = basis.group

        in_irrep = group.irrep(*group.get_irrep_id(in_irrep))
        out_irrep = group.irrep(*group.get_irrep_id(out_irrep))

        assert in_irrep.group == group
        assert out_irrep.group == group
        assert in_irrep.group == out_irrep.group

        self.m = in_irrep.id
        self.n = out_irrep.id

        self._js_cg = group._tensor_product_irreps(self.m, self.n)

        self._js_cg = [(j, jJl) for j, jJl in self._js_cg if basis.multiplicity(j) > 0]

        _js = basis.js

        _js = [(j, jJl) for j, jJl in _js if basis.multiplicity(j) > 0]

        dim = 0
        self._dim_harmonics = {}
        self._jJl = {}
        for j, jJl in self._js_cg:
            self._dim_harmonics[j] = (
                basis.multiplicity(j)
                * jJl
                * 1  # group.irrep(*j).sum_of_squares_constituents
            )
            self._jJl[j] = jJl
            dim += self._dim_harmonics[j]

        super(LearnableWignerEckartBasis, self).__init__(
            basis, in_irrep, out_irrep, dim, harmonics=[_j for _j, _ in _js]
        )
        # SteerableFiltersBasis: a `G`-steerable basis for scalar functions over the base space
        self.basis = basis

        # _coeff = [
        #     torch.einsum(
        #         # 'mnsi,koi->mnkso',
        #         "mnsi,koi->ksmno",
        #         torch.tensor(
        #             group._clebsh_gordan_coeff(self.n, self.m, j), dtype=torch.float32
        #         ),
        #         torch.tensor(group.irrep(*j).endomorphism_basis(), dtype=torch.float32),
        #     )
        #     for j in self.js
        # ]

        _cg = [
            torch.tensor(
                group._clebsh_gordan_coeff(self.n, self.m, j), dtype=torch.float32
            )
            for j, _ in self._js_cg
        ]

        self._irrepmap = IrrepsMap._generator(self.layer_id, basis)
        self._c = dict()
        for _, (j, _) in enumerate(self._js_cg):
            for _, basis_j in enumerate(self.js):
                param = self._irrepmap(basis_j, j)
                self._c[f"equivariance layer {self.layer_id} {(basis_j, j)}"] = param

        self._c = torch.nn.ParameterDict(self._c)

        # for i, _ in enumerate(_cg):
        #     print(_cg[i], _cg[i].shape)
        #     # _cg[i] = torch.ones(1, 1, 1, 1)

        # self._c = torch.nn.ParameterList(
        #     [
        #         torch.nn.Parameter(
        #             torch.tensor(
        #                 group.irrep(*j).endomorphism_basis(), dtype=torch.float32
        #             )
        #         )
        #         for j in self.js
        #     ]
        # )

        # for j in self.js:
        #     c = group.irrep(*j).endomorphism_basis()
        #     cg = group._clebsh_gordan_coeff(self.n, self.m, j)
        #     print(self.m, self.n, j)
        #     print(c, c.shape)
        #     print(cg, cg.shape)
        #     print()

        # print("stuff and things")
        # for i, j in enumerate(self.js):
        #     print(
        #         self.m,
        #         self.n,
        #         j,
        #         "\n",
        #         group._clebsh_gordan_coeff(self.n, self.m, j),
        #         "\n",
        #         self._c[i],
        #     )

        # print("\n")

        # for b, j in enumerate(self.js):
        #     coeff = _coeff[b]
        #     assert self._jJl[j] == coeff.shape[1]
        #     self.register_buffer(f"coeff_{b}", coeff)

        for b, j in enumerate(self._js_cg):
            self.register_buffer(f"clebsch_gordan_{b}", _cg[b])

    def coeff(self, idx_basis: int, idx_cg_j: int) -> torch.Tensor:
        # return getattr(self, f"coeff_{idx}")
        clebsch = getattr(self, f"clebsch_gordan_{idx_cg_j}")
        basis_j, (j_cg, _) = self.js[idx_basis], self._js_cg[idx_cg_j]
        key = f"equivariance layer {self.layer_id} {(basis_j, j_cg)}"
        return torch.einsum("mnsi,koi->ksmno", clebsch, self._c[key])

        # return torch.einsum("mnsi,koi->ksmno", clebsch, self._irrepmap(j_basis, j_cg))

    def sample_harmonics(
        self, points: Dict[Tuple, torch.Tensor], out: Dict[Tuple, torch.Tensor] = None
    ) -> Dict[Tuple, torch.Tensor]:
        if out is None:
            out = {
                j: torch.zeros(
                    (
                        points[j].shape[0],
                        self.dim_harmonic(j),
                        self.shape[0],
                        self.shape[1],
                    ),
                    device=points[j].device,
                    dtype=points[j].dtype,
                )
                for j in self._js_cg
            }

        for j in self._js_cg:
            if j in out:
                assert out[j].shape == (
                    points[j].shape[0],
                    self.dim_harmonic(j),
                    self.shape[0],
                    self.shape[1],
                ), (
                    out[j].shape,
                    points[j].shape[0],
                    self.dim_harmonic(j),
                    self.shape[0],
                    self.shape[1],
                )
        for b_cg, (j_cg, _) in enumerate(self._js_cg):
            if j_cg not in out:
                continue
            for b_basis, j_basis in enumerate(self.js):
                coeff = self.coeff(b_basis, b_cg)

                jJl = coeff.shape[1]

                Ys = points[j_basis]

                out[j_cg].view(
                    (
                        Ys.shape[0],
                        1,  # self.group.irrep(*j).sum_of_squares_constituents,
                        jJl,
                        Ys.shape[1],
                        self.out_irrep.size,
                        self.in_irrep.size,
                    )
                )[:] += torch.einsum(
                    # 'Nnksm,miS->NnksiS',
                    "kspnm,qim->qksipn",
                    coeff,
                    Ys,
                )

        return out

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
        nr_c_mats = 1  # self.group.irrep(*j).sum_of_squares_constituents
        for k in range(nr_c_mats):
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

    def __iter__(self):
        for j in self.js:
            for attr in self.attrs_j_iter(j):
                yield attr
        # return chain(self.attrs_j_iter(j) for j in self.js)

    def __eq__(self, other):
        if not isinstance(other, LearnableWignerEckartBasis):
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
            for b, (j, i) in enumerate(zip(self.js, other.js)):
                if j != i or not torch.allclose(self.coeff(b), other.coeff(b)):
                    return False
            return True

    def __hash__(self):
        return (
            hash(self.basis)
            + hash(self.in_irrep)
            + hash(self.out_irrep)
            + hash(tuple(self.js))
            + hash(self.layer_id)
        )

    _cached_instances = {}

    @classmethod
    def _generator(
        cls,
        basis: SteerableFiltersBasis,
        psi_in: Union[IrreducibleRepresentation, Tuple],
        psi_out: Union[IrreducibleRepresentation, Tuple],
        **kwargs,
    ) -> "IrrepBasis":
        assert len(kwargs) == 1
        assert "layer_id" in kwargs
        layer_id = kwargs["layer_id"]
        psi_in = basis.group.irrep(*basis.group.get_irrep_id(psi_in))
        psi_out = basis.group.irrep(*basis.group.get_irrep_id(psi_out))

        key = (basis, psi_in.id, psi_out.id, layer_id)

        if key not in cls._cached_instances:
            cls._cached_instances[key] = LearnableWignerEckartBasis(
                basis, in_irrep=psi_in, out_irrep=psi_out, layer_id=layer_id
            )
        return cls._cached_instances[key]
