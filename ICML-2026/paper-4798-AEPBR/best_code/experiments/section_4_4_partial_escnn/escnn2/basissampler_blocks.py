from escnn.group import Representation
from escnn.kernels import KernelBasis, EmptyBasisException


from escnn.nn.modules.basismanager import retrieve_indices
from escnn.nn.modules.basismanager import BasisManager

from .basissampler_singleblock import block_basissampler

from typing import Callable, Tuple, Dict, List, Iterable, Union
from collections import defaultdict

import torch
import numpy as np
import math


__all__ = ["BlocksBasisSampler"]


class BlocksBasisSampler(torch.nn.Module, BasisManager):
    def __init__(
        self,
        in_reprs: List[Representation],
        out_reprs: List[Representation],
        basis_generator: Callable[[Representation, Representation], KernelBasis],
        basis_filter: Callable[[dict], bool] = None,
        recompute: bool = False,
    ):
        r"""

        Module which performs the expansion of an analytical filter basis and samples it on arbitrary input points.

        Args:
            in_reprs (list): the input field type
            out_reprs (list): the output field type
            basis_generator (callable): method that generates the analytical filter basis
            basis_filter (callable, optional): filter for the basis elements. Should take a dictionary containing an
                                               element's attributes and return whether to keep it or not.
            recompute (bool, optional): whether to recompute new bases or reuse, if possible, already built tensors.

        """

        super(BlocksBasisSampler, self).__init__()

        self._in_reprs = in_reprs
        self._out_reprs = out_reprs
        self._input_size = sum(r.size for r in in_reprs)
        self._output_size = sum(r.size for r in out_reprs)

        self._in_sizes = {r.name: r.size for r in in_reprs}

        uniform_input = all(r == in_reprs[0] for r in in_reprs)
        uniform_output = all(r == out_reprs[0] for r in out_reprs)
        _uniform = uniform_input and uniform_output

        # we group the bases by their input and output representations
        _block_sampler_modules = {}

        # iterate through all different pairs of input/output representations
        # and, for each of them, build a basis
        for i_repr in set(in_reprs):
            for o_repr in set(out_reprs):
                reprs_names = (i_repr.name, o_repr.name)
                try:
                    basis = basis_generator(i_repr, o_repr)

                    # BasisSamplerSingleBlock: sampler for block with input i_repr and output o_repr
                    block_sampler = block_basissampler(
                        basis, basis_filter=basis_filter, recompute=recompute
                    )
                    _block_sampler_modules[reprs_names] = block_sampler

                    # register the block sampler as a submodule
                    self.add_module(f"block_sampler_{reprs_names}", block_sampler)

                except EmptyBasisException:
                    # print(f"Empty basis at {reprs_names}")
                    pass

        if len(_block_sampler_modules) == 0:
            print("WARNING! The basis for the block sampler of the filter is empty!")

        # the list of all pairs of input/output representations which don't have an empty basis
        self._representations_pairs = sorted(list(_block_sampler_modules.keys()))

        self._n_pairs = len(set(in_reprs)) * len(set(out_reprs))
        if _uniform:
            assert self._n_pairs <= 1
        self._uniform = _uniform and self._n_pairs == 1

        # retrieve for each representation in both input and output fields:
        # - the number of its occurrences,
        # - the indices where it occurs and
        # - whether its occurrences are contiguous or not
        self._in_count, _in_indices, _in_contiguous = retrieve_indices(in_reprs)
        self._out_count, _out_indices, _out_contiguous = retrieve_indices(out_reprs)

        self._weights_ranges = {}

        last_weight_position = 0

        self._contiguous = {}

        # iterate through the different groups of blocks
        # i.e., through all input/output pairs
        for io_pair in self._representations_pairs:
            self._contiguous[io_pair] = (
                _in_contiguous[io_pair[0]] and _out_contiguous[io_pair[1]]
            )

            # build the indices tensors
            if self._contiguous[io_pair]:
                in_indices = [
                    _in_indices[io_pair[0]].min(),
                    _in_indices[io_pair[0]].max() + 1,
                    _in_indices[io_pair[0]].max() + 1 - _in_indices[io_pair[0]].min(),
                ]
                out_indices = [
                    _out_indices[io_pair[1]].min(),
                    _out_indices[io_pair[1]].max() + 1,
                    _out_indices[io_pair[1]].max() + 1 - _out_indices[io_pair[1]].min(),
                ]

                setattr(self, "in_indices_{}".format(io_pair), in_indices)
                setattr(self, "out_indices_{}".format(io_pair), out_indices)

            else:
                out_indices, in_indices = (
                    _out_indices[io_pair[1]],
                    _in_indices[io_pair[0]],
                )
                # out_indices, in_indices = torch.meshgrid([_out_indices[io_pair[1]], _in_indices[io_pair[0]]])
                # in_indices = in_indices.reshape(-1)
                # out_indices = out_indices.reshape(-1)

                # register the indices tensors and the bases tensors as parameters of this module
                self.register_buffer("in_indices_{}".format(io_pair), in_indices)
                self.register_buffer("out_indices_{}".format(io_pair), out_indices)

            # number of occurrences of the input/output pair `io_pair`
            n_pairs = self._in_count[io_pair[0]] * self._out_count[io_pair[1]]
            # count the actual number of parameters
            total_weights = _block_sampler_modules[io_pair].dimension() * n_pairs

            # evaluate the indices in the global weights tensor to use for the basis belonging to this group
            self._weights_ranges[io_pair] = (
                last_weight_position,
                last_weight_position + total_weights,
            )

            # increment the position counter
            last_weight_position += total_weights

        self._dim = last_weight_position

    def normalize(self, points: torch.Tensor):
        """
        Normalize the block samplers using points.
        """
        for io_pair in self._representations_pairs:
            block_sampler = getattr(self, f"block_sampler_{io_pair}")
            block_sampler.normalize(points)

    def get_element_info(self, idx: int) -> Dict:
        assert 0 <= idx < self._dim, idx

        reprs_names = None
        relative_idx = None
        for pair, idx_range in self._weights_ranges.items():
            if idx_range[0] <= idx < idx_range[1]:
                reprs_names = pair
                relative_idx = idx - idx_range[0]
                break
        assert reprs_names is not None and relative_idx is not None

        block_sampler = getattr(self, f"block_sampler_{reprs_names}")
        block_idx = relative_idx // block_sampler.dimension()
        relative_idx = relative_idx % block_sampler.dimension()

        attr = block_sampler.get_element_info(relative_idx).copy()

        block_count = 0
        out_irreps_count = 0
        for o, o_repr in enumerate(self._out_reprs):
            in_irreps_count = 0
            for i, i_repr in enumerate(self._in_reprs):
                if reprs_names == (i_repr.name, o_repr.name):
                    if block_count == block_idx:
                        # retrieve the attributes of each basis element and build a new list of
                        # attributes adding information specific to the current block
                        attr.update(
                            {
                                "in_irreps_position": in_irreps_count
                                + attr["in_irrep_idx"],
                                "out_irreps_position": out_irreps_count
                                + attr["out_irrep_idx"],
                                "in_repr": reprs_names[0],
                                "out_repr": reprs_names[1],
                                "in_field_position": i,
                                "out_field_position": o,
                            }
                        )

                        attr["block_id"] = attr["id"]
                        attr["id"] = idx

                        return attr

                    block_count += 1

                in_irreps_count += len(i_repr.irreps)
            out_irreps_count += len(o_repr.irreps)

        raise ValueError(f"Parameter with index {idx} not found!")

    def get_basis_info(self) -> Iterable[Dict]:
        out_irreps_counts = [0]
        out_block_counts = defaultdict(list)
        for o, o_repr in enumerate(self._out_reprs):
            out_irreps_counts.append(out_irreps_counts[-1] + len(o_repr.irreps))
            out_block_counts[o_repr.name].append(o)

        in_irreps_counts = [0]
        in_block_counts = defaultdict(list)
        for i, i_repr in enumerate(self._in_reprs):
            in_irreps_counts.append(in_irreps_counts[-1] + len(i_repr.irreps))
            in_block_counts[i_repr.name].append(i)

        # iterate through the different group of blocks
        # i.e., through all input/output pairs
        idx = 0
        for reprs_names in self._representations_pairs:
            block_sampler = getattr(self, f"block_sampler_{reprs_names}")

            # since this method returns an iterable of attributes built on the fly, load all attributes first and then
            # iterate on this list
            attrs = list(block_sampler.get_basis_info())

            for o in out_block_counts[reprs_names[1]]:
                out_irreps_count = out_irreps_counts[o]
                for i in in_block_counts[reprs_names[0]]:
                    in_irreps_count = in_irreps_counts[i]

                    # retrieve the attributes of each basis element and build a new list of
                    # attributes adding information specific to the current block
                    for attr in attrs:
                        attr = attr.copy()
                        attr.update(
                            {
                                "in_irreps_position": in_irreps_count
                                + attr["in_irrep_idx"],
                                "out_irreps_position": out_irreps_count
                                + attr["out_irrep_idx"],
                                "in_repr": reprs_names[0],
                                "out_repr": reprs_names[1],
                                "in_field_position": i,
                                "out_field_position": o,
                            }
                        )

                        # build the ids of the basis vectors
                        # add names and indices of the input and output fields
                        # id = '({}-{},{}-{})'.format(reprs_names[0], i, reprs_names[1], o)
                        # # add the original id in the block submodule
                        # id += "_" + attr["id"]
                        #
                        # # update with the new id
                        # attr["id"] = id
                        #
                        # attr["idx"] = idx

                        attr["block_id"] = attr["id"]
                        attr["id"] = idx

                        assert idx < self._dim

                        idx += 1

                        yield attr

    def dimension(self) -> int:
        return self._dim

    def _compute_out_block(
        self, weights: torch.Tensor, input: torch.Tensor, points: torch.Tensor, io_pair
    ) -> torch.Tensor:
        # retrieve the basis
        block_sampler = getattr(self, f"block_sampler_{io_pair}")

        # retrieve the linear coefficients for the basis sampler
        coefficients = weights[
            self._weights_ranges[io_pair][0] : self._weights_ranges[io_pair][1]
        ]

        # reshape coefficients for the batch matrix multiplication
        coefficients = coefficients.view(
            self._out_count[io_pair[1]],  # u
            self._in_count[io_pair[0]],  # j
            block_sampler.dimension(),  # k
        )

        # expand the current subset of basis vectors and set the result in the appropriate place in the filter
        _basis = block_sampler(points)
        # p, o, i, k = _basis.shape

        # TODO: torch.einsum does not optimize the order of operations. Need to do this manually!
        _out = torch.einsum(
            "ujk,poik,pji->puo",
            coefficients,
            _basis,
            input,
        )

        return _out

    def _contract_basis_block(
        self, weights: torch.Tensor, points: torch.Tensor, io_pair
    ) -> torch.Tensor:
        # retrieve the basis
        block_sampler = getattr(self, f"block_sampler_{io_pair}")

        # retrieve the linear coefficients for the basis sampler
        coefficients = weights[
            self._weights_ranges[io_pair][0] : self._weights_ranges[io_pair][1]
        ]

        # reshape coefficients for the batch matrix multiplication
        coefficients = coefficients.view(
            self._out_count[io_pair[1]],  # u
            self._in_count[io_pair[0]],  # j
            block_sampler.dimension(),  # k
        )

        # expand the current subset of basis vectors and set the result in the appropriate place in the filter
        _basis = block_sampler(points)
        # p, o, i, k = _basis.shape

        _filter = torch.einsum(
            "ujk,poik->pujoi",
            coefficients,
            _basis,
        ).permute(0, 1, 3, 2, 4)

        return _filter

    def forward(self, weights: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Forward step of the Module which expands the basis, samples it on the input `points` and returns
        the filter built.

        Args:
            weights (torch.Tensor): the learnable weights used to linearly combine the basis filters
            points (torch.Tensor): the points where the filter should be sampled

        Returns:
            the filter built

        """
        assert weights.shape[0] == self.dimension(), (
            weights.shape[0],
            self.dimension(),
        )
        assert len(weights.shape) == 1

        S = points.shape[0]

        if self._uniform:
            # if there is only one block (i.e. one type of input field and one type of output field),
            # we can return the computed block immediately, instead of copying it inside a preallocated empty tensor
            io_pair = self._representations_pairs[0]
            _filter = self._contract_basis_block(weights, points, io_pair)
            _filter = _filter.reshape(S, self._output_size, self._input_size)

        else:
            # to support Automatic Mixed Precision (AMP), we can not preallocate the output tensor with a specific dtype
            # Instead, we check the dtype of the first `expanded` block. For this reason, we postpose the allocation
            # of the full _filter tensor

            _filter = None

            # iterate through all input-output field representations pairs
            for io_pair in self._representations_pairs:
                # retrieve the indices
                in_indices = getattr(self, f"in_indices_{io_pair}")
                out_indices = getattr(self, f"out_indices_{io_pair}")

                # expand the current subset of basis vectors and set the result in the appropriate place in the filter
                expanded = self._contract_basis_block(weights, points, io_pair)

                if _filter is None:
                    # build the tensor which will contain the filter
                    # this lazy strategy allows us to use expanded.dtype which is dynamically chosen by PyTorch's AMP
                    _filter = torch.zeros(
                        S,
                        self._output_size,
                        self._input_size,
                        device=weights.device,
                        dtype=expanded.dtype,
                    )

                if self._contiguous[io_pair]:
                    _filter[
                        :,
                        out_indices[0] : out_indices[1],
                        in_indices[0] : in_indices[1],
                    ] = expanded.reshape(S, out_indices[2], in_indices[2])
                else:
                    out_indices, in_indices = torch.meshgrid([out_indices, in_indices])
                    in_indices = in_indices.reshape(-1)
                    out_indices = out_indices.reshape(-1)
                    _filter[
                        :,
                        out_indices,
                        in_indices,
                    ] = expanded.reshape(S, -1)

            if _filter is None:
                # just in case
                _filter = torch.zeros(
                    S,
                    self._output_size,
                    self._input_size,
                    device=weights.device,
                    dtype=weights.dtype,
                )

        # return the new filter
        return _filter

    def _expand_filter_then_compute(
        self, weights: torch.Tensor, input: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        filter = self(weights, points)

        return torch.einsum("poi,pi->po", filter, input)

    def _compute_then_expand_filter(
        self, weights: torch.Tensor, input: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        S = input.shape[0]
        assert S > 0, S

        if self._uniform:
            # if there is only one block (i.e. one type of input field and one type of output field),
            #  we can return the computed block immediately, instead of copying it inside a preallocated empty tensor
            io_pair = self._representations_pairs[0]
            in_repr = io_pair[0]
            _input = input.view(S, self._in_count[in_repr], self._in_sizes[in_repr])
            return self._compute_out_block(weights, _input, points, io_pair).reshape(
                S, self._output_size
            )

        else:
            # build the tensor which will contain the output
            _out = torch.zeros(
                S, self._output_size, device=input.device, dtype=input.dtype
            )

            # iterate through all input-output field representations pairs
            for io_pair in self._representations_pairs:
                # retrieve the indices
                in_indices = getattr(self, f"in_indices_{io_pair}")
                out_indices = getattr(self, f"out_indices_{io_pair}")

                if self._contiguous[io_pair]:
                    _input = input[:, in_indices[0] : in_indices[1]]
                else:
                    _input = input[:, in_indices]

                in_repr = io_pair[0]
                _input = _input.view(
                    S, self._in_count[in_repr], self._in_sizes[in_repr]
                )
                # expand the current subset of basis vectors and set the result in the appropriate place in the filter
                block_out = self._compute_out_block(weights, _input, points, io_pair)

                if self._contiguous[io_pair]:
                    _out[:, out_indices[0] : out_indices[1]] += block_out.reshape(
                        S, out_indices[2]
                    )
                else:
                    _out[:, out_indices] += block_out.reshape(S, -1)

            return _out

    def compute_messages(
        self,
        weights: torch.Tensor,
        input: torch.Tensor,
        points: torch.Tensor,
        conv_first: bool = True,
    ) -> torch.Tensor:
        """
        Expands the basis with the learnable weights to generate the filter and use it to compute the messages along the
        edges.

        Each point in `points` corresponds to an edge in a graph.
        Each point is associated with a row of `input`.
        This row is a feature associated to the source node of the edge which needs to be propagated to the target
        node of the edge.

        Args:
            weights (torch.Tensor): the learnable weights used to linearly combine the basis filters
            input (torch.Tensor): the input features associated with each point
            points (torch.Tensor): the points where the filter should be sampled
            conv_first (bool, optional): perform convolution with the basis filters and, then, combine the responses
                                         with the learnable weights. This generally has computational benefits.
                                         (Default ``True``).

        Returns:
            the messages computed

        """
        assert weights.shape[0] == self.dimension()
        assert len(weights.shape) == 1

        assert len(input.shape) == 2
        assert input.shape[1] == self._input_size
        assert input.shape[0] == points.shape[0]

        if conv_first:
            return self._compute_then_expand_filter(weights, input, points)
        else:
            return self._expand_filter_then_compute(weights, input, points)

    def __hash__(self):
        _hash = 0
        for io in self._representations_pairs:
            n_pairs = self._in_count[io[0]] * self._out_count[io[1]]
            _hash += hash(getattr(self, f"block_sampler_{io}")) * n_pairs

        return _hash

    def __eq__(self, other):
        if not isinstance(other, BlocksBasisSampler):
            return False

        if self._dim != other._dim:
            return False

        if self._representations_pairs != other._representations_pairs:
            return False

        for io in self._representations_pairs:
            if self._contiguous[io] != other._contiguous[io]:
                return False

            if self._weights_ranges[io] != other._weights_ranges[io]:
                return False

            if self._contiguous[io]:
                if getattr(self, f"in_indices_{io}") != getattr(
                    other, f"in_indices_{io}"
                ):
                    return False
                if getattr(self, f"out_indices_{io}") != getattr(
                    other, f"out_indices_{io}"
                ):
                    return False
            else:
                if torch.any(
                    getattr(self, f"in_indices_{io}")
                    != getattr(other, f"in_indices_{io}")
                ):
                    return False
                if torch.any(
                    getattr(self, f"out_indices_{io}")
                    != getattr(other, f"out_indices_{io}")
                ):
                    return False

            if getattr(self, f"block_sampler_{io}") != getattr(
                other, f"block_sampler_{io}"
            ):
                return False

        return True
