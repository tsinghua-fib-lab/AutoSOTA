# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools

from jax import numpy as jnp, Array

from jax_md_mod import custom_partition
from jax_md import partition, dataclasses

import numpy as onp

import pytest

import networkx as nx


# A simple methane-like graph
star_graph_dense = jnp.asarray([
    [1, 2, 3, 4, 5, 5, 5, 5],
    [0, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 5, 5, 5, 5, 5, 5],
])

star_graph_sparse = jnp.asarray(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
        [5, 5],
        [5, 5],
        [5, 5],
        [5, 5]
    ]
)

fc_graph_dense = jnp.asarray([
    [1, 2, 3, 4, 4],
    [0, 2, 3, 4, 4],
    [0, 1, 3, 4, 4],
    [0, 1, 2, 4, 4]
])

fc_graph_sparse = jnp.asarray([
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 2],
        [1, 3],
        [2, 0],
        [2, 1],
        [2, 3],
        [3, 0],
        [3, 1],
        [3, 2],
        [4, 4],
        [4, 4]
    ]
])


@dataclasses.dataclass
class NeighborIdx:
    idx: Array
    format: partition.NeighborListFormat
    reference_position: Array = None


class TestEdgeMasking:

    @pytest.mark.parametrize("graph", (star_graph_dense, fc_graph_dense))
    @pytest.mark.parametrize("exclude", (
        [[0, 1], [1, 2]],
        [[0, 1], [0, 1]],
        [[0, 1], [1, 0]]
    ))
    @pytest.mark.parametrize("mask", (
        [True, True],
        [False, True]
    ))
    def test_exclude_dense(self, graph, exclude, mask):
        exclude = jnp.asarray(exclude)
        mask = jnp.asarray(mask)


        neighbor = NeighborIdx(
            idx=graph, format=partition.NeighborListFormat.Dense)

        # Substract from graph
        neighbor = custom_partition.exclude_from_neighbor_list(
            neighbor, exclude, mask
        )

        print(f"New neighbor list")
        print(neighbor)

        # Check
        for (i, j), mask in zip(exclude, mask):
            if not mask: continue

            assert i not in neighbor.idx[j]
            assert j not in neighbor.idx[i]

    @pytest.mark.parametrize("graph", (star_graph_sparse, fc_graph_sparse))
    @pytest.mark.parametrize("exclude", (
        [[0, 1], [1, 2]],
        [[0, 1], [0, 1]],
        [[0, 1], [1, 0]]
    ))
    @pytest.mark.parametrize("mask", (
        [True, True],
        [False, True]
    ))
    def test_exclude_sparse(self, graph, exclude, mask):
        exclude = jnp.asarray(exclude)
        mask = jnp.asarray(mask)

        neighbor = NeighborIdx(
            idx=graph, format=partition.NeighborListFormat.Sparse)

        # Substract from graph
        neighbor = custom_partition.exclude_from_neighbor_list(
            neighbor, exclude, mask
        )

        print(f"New neighbor list")
        print(neighbor)

        # Check
        for (i, j), mask in zip(exclude, mask):
            if not mask: continue

            assert not jnp.any(
                jnp.logical_and(
                    i == neighbor.idx[0, :],
                    j == neighbor.idx[1, :]
                )
            )
            assert not jnp.any(
                jnp.logical_and(
                    j == neighbor.idx[0, :],
                    i == neighbor.idx[1, :]
                )
            )


class TestTriplets:

    @pytest.mark.parametrize("graph", (star_graph_dense, fc_graph_dense))
    def test_get_dense(self, graph):
        neighbor = NeighborIdx(
            idx=graph, format=partition.NeighborListFormat.Dense)

        ij, kj, mask = custom_partition.get_triplet_indices(neighbor)
        mask = onp.asarray(mask)

        # Remove all invalid edges
        ij = ij[mask, :]
        kj = kj[mask, :]

        # Add the graph to nx and check whether all found triplets exist
        graph = custom_partition.to_networkx(neighbor)

        for i, j, k in zip(ij[:, 0], ij[:, 1], kj[:, 0]):
            assert nx.is_simple_path(graph, [int(i), int(j), int(k)])

        print(onp.append(ij, kj[:, (0,)], axis=1))
        print(mask)

        # Search all triplets in the graph
        for a, b in itertools.combinations(graph.nodes, 2):
            for path in nx.all_simple_paths(graph, a, b, cutoff=3):
                if len(path) != 3: continue

                # Check whether triplets were also found by jax algorithm
                found = False
                for i, j, k in zip(ij[:, 0], ij[:, 1], kj[:, 0]):
                    if (int(i), int(j), int(k)) == tuple(path):
                        print(f"Found {(int(i), int(j), int(k))} == {path}")
                        found = True
                        break

                if not found:
                    print(f"Missing {path}")

                assert found
