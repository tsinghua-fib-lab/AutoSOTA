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

from chemtrain.deploy import graphs

import numpy as onp

import pytest

class TestPruning:

    def test_prune_keeps_ghost_sources_of_local_receivers(self):
        neighbor_list = graphs.SimpleSparseNeighborList(
            senders=jnp.asarray([2, 3]),
            receivers=jnp.asarray([0, 3]),
            max_edges=2,
        )
        local = jnp.asarray([True, False, False, False])

        pruned, n_valid = graphs.prune_neighbor_list(
            neighbor_list,
            local,
            max_edges=neighbor_list.max_edges,
            nbr_order=1,
            half_list=False,
        )

        assert n_valid == 1
        assert onp.any(
            (onp.asarray(pruned.senders) == 2)
            & (onp.asarray(pruned.receivers) == 0)
        )

    def test_prune_keeps_ghost_receivers_of_local_senders(self):
        neighbor_list = graphs.SimpleSparseNeighborList(
            senders=jnp.asarray([0, 3]),
            receivers=jnp.asarray([2, 3]),
            max_edges=2,
        )
        local = jnp.asarray([True, False, False, False])

        pruned, n_valid = graphs.prune_neighbor_list(
            neighbor_list,
            local,
            max_edges=neighbor_list.max_edges,
            nbr_order=1,
            half_list=False,
        )

        assert n_valid == 1
        assert onp.any(
            (onp.asarray(pruned.senders) == 0)
            & (onp.asarray(pruned.receivers) == 2)
        )

    @pytest.mark.parametrize(
        "list, local, order, pruned", [
            (
                graphs.SimpleSparseNeighborList(
                    jnp.asarray([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 2]),
                    jnp.asarray([1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 2, 0]),
                    max_edges=12
                ),
                jnp.asarray([1, 0, 0, 0, 0, 0], dtype=bool), 1,
                graphs.SimpleSparseNeighborList(
                    jnp.asarray([0, 1, 1, 2, 0, 2, 6, 6, 6, 6, 6, 6]),
                    jnp.asarray([1, 0, 2, 1, 2, 0, 6, 6, 6, 6, 6, 6]),
                    max_edges=6
                ),
            ),
            (
                graphs.SimpleSparseNeighborList(
                    jnp.asarray([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 2]),
                    jnp.asarray([1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 2, 0]),
                    max_edges=12
                ),
                jnp.asarray([0, 0, 0, 1, 0, 0], dtype=bool), 1,
                graphs.SimpleSparseNeighborList(
                    jnp.asarray([2, 3, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6]),
                    jnp.asarray([3, 2, 4, 3, 6, 6, 6, 6, 6, 6, 6, 6]),
                    max_edges=4
                ),
            ),
            (
                graphs.SimpleSparseNeighborList(
                    jnp.asarray([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 2]),
                    jnp.asarray([1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 2, 0]),
                    max_edges=12
                ),
                jnp.asarray([0, 0, 0, 0, 4, 0], dtype=bool), 2,
                graphs.SimpleSparseNeighborList(
                    jnp.asarray([2, 3, 3, 4, 4, 5, 6, 6, 6, 6, 6, 6]),
                    jnp.asarray([3, 2, 4, 3, 5, 4, 6, 6, 6, 6, 6, 6]),
                    max_edges=6
                ),
            )
        ]
    )
    def test_prune_unreachable(self, list, local, order, pruned):

        pruned_list, max_edges = graphs.prune_neighbor_list(
            list, local, max_edges=list.max_edges, nbr_order=order, half_list=False)

        print(f"Senders:   {pruned_list.senders}")
        print(f"Receivers: {pruned_list.receivers}")

        assert onp.all(pruned_list.senders == pruned.senders)
        assert onp.all(pruned_list.receivers == pruned.receivers)
        assert max_edges == pruned.max_edges
