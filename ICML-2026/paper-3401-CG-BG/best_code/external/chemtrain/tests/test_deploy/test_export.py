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

import jax
from jax import numpy as jnp, Array

from chemtrain.deploy import comm, graphs, exporter

from jax_md_mod import custom_partition
from jax_md import space, partition, util as md_util

import numpy as onp

import pytest


def model_neighborlist_pp(displacement: space.DisplacementFn,
                          r_cutoff: float,
                          positions_test: jnp.ndarray = None,
                          neighbor_test: partition.NeighborList = None,
                          max_edge_multiplier: float = 1.25,
                          max_edges=None,
                          avg_num_neighbors: float = None,
                          ):
    """Export test model."""
    r_cutoff = jnp.array(r_cutoff, dtype=md_util.f32)

    # Checking only necessary if neighbor list is dense
    _avg_num_neighbors = None
    if positions_test is not None and neighbor_test is not None:
        _avg_num_neighbors, _ = custom_partition.test_graph_statistics(
            displacement, positions_test, neighbor_test,
            r_cutoff, max_edge_multiplier=max_edge_multiplier
        )

    if avg_num_neighbors is None:
        avg_num_neighbors = _avg_num_neighbors

    assert avg_num_neighbors is not None, (
        "Average number of neighbors not set and no test graph was provided."
    )

    def model(position: md_util.Array,
              neighbor: partition.NeighborList,
              species: md_util.Array = None,
              mask: md_util.Array = None,
              **dynamic_kwargs):
        if species is None:
            species = jnp.zeros(position.shape[0], dtype=jnp.int32)
        if mask is None:
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        vectors, senders, receivers = custom_partition.readout_vectors(
            displacement, r_cutoff, position, neighbor, species,
            mask, max_edges=max_edges, **dynamic_kwargs
        )

        vectors /= r_cutoff

        pot = (jnp.linalg.norm(vectors, axis=-1) - 1.0) ** 2
        return jax.ops.segment_sum(pot, senders, num_segments=position.shape[0])

    return jax.jit(model)





class TestExport:

    @pytest.fixture(scope="function")
    def setup_export(self):
        class ExportedModel(exporter.Exporter):

            graph_type = graphs.SimpleSparseNeighborList
            r_cutoff = 5.0
            unit_style = "real"
            nbr_order = [1, 1]

            def __init__(self, max_edge_multiplier=None, max_edges=None):
                self.max_edge_multiplier = max_edge_multiplier
                self.max_edges = max_edges

            def energy_fn(self, pos, species,
                          graph: graphs.SimpleSparseNeighborList):
                neighbors = graph.to_neighborlist()
                displacement_fn, _ = space.free()

                model = model_neighborlist_pp(
                    displacement_fn, self.r_cutoff,
                    max_edges=self.max_edges, avg_num_neighbors=20.0
                )

                pot = model(pos, neighbors, species=species)
                return pot

        yield ExportedModel

    def test_no_max_edges(self, tmp_path, setup_export):
        model = setup_export(max_edges=None)

        model.export()
        model.save(tmp_path / "exported_no_max_edges.ptb")

    def test_export_custom_call_allowlist(self, setup_export):
        model = setup_export(max_edges=None)
        model.export(custom_calls=exporter.OPENEQUIVARIANCE_CUSTOM_CALLS)
        assert [variant.name for variant in model._proto.variants] == [
            "default"
        ]

    def test_export_resets_quantity_metadata(self, setup_export):
        model = setup_export(max_edges=None)
        model._export_quantities = ["stale_quantity"]

        model.export()

        assert set(model._export_quantities) == {"U", "F"}
        assert "stale_quantity" not in model._export_quantities
        assert list(model._proto.quantities) == model._export_quantities

    def test_symbolic_max_edges(self, tmp_path, setup_export):
        class ExportedModelSymbolic(setup_export):

            def __init__(self):
                super().__init__(max_edge_multiplier=0.5, max_edges=None)

        model = ExportedModelSymbolic()

        model.export()
        model.save(tmp_path / "exported_no_max_edges.ptb")

    def test_static_max_edges(self, tmp_path, setup_export):
        # This test ensures that the maximum number of edges during export
        # must be symbolic. If it is static, the number of edges would be
        # also fixed during execution.

        class ExportedModelStatic(setup_export):
            def __init__(self):
                super().__init__(max_edge_multiplier=None, max_edges=10)

        model = ExportedModelStatic()

        with pytest.raises(TypeError, match="max_edges must be symbolic"):
            model.export()

        with pytest.raises(AssertionError, match="has not been exported yet"):
            model.save(tmp_path / "exported_no_max_edges.ptb")

    def test_communication_export_records_variants_and_widths(self, setup_export):
        class CommunicatingModel(setup_export):
            def energy_fn(self, pos, species, graph, comm=None):
                energy = super().energy_fn(pos, species, graph)
                if comm is not None:
                    energy = comm.gather(comm.gather(energy))
                return energy

        model = CommunicatingModel(max_edges=None)
        model.export(communication=True)

        assert [variant.name for variant in model._proto.variants] == [
            "default", "comm"
        ]
        default, communicating = model._proto.variants
        assert list(default.neighbor_list.nbr_order) == [1, 1]
        assert list(communicating.neighbor_list.nbr_order) == [1, 1]
        assert communicating.uses_communication
        assert communicating.communication_forward_sites == 2
        assert list(communicating.communication_widths) == [1, 1]
        assert communicating.communication_buffer_width == 1
        assert not model._proto.uses_communication

        # A second export retraces the model. Site metadata must be replaced,
        # not appended to state left by the first trace.
        model.export(communication=True)
        communicating = model._proto.variants[1]
        assert communicating.communication_forward_sites == 2
        assert list(communicating.communication_widths) == [1, 1]

    def test_three_argument_model_rejects_communication(self, setup_export):
        model = setup_export(max_edges=None)
        with pytest.raises(TypeError, match="comm=None"):
            model.export(communication=True)
