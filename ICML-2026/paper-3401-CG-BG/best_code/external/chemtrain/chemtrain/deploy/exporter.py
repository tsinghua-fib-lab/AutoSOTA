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

"""Exporting potential models to MLIR."""

import abc
import functools
import inspect

import jax
from jax import numpy as jnp, export, lax

from typing import Dict, NamedTuple, Any, List, Tuple, Callable, NoReturn

import jax_md_mod
from jax_md import util as md_util, space

from . import comm, graphs, util
from ._protobuf import model_pb2 as model_proto


OPENEQUIVARIANCE_CUSTOM_CALLS = (
    "conv_forward",
    "conv_backward",
    "conv_double_backward",
)

COMMUNICATION_CUSTOM_CALLS = comm.CUSTOM_CALL_TARGETS


class _ExportComputation:
    """Create fresh communication state for every JAX trace.

    The callable may be retraced, so each invocation replaces ``widths``
    instead of appending metadata to state left by an earlier trace.
    """

    def __init__(
        self, owner, neighbor_orders, enabled=None, expected=None,
    ):
        self.owner = owner
        self.neighbor_orders = neighbor_orders
        self.enabled = enabled
        self.expected = expected
        self.widths = None

    def __call__(self, *args):
        communication = None
        if self.enabled is not None:
            communication = comm.ExportCommunication(
                enabled=self.enabled,
                expected_widths=self.expected,
            )
        result = self.owner._energy_fn(
            *args,
            communication=communication,
            neighbor_orders=self.neighbor_orders,
        )
        if communication is not None:
            communication.validate()
            self.widths = tuple(communication.widths)
        return result


class Exporter(metaclass=abc.ABCMeta):
    """Exports a potential model to an MLIR module.

    To export a potential model, subclass this class, select an appropriate
    graph type and define the energy function:

    Usage:

        >>> import jax.numpy as jnp
        >>> from jax_md_mod import custom_energy
        >>> from jax_md import partition, space
        >>> from chemtrain.deploy import exporter, graphs
        >>> class LennardJonesExport(exporter.Exporter):
        ...
        ...     graph_type = graphs.SimpleSparseNeighborList
        ...     r_cutoff = 5.0
        ...     unit_style = "real"
        ...     nbr_order = [1, 1]
        ...
        ...     def energy_fn(self, pos, species, graph):
        ...
        ...         neighbors = partition.NeighborList(
        ...             jnp.stack((graph.senders, graph.receivers)),
        ...             pos, None, None, graph.senders.size, partition.Sparse,
        ...             None, None, None
        ...         )
        ...
        ...         assert neighbors.idx.shape[0] == 2, "Wrong shape"
        ...
        ...         displacement_fn, _ = space.free()
        ...         apply_fn = custom_energy.customn_lennard_jones_neighbor_list(
        ...             displacement_fn, None, None,
        ...             sigma=jnp.asarray([3.165]), epsilon=jnp.asarray([1.0]),
        ...             r_onset=4.0, r_cutoff=5.0,
        ...             initialize_neighbor_list=False,
        ...             per_particle=True # Important for export
        ...         )
        ...
        ...         return apply_fn(pos, neighbors, species=species)
        ...
        >>> model = LennardJonesExport()
        >>> try:
        ...     model.save("out.ptb")
        ... except AssertionError as e:
        ...     # We need to call the export method first
        ...     print(f"Error: {e}")
        Error: Model has not been exported yet. Please call `export()` first.
        >>>
        >>> model.export()


    Attributes:
        graph_type: Specifies the required neighborhood representation and
            how to generate it from the input data.
            See ref:`chemtrain.deploy.graphs`.
        nbr_order: List of two integers specifying the number of neighbors
            required for the newton and non-newton setting to correctly
            compute forces.
        r_cutoff: Cutoff radius for the potential.
        unit_style: Specifies the units in which the potential requires
            positions and returns energies. The force units depend solely
            on the length and energy units.
        has_aux: If True, the energy function returns additional quantities
            besides the potential energy as dictionary.

    """

    # Use the default graph containing the full neighbor indices
    graph_type: graphs.NeighborList = graphs.SimpleSparseNeighborList

    # Order to which neighbors are required for a correct force computation
    # in the newton and the non-newton setting
    nbr_order: List[int] = [1, 1]

    r_cutoff: float
    unit_style: str = "real"
    has_aux: bool = False

    _symbols: List[str] = None
    _constraints: List[str] = None
    _init_fns: List[Callable] = None
    _proto: model_proto.Model = None

    @abc.abstractmethod
    def energy_fn(self, position, species, graph, comm=None):
        """Computes the energy for positions and a graph representation.

        Args:
            position: (N, dim) Array of particle positions, including ghost
                atoms that are not within the local domain.
            species: (N) Array of atoms species.
            graph: Graph representation of the neighborhood around atoms.
            comm: Optional feature-communication interface. Models that use
                communication call ``comm.gather(features)`` at their message
                passing boundaries. Existing three-argument implementations
                remain valid for ordinary exports.

        Returns:
            Must return an energy contribution associated to each particle.

        """
        pass

    @staticmethod
    @util.define_symbols("n_atoms")
    def _define_position_shapes(n_atoms, **kwargs):
        shape_defs = (
            jax.ShapeDtypeStruct((n_atoms, 3), jnp.float32),
            jax.ShapeDtypeStruct((n_atoms,), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32), # n_local
            jax.ShapeDtypeStruct((), jnp.int32), # n_ghost
            jax.ShapeDtypeStruct((), jnp.bool_), # newton flag
        )

        return shape_defs

    def _add_shapes(self, init_fn):
        init_fn(self._symbols, self._constraints, self._init_fns)

    def _create_shapes(self):
        all_symbols = ",".join(self._symbols)
        symbols = {
            key: symb for key, symb in zip(
                self._symbols,
                export.symbolic_shape(all_symbols, constraints=self._constraints),
            )
        }
        shapes = []
        for init_fn in self._init_fns:
            shapes.extend(init_fn(**symbols))

        # Reset
        self._symbols, self._constraints, self._init_fns = [], [], []
        return shapes


    def _energy_fn(
        self, position, species, n_local, n_ghost, newton, *graph_args,
        communication=None, neighbor_orders=None,
    ):
        # Expects particles to be sorted by local, ghost, and padding atoms

        valid_mask = jnp.arange(position.shape[0]) < (n_local + n_ghost)
        local_mask = jnp.arange(position.shape[0]) < n_local

        graph, build_statistics = self.graph_type.create_from_args(
            self.r_cutoff, neighbor_orders, position, species,
            local_mask, valid_mask, newton, *graph_args)
        graph = lax.stop_gradient(graph)

        @functools.partial(jax.grad, has_aux=True)
        def force_and_aux(pos):
            if communication is None:
                out = self.energy_fn(pos, species, graph)
            else:
                out = self.energy_fn(pos, species, graph, comm=communication)
            if self.has_aux:
                per_atom_energies, aux = out
            else:
                per_atom_energies = out
                aux = {}

            assert per_atom_energies.shape == local_mask.shape, (
                f"Per particle energies have shape {per_atom_energies.shape}, "
                f"but should have shape {local_mask.shape}."
            )

            # Attention: Force is negative gradient of potential.
            # Depending on the newton flag, we either compute:
            # - the gradient of the _total potential_ w.r.t. the _local atoms_
            # - the gradient of the _local potential_ w.r.t. _all atoms_
            # The latter case equals newton=true and requires additional
            # communication to sum up the forces on the ghost atoms.
            total_energy = md_util.high_precision_sum(
                jnp.where(valid_mask, per_atom_energies, jnp.float32(0.0)))
            local_energy = md_util.high_precision_sum(
                jnp.where(local_mask, per_atom_energies, jnp.float32(0.0))
            )

            force_energy = jnp.where(newton, local_energy, total_energy)
            force_energy = jnp.negative(force_energy)

            # Differentiate w.r.t. the total potential in the box, but exclude
            # ghost atom contributions to the total potential

            return force_energy, (per_atom_energies, aux)

        force, (energy, aux) = force_and_aux(position)
        predictions = dict(U=energy, F=force, **aux)

        for key, value in predictions.items():
            assert value.shape[0] == local_mask.size, (
                f"Wrong shape for prediction {key}. All model outputs "
                f"must be per-atom quantities."
            )

        return predictions, build_statistics

    def export(self, *, communication=False, custom_calls=()) -> None:
        """Export default and, optionally, communicating model variants.

        A communicating ``energy_fn`` accepts the exporter-supplied
        ``comm=None`` argument and calls ``comm.gather`` at every required
        message-passing boundary. Communication sites and their packed widths
        must be structurally fixed across JAX traces; data-dependent gather
        sequences are unsupported.

        The communicating variant uses one-hop neighbor orders ``[1, 1]`` and
        is supported by chemtrain-deploy only with LAMMPS Newton pair forces
        enabled. Newton-off simulations must select the default variant, whose
        neighbor orders remain ``self.nbr_order``.

        ``custom_calls`` is an explicit allowlist for model-specific FFI
        targets such as OpenEquivariance. Communication targets are added only
        to the communication variant. Existing exporters need no changes when
        calling ``export()`` without arguments.

        """
        # Quantities describe one export transaction. A failed or repeated
        # export must not compare new variants with metadata from an earlier
        # invocation on the same exporter instance.
        self._export_quantities = None
        custom_calls = tuple(custom_calls)
        signature = inspect.signature(self.energy_fn)
        supports_comm = "comm" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if communication and not supports_comm:
            raise TypeError(
                "communication=True requires energy_fn(..., comm=None)"
            )

        proto = model_proto.Model()
        proto.unit_style = self.unit_style

        default = self._export_variant(
            name="default",
            neighbor_orders=self.nbr_order,
            communication_widths=None,
            custom_calls=custom_calls,
        )
        proto.variants.add().CopyFrom(default)

        # Legacy readers see exactly the ordinary model. Variant selection is
        # additive and therefore cannot silently enable communication.
        proto.mlir_module = default.mlir_module
        proto.neighbor_list.CopyFrom(default.neighbor_list)
        proto.quantities.extend(self._export_quantities)
        proto.uses_communication = False
        proto.communication_buffer_width = 0

        if communication:
            communication_widths = self._trace_communication_widths([1, 1])
            communicating = self._export_variant(
                name="comm",
                neighbor_orders=[1, 1],
                communication_widths=communication_widths,
                custom_calls=custom_calls + COMMUNICATION_CUSTOM_CALLS,
            )
            communicating.uses_communication = True
            communicating.communication_forward_sites = len(
                communication_widths
            )
            communicating.communication_widths.extend(communication_widths)
            communicating.communication_buffer_width = max(
                communication_widths, default=0
            )
            proto.variants.add().CopyFrom(communicating)

        self._proto = proto

    def _variant_inputs(self, neighbor_orders):
        """Build graph metadata and symbolic inputs for one variant."""
        self._symbols: List[str] = []
        self._constraints: List[str] = []
        self._init_fns: List[Callable] = []
        assert len(neighbor_orders) == 2, (
            "The nbr_order must contain the order of required neighbors for "
            "the newton and non-newton setting."
        )
        metadata = model_proto.Model()
        metadata.neighbor_list.cutoff = self.r_cutoff
        metadata.neighbor_list.nbr_order.extend(neighbor_orders)
        self.graph_type.set_properties(metadata)

        # Using the ghost mask in the last layer we can compute correct forces
        # by accounting for their contribution to the gradient but
        # mask them out when we compute the total potential to not count
        # them double.
        self._add_shapes(self._define_position_shapes)
        self._add_shapes(self.graph_type.create_symbolic_input_format)
        return metadata, self._create_shapes()

    def _trace_communication_widths(self, neighbor_orders):
        """Record widths in an isolated trace with communication disabled."""
        _, shapes = self._variant_inputs(neighbor_orders)
        computation = _ExportComputation(
            self, neighbor_orders, enabled=False,
        )
        jax.eval_shape(jax.jit(computation), *shapes)
        if not computation.widths:
            raise ValueError(
                "communication=True was requested, but energy_fn did not "
                "call comm.gather"
            )
        return computation.widths

    def _export_variant(
        self, *, name, neighbor_orders, communication_widths,
        custom_calls,
    ):
        """Trace one model variant and return its self-contained metadata."""
        metadata, shapes = self._variant_inputs(neighbor_orders)

        computation = _ExportComputation(
            self,
            neighbor_orders,
            enabled=True if communication_widths is not None else None,
            expected=communication_widths,
        )
        export_fn = jax.jit(computation)

        exp: export.Exported = export.export(
            export_fn,
            platforms=["cuda"],
            disabled_checks=tuple(
                export.DisabledSafetyCheck.custom_call(target)
                for target in custom_calls
            ),
        )(*shapes)

        if (communication_widths is not None and
                computation.widths != communication_widths):
            raise ValueError(
                "Communication metadata changed during final export: "
                f"{computation.widths} != {communication_widths}"
            )

        # Reconstruct the output to save the returned statistics and...
        predictions, statistics = exp.out_tree.unflatten(exp.out_avals)

        metadata.neighbor_list.statistics_keys.extend(statistics.keys())
        quantities = list(predictions.keys())
        if self._export_quantities is not None:
            if quantities != self._export_quantities:
                raise ValueError(
                    "All exported variants must return the same quantities"
                )
        else:
            self._export_quantities = quantities

        variant = model_proto.Model.ModelVariant()
        variant.name = name
        # JAX structurally checks every StableHLO custom call against the
        # target-specific DisabledSafetyCheck entries supplied above. Avoid
        # duplicating that check with version-sensitive MLIR text parsing.
        variant.mlir_module = exp.mlir_module()
        variant.neighbor_list.CopyFrom(metadata.neighbor_list)
        return variant

    def __str__(self):
        assert self._proto is not None, (
            "Model has not been exported yet. Please call `export()` first."
        )

        return str(self._proto)

    def save(self, file: str) -> None:
        """Save the exported protocol buffer to ``file``."""
        assert self._proto is not None, (
            "Model has not been exported yet. Please call `export()` first."
        )

        with open(file, "wb") as f:
            f.write(self._proto.SerializeToString())
