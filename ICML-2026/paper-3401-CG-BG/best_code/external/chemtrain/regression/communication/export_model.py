#!/usr/bin/env python3
# Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich
# SPDX-License-Identifier: Apache-2.0
"""Export the MACE bundle exercised by the communication regression."""

import argparse
import functools
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MACE-MP medium-0b3 with default and comm variants."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the chemtrain-deploy model bundle.",
    )
    parser.add_argument(
        "--reference_output",
        type=Path,
        required=True,
        help="Output path for the Torch model used as the numerical reference.",
    )
    parser.add_argument(
        "--family", default="mp", help="MACE foundation-model family."
    )
    parser.add_argument(
        "--version", default="medium-0b3", help="MACE model version."
    )
    args = parser.parse_args()

    # These settings must precede the JAX, MACE, and OpenEquivariance imports.
    os.environ["OEQ_NOTORCH"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["JAX_PLATFORMS"] = "cuda"

    import torch
    from chemtrain.compose import mace_jax as mace_jax_compose
    from chemtrain.deploy import exporter, graphs as export_graphs
    from jax_md import space
    from mace_jax.modules.wrapper_ops import (
        EquivarianceConfig,
        OpenEquivarianceConfig,
    )

    class MaceCommunicationExporter(exporter.Exporter):
        """Expose a converted MACE model through chemtrain's exporter API."""

        graph_type = export_graphs.SimpleSparseNeighborList
        unit_style = "metal"

        def __init__(self, model_apply, model_config, displacement):
            self.model = model_apply
            self.r_cutoff = model_config["r_max"]
            self.nbr_order = [
                model_config["num_interactions"],
                2 * model_config["num_interactions"],
            ]
            self.displacement = displacement
            super().__init__()

        def energy_fn(self, position, species, graph, comm=None):
            neighbor = graph.to_neighborlist()
            # The connector converts LAMMPS atom types to zero-based values;
            # MACE uses one-based atomic numbers.
            return self.model(
                position,
                neighbor,
                species=species + 1,
                comm=comm,
            )

    equivariance = EquivarianceConfig(
        layout="mul_ir",
        openeq_config=OpenEquivarianceConfig(
            enabled=True,
            optimize_all=False,
            optimize_channelwise=True,
            optimize_fctp=False,
            conv_fusion=True,
            group="O3_e3nn",
        ),
    )
    torch_model, model_config = mace_jax_compose.load_foundational_model(
        family=args.family,
        version=args.version,
    )
    displacement, _ = space.free()
    variables, apply_fn = mace_jax_compose.mace_jax_neighborlist_from_torch(
        model_config,
        torch_model,
        displacement,
        max_edge_multiplier=None,
        per_particle=True,
        species_mapping=mace_jax_compose.AtomicNumberMapping(max_number=100),
        scale_pot=1.0,
        scale_pos=1.0,
        equivariance_config=equivariance,
    )

    model = MaceCommunicationExporter(
        functools.partial(apply_fn, variables),
        model_config,
        displacement,
    )
    model.export(
        communication=True,
        custom_calls=exporter.OPENEQUIVARIANCE_CUSTOM_CALLS,
    )

    variants = {variant.name: variant for variant in model._proto.variants}
    if set(variants) != {"default", "comm"}:
        raise RuntimeError(
            "Export must produce exactly the default and comm variants; got "
            f"{sorted(variants)}"
        )
    communication = variants["comm"]
    if not communication.uses_communication:
        raise RuntimeError("The comm variant does not declare communication")
    if communication.communication_forward_sites <= 0:
        raise RuntimeError("The comm variant contains no forward gather sites")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    args.reference_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_model, args.reference_output)
    print(
        "Exported communication regression model: "
        f"gathers={communication.communication_forward_sites}, "
        f"width={communication.communication_buffer_width}, "
        f"neighbor_order={list(communication.neighbor_list.nbr_order)}, "
        f"path={args.output}, reference={args.reference_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
