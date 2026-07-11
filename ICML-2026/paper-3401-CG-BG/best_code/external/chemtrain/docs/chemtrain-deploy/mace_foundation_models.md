# MACE Foundation Models

This example shows how to export the [MACE-MH-1](https://github.com/ACEsuit/mace-foundations)
foundation model and run it in LAMMPS via chemtrain-deploy.
The simulation uses a ~10,000-atom water box on 2 GPUs.

## Prerequisites

- A working chemtrain-deploy installation; see [](installation).
- `chemtrain`
- MACE (PyTorch), as well as MACE-JAX with OpenEquivariance support
- A PDB file for the initial configuration. The example below uses
  `water_10k.pdb` from `docs/_data/`.

## Overview

The workflow has three steps:

1. **Export** — convert the PyTorch MACE-MH-1 weights to a chemtrain-deploy
   model bundle (`.ptb`).
2. **Prepare** — convert the PDB file to a LAMMPS data file.
3. **Simulate** — run the LAMMPS script against the exported model.

## Step 1: Export the Model

The export script loads MACE-MH-1 with the `omol` head, wraps it in an
{class}`chemtrain.deploy.exporter.Exporter` subclass, and writes both a
`default` (single-GPU) and a `comm` (multi-GPU) variant to the bundle.

`export_mace_mh1.py`:
```python
import argparse
import functools
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference_output", type=Path, required=True)
    parser.add_argument("--family", default="mp")
    parser.add_argument("--version", default="mh-1")
    args = parser.parse_args()

    # Must precede JAX / MACE / OpenEquivariance imports.
    os.environ["OEQ_NOTORCH"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["JAX_PLATFORMS"] = "cuda"

    import torch
    from chemtrain.compose import mace_jax as mace_jax_compose
    from chemtrain.deploy import exporter, graphs as export_graphs
    from jax_md import space
    from mace_jax.modules.wrapper_ops import (
        EquivarianceConfig,
        OpenEquivarianceConfig,
    )

    class MaceWaterExporter(exporter.Exporter):
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
            # The connector delivers 0-based LAMMPS atom types;
            # MACE expects 1-based atomic numbers.
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
        head="omol",
    )

    model = MaceWaterExporter(
        functools.partial(apply_fn, variables),
        model_config,
        displacement,
    )
    model.export(
        communication=True,
        custom_calls=exporter.OPENEQUIVARIANCE_CUSTOM_CALLS,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    args.reference_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_model, args.reference_output)


if __name__ == "__main__":
    main()
```

A few things to note:

- `head="omol"` selects the Open Molecules head of MACE-MH-1.
  The model ships with multiple heads trained on different datasets;
- `scale_pot=1.0` and `scale_pos=1.0` keep the model's native units
  (eV and Å). `unit_style = "metal"` tells LAMMPS to expect the same.
- `max_edge_multiplier=None` disables the static edge-count cap, which is
  required for the deployment path because the graph size is managed by the
  connector's dynamic padding instead.
- `per_particle=True` returns per-atom energies, which is required by
  {class}`chemtrain.deploy.exporter.Exporter`.
- `communication=True` instructs the exporter to trace and bundle a second
  variant that performs halo exchanges between message-passing layers,
  enabling multi-GPU decomposition.

Run the export (the model weights are downloaded automatically on first use):

```bash
python export_mace_mh1.py \
    --output results/mace_mh1_water10k.ptb \
    --reference_output results/mace_mh1_reference.model
```

## Step 2: Prepare the LAMMPS Data File

LAMMPS needs a data file with atom types encoded as atomic numbers
(H → 1, O → 8).
The helper script `prepare_lammps_data.py` reads a PDB and writes a
compatible data file:

```bash
python prepare_lammps_data.py \
    --pdb water_10k.pdb \
    --output results/water_10k.lmpdat
```

The script reads the box length from a `REMARK L = <value> A` line in the PDB
and maps element symbols to atom types via the dictionary
`{"H": 1, "O": 8}`.

## Step 3: Run the Simulation

`simulation.lmp`:
```text
variable model      index results/mace_mh1_water10k.ptb
variable data_file  index results/water_10k.lmpdat
variable traj_dump  index results/water10k.lammpstrj
variable atom_pad   index 1.2
variable edge_pad   index 1.5
variable proc_x     index 2
variable proc_y     index 1
variable proc_z     index 1

units metal
atom_style atomic
boundary p p p
newton on

processors ${proc_x} ${proc_y} ${proc_z}
read_data ${data_file}

# Select the distributed comm variant; requires newton on and multiple ranks.
pair_style chemtrain_deploy cuda 0.85 comm on
pair_coeff * * ${model} ${atom_pad} ${edge_pad}

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

thermo 10
thermo_style custom step atoms temp pe ke etotal press

dump trajectory all custom 50 ${traj_dump} id type x y z fx fy fz
dump_modify trajectory sort id first yes

# Validate neighbor construction and force evaluation.
run 0

min_style quickmin
minimize 1.0e-4 1.0e-6 20 200

velocity all create 300.0 20260626 mom yes rot no dist gaussian
fix thermostat all nvt temp 300.0 300.0 0.1
timestep 0.0005

run 1000
```

Key settings:

- `pair_style chemtrain_deploy cuda 0.85 comm on` selects the CUDA backend
  with 85 % memory allocation and enables halo-exchange communication.
- The two multipliers in `pair_coeff` (`atom_pad`, `edge_pad`) control how
  much extra space is reserved in the atom and neighbor-list buffers.
- `processors 2 1 1` decomposes the box into two domains along x, one per GPU.

## Running on 2 GPUs

Expose exactly the two GPUs you want to use via `CUDA_VISIBLE_DEVICES` and
launch LAMMPS with 2 MPI ranks:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export LAMMPS_PLUGIN_PATH=/path/to/chemtrain-deploy/build
export JCN_PJRT_PATH=/path/to/chemtrain-deploy/lib

mpirun -np 2 lmp -in simulation.lmp
```

The plugin assigns one GPU per MPI rank based on rank order within the visible
device set, so `CUDA_VISIBLE_DEVICES=0,1` maps rank 0 → GPU 0 and rank 1 → GPU 1.
`OMP_NUM_THREADS=1` avoids CPU thread oversubscription when MPI and OpenMP
are both active.
