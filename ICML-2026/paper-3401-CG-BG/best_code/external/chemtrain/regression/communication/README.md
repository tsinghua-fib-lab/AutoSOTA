# Communication regression

This regression tests the distributed MACE execution path provided by
chemtrain-deploy. It compares the ordinary model variant with a variant that
exchanges intermediate atom features through LAMMPS.

Both variants contain the same MACE weights. `comm off` selects the baseline
export, which evaluates the full local-plus-ghost environment in each domain.
`comm on` selects the export that exchanges intermediate learned atom features
between MPI domains during message passing. Agreement between them isolates
the communication implementation; it does not by itself prove that shared
model conversion or deployment code is physically correct.

The test covers:

- per-atom energy, force, position, and total-energy-trace agreement;
- per-variant NVE total-energy conservation over two short trajectory segments;
- atom- and neighbor-buffer growth;
- recompilation after a simulation box is compressed;
- supported and unsupported Newton settings;
- molecular predictions for a system divided between two MPI ranks; and
- agreement with predictions from the original MACE model.

The directory contains all input structures and LAMMPS input files required by
the regression.

## Requirements for native execution

- chemtrain with the MACE-JAX dependencies installed;
- a CUDA build of chemtrain-deploy and its LAMMPS plugin;
- OpenEquivariance;
- `lmp`, `mpirun`, and `mace_eval_configs` available on `PATH`; and
- two visible CUDA devices.

The first export may populate the normal MACE model cache.
Activate the chemtrain-deploy environment described in the installation
documentation before running the regression. It must provide the Python
packages, LAMMPS executable, plugin paths, and PJRT runtime.

## Running the regression

The standard test exports the models and runs every regression case:

```bash
# Activate the chemtrain-deploy environment first.
./run_standard_test.sh
```

Runtime settings are listed at the top of `run_standard_test.sh`. They can be
overridden without editing the script. For example:

```bash
CUDA_VISIBLE_DEVICES=6,7 \
OMP_NUM_THREADS=4 \
MPI_LAUNCHER="mpirun -np 2" \
OUTPUT_DIRECTORY="$PWD/results-a100" \
./run_standard_test.sh
```

The output directory contains the exported model, Torch reference model,
LAMMPS logs, trajectory dumps, captured screen output, and `summary.json`. Its
default location is `results` beside the script.

Predictions from the two deployed variants are aligned by atom ID before their
per-atom energies and forces are compared. Thermodynamic total energies are
still used to check trajectory energy drift and agreement with the original
MACE model, whose command-line output does not expose atomic contributions.

The Python driver can also be called directly:

```bash
python run_regression.py
```

An existing export and reference model can be reused:

```bash
python run_regression.py \
  --skip_export \
  --model results/model.ptb \
  --reference_model results/reference.model
```

Use `python run_regression.py --help` for executable, launcher, input, and
output options.

## What constitutes a pass

`run_standard_test.sh` exits nonzero as soon as any required command or check
fails. Equality with a limit passes; a larger value fails.

| Check | Pass criterion |
|---|---:|
| Same-coordinate per-atom energy | maximum absolute error ≤ `1e-4 eV` |
| Newton on/off fallback per-atom energy | maximum absolute error ≤ `1e-3 eV` |
| Force | maximum absolute Cartesian-component error ≤ `5e-3 eV/Å` |
| Trajectory position | maximum absolute component error ≤ `1e-4 Å` |
| Total-energy trace | maximum baseline-subtracted disagreement ≤ `5e-5 eV/atom` |
| NVE energy conservation | maximum per-variant drift ≤ `1e-3 eV/atom` |
| Original-MACE total energy | error ≤ `1e-3 eV/atom` |
| Original-MACE force | maximum absolute component error ≤ `5e-2 eV/Å` |

- `trajectory.lmp` is launched twice from the same deterministic initial state,
  once with each deployed model variant. Frame 0 energies are a strict
  same-coordinate per-atom comparison. Across all 40 integrated steps, maximum
  force and position differences and disagreement between the two total-energy
  traces must remain below their limits. Each variant must also conserve NVE
  total energy within the documented limit before and after the deliberate box
  compression. The deliberately compressed low-padding case must record both
  atom-buffer and edge-buffer recompilation on both variants. Each dump must
  contain exactly steps 0–40.
- `newton.lmp` uses `run 0`: LAMMPS evaluates neighbors, energy, and forces but
  advances no timestep. The static crystal contains a deterministic random
  plane defect near the MPI boundary and must produce nonzero forces. Default
  predictions with Newton pair forces on and off must agree in total energy and
  forces. Their per-atom energy comparison uses the documented fallback limit
  because LAMMPS changes the neighbor-list style between Newton modes. Internal
  communication with Newton on must agree with the default Newton-on result
  using the stricter per-atom energy limit. The reference maximum force must be
  at least `1e-3 eV/Å`. Internal communication with Newton off must fail and print
  `Communication requires Newton pair forces`.
- `predict.lmp` reruns fixed molecular frames without integrating them. Both
  ranks must own atoms, the default and communication variants must agree in
  atom-ID-aligned energies and forces, and both variants must remain within the
  separate tolerances for the original MACE model.

For each deployed variant, initial compilation count must be at least two in
total across the two ranks. In the low-padding case, atom and edge
recompilation counts must each be at least one.

Total potential-energy differences and later-frame per-atom energy differences
are retained as diagnostics. They are not primary pass/fail quantities: total
energies can hide cancellation, while independently integrated later frames no
longer have exactly identical coordinates.

## Recompilation statistics

The LAMMPS summary reports initial compilations and runtime recompilations.
Runtime causes are reported separately for atom buffers and neighbor buffers.
A single compilation may have both causes, so the cause counts are not expected
to sum to the total compilation count.

## Test limits

This regression uses two MPI ranks and periodic orthogonal cells. It does not
cover empty ranks, restarts, triclinic cells, nonperiodic boundaries, virial or
stress predictions, multi-axis processor grids, atom migration through a rank
boundary, or long-duration energy conservation. Default-versus-communication
agreement can also miss a defect shared by both deployment paths; the original
MACE comparison supplies an independent reference, but currently covers only
the bundled molecular frames and uses the separately documented MACE-reference
tolerances.
