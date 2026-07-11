#!/usr/bin/env python3
# Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich
# SPDX-License-Identifier: Apache-2.0
"""Run the end-to-end chemtrain communication regression.

This is an integration test for the distributed chemtrain/LAMMPS MACE path. It
exercises model export, two-rank LAMMPS execution, MPI communication, JAX
recompilation behavior, and independent MACE reference predictions.

The JSON summary is intentionally machine-readable. Human-facing context is kept
in this file and in the generated Markdown report.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import ase.io
import numpy as onp


# ---------------------------------------------------------------------------
# Regression thresholds
# ---------------------------------------------------------------------------
#
# Direct default/communication comparisons use tight tolerances because both
# paths should evaluate the same exported model through LAMMPS. The independent
# MACE reference check uses looser tolerances because it crosses a separate CLI
# and model-loading path.

MAX_ATOMIC_ENERGY_ERROR_EV = 1.0e-4
MAX_FORCE_ERROR_EV_PER_ANGSTROM = 5.0e-3
MAX_POSITION_ERROR_ANGSTROM = 1.0e-4
MAX_TOTAL_ENERGY_TRACE_ERROR_EV_PER_ATOM = 5.0e-5
MAX_NVE_ENERGY_DRIFT_EV_PER_ATOM = 1.0e-3
MAX_NEWTON_FALLBACK_ATOMIC_ENERGY_ERROR_EV = 1.0e-3
MIN_NEWTON_REFERENCE_FORCE_EV_PER_ANGSTROM = 1.0e-3
MACE_ENERGY_ERROR_EV_PER_ATOM = 1.0e-3
MACE_FORCE_ERROR_EV_PER_ANGSTROM = 5.0e-2
MODEL_CUTOFF_ANGSTROM = 5.0


# ---------------------------------------------------------------------------
# Reported metrics
# ---------------------------------------------------------------------------
#
# compare_lammps_runs reports:
#
# atomic_energy_error_ev:
#     Maximum strict per-atom energy difference. For NVE trajectories this is
#     checked only on the initial frame, where both variants evaluate identical
#     coordinates.
#
# trajectory_atomic_energy_error_ev:
#     Diagnostic maximum per-atom energy difference across the full trajectory.
#     This is not used for pass/fail because later NVE frames may no longer
#     evaluate identical coordinates.
#
# force_error_ev_per_angstrom:
#     Maximum absolute force-component difference. In a trajectory this is an
#     end-to-end trajectory-agreement quantity after frame 0, because the two
#     forces are evaluated at slightly different coordinates. Static/rerun
#     cases remain strict same-coordinate force comparisons.
#
# position_error_angstrom:
#     Maximum absolute coordinate difference.
#
# total_energy_trace_error_ev_per_atom:
#     Maximum disagreement between changes in the two NVE total-energy traces,
#     normalized per atom. This checks trajectory correspondence, not absolute
#     energy conservation; the deliberate box compression does work on the
#     system and separates two short NVE segments.
#
# total_energy_error_ev_per_atom:
#     Diagnostic total potential-energy difference per atom. Reported but not
#     used for pass/fail.
#
# *_nve_energy_drift_ev_per_atom:
#     Maximum total-energy drift inside each uninterrupted NVE segment for each
#     variant separately. The trajectory contains a deliberate box compression,
#     so conservation is checked before and after that operation with separate
#     baselines.
#
# The molecular MACE comparison reports maximum force-component error and
# maximum total-energy error per atom against an independent MACE CLI run.


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------

def run_command(
    name: str,
    command: list[str],
    *,
    working_directory: Path,
    output_directory: Path,
    environment: dict[str, str],
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one external regression command and preserve its screen output.

    LAMMPS, MACE, and MPI diagnostics are part of the regression evidence.
    Capturing stdout and stderr together keeps CI artifacts useful when a case
    fails before producing complete dumps or logs.
    """
    print(f"\n[{name}] {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=working_directory,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    screen_path = output_directory / f"{name}.screen"
    screen_path.write_text(completed.stdout)

    if expect_success and completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {completed.returncode}; "
            f"see {screen_path}"
        )
    if not expect_success and completed.returncode == 0:
        raise AssertionError(f"{name} unexpectedly succeeded")
    return completed


# ---------------------------------------------------------------------------
# LAMMPS output parsing
# ---------------------------------------------------------------------------

def read_lammps_dump(path: Path) -> list[dict[str, onp.ndarray]]:
    """Read a LAMMPS custom dump into atom-aligned frame dictionaries.

    Frames are sorted by atom ID so default and communication runs can be
    compared directly even when LAMMPS writes atoms in different rank-local
    order.
    """
    frames: list[dict[str, onp.ndarray]] = []
    lines = path.read_text().splitlines()
    index = 0

    while index < len(lines):
        if lines[index] != "ITEM: TIMESTEP":
            index += 1
            continue

        step = int(lines[index + 1])
        count = int(lines[index + 3])
        header_index = index + 8
        header = lines[header_index].split()
        if header[:2] != ["ITEM:", "ATOMS"]:
            raise RuntimeError(f"Malformed atom header in {path}")

        columns = header[2:]
        data = onp.asarray(
            [
                [float(value) for value in line.split()]
                for line in lines[
                    header_index + 1 : header_index + 1 + count
                ]
            ]
        )
        column_index = {name: number for number, name in enumerate(columns)}
        data = data[onp.argsort(data[:, column_index["id"]].astype(int))]
        frame = {name: data[:, number] for name, number in column_index.items()}
        frame["step"] = onp.asarray(step)
        frames.append(frame)

        index = header_index + 1 + count

    if not frames:
        raise RuntimeError(f"No frames found in {path}")
    return frames


def read_thermo_columns(text: str, quantities: tuple[str, ...]) -> dict[str, onp.ndarray]:
    """Collect named columns from all LAMMPS thermo tables in order.

    Some LAMMPS inputs print several thermo tables. MPI output can also
    interleave plugin diagnostics into a thermo table before the final rank has
    printed its last row, so unrelated lines are skipped until LAMMPS prints the
    table-closing ``Loop time`` line.
    """
    collected = {quantity: [] for quantity in quantities}
    lines = text.splitlines()

    for index, line in enumerate(lines):
        header = line.split()
        if (
            not header
            or header[0] != "Step"
            or any(quantity not in header for quantity in quantities)
        ):
            continue

        columns = {
            quantity: header.index(quantity)
            for quantity in quantities
        }
        for row in lines[index + 1 :]:
            row_header = row.split()
            if row_header and row_header[0] == "Step":
                break
            if row.startswith("Loop time"):
                break
            fields = row.split()
            if len(fields) != len(header):
                continue
            try:
                values = {
                    quantity: float(fields[column])
                    for quantity, column in columns.items()
                }
            except ValueError:
                continue
            for quantity, value in values.items():
                collected[quantity].append(value)

    for quantity, values in collected.items():
        if not values:
            raise RuntimeError(f"No LAMMPS thermo column {quantity!r} was found")
    return {
        quantity: onp.asarray(values)
        for quantity, values in collected.items()
    }


def read_thermo_quantity(text: str, quantity: str) -> onp.ndarray:
    """Collect one named quantity from all LAMMPS thermo tables in order."""
    return read_thermo_columns(text, (quantity,))[quantity]


def read_potential_energies(text: str) -> onp.ndarray:
    """Collect potential energies used by static and reference comparisons."""
    return read_thermo_quantity(text, "PotEng")


def maximum_segmented_nve_drift(text: str, atoms: int) -> float:
    """Return the largest total-energy drift inside uninterrupted NVE segments.

    ``trajectory.lmp`` prints thermo output for ``run 20``, ``run 0``, and a
    second ``run 20``. The box compression between the two real runs changes the
    energy by construction, so the baseline is restarted whenever thermo step
    numbers stop increasing.
    """
    thermo = read_thermo_columns(text, ("Step", "TotEng"))
    steps = thermo["Step"]
    total_energy = thermo["TotEng"]
    maximum_drift = 0.0
    segment_start = 0

    for index in range(1, len(steps) + 1):
        ends_segment = index == len(steps) or steps[index] <= steps[index - 1]
        if not ends_segment:
            continue

        segment = total_energy[segment_start:index]
        if len(segment) > 1:
            drift = onp.max(onp.abs(segment - segment[0])) / atoms
            maximum_drift = max(maximum_drift, float(drift))
        segment_start = index

    return maximum_drift


# ---------------------------------------------------------------------------
# Shared numerical comparisons
# ---------------------------------------------------------------------------

def compare_lammps_runs(
    case: str,
    reference_dump: Path,
    communication_dump: Path,
    reference_screen: str,
    communication_screen: str,
    *,
    atomic_energy_tolerance: float = MAX_ATOMIC_ENERGY_ERROR_EV,
) -> dict[str, float]:
    """Compare two LAMMPS outputs for the same regression case.

    The comparison checks local atom-wise quantities before total-energy traces,
    because local communication errors can cancel in summed observables. For NVE
    trajectories, strict per-atom energy checks are limited to the initial frame;
    later frames may no longer evaluate identical coordinates after integration.
    """
    reference = read_lammps_dump(reference_dump)
    communication = read_lammps_dump(communication_dump)
    if len(reference) != len(communication):
        raise AssertionError(
            f"{case}: frame counts differ: {len(reference)} and "
            f"{len(communication)}"
        )

    # Local atom-wise checks catch communication errors that total energies can
    # hide. Position columns differ between trajectory and static/rerun dumps.
    trajectory_case = "xu" in reference[0]
    position_errors = []
    force_errors = []
    atomic_energy_errors = []
    for frame_index, (expected, actual) in enumerate(
        zip(reference, communication, strict=True)
    ):
        for exact_column in ("id", "type", "step"):
            onp.testing.assert_array_equal(
                actual[exact_column],
                expected[exact_column],
                err_msg=f"{case}, frame {frame_index}: {exact_column} differs",
            )

        position_columns = (
            ("xu", "yu", "zu") if "xu" in expected else ("x", "y", "z")
        )
        position_errors.append(
            onp.column_stack(
                [actual[name] - expected[name] for name in position_columns]
            )
        )
        force_errors.append(
            onp.column_stack(
                [actual[name] - expected[name] for name in ("fx", "fy", "fz")]
            )
        )
        if "c_atom_energy" not in expected or "c_atom_energy" not in actual:
            raise AssertionError(
                f"{case}, frame {frame_index}: per-atom energy is missing"
            )
        atomic_energy_errors.append(
            actual["c_atom_energy"] - expected["c_atom_energy"]
        )

    # PotEng is reported as a useful diagnostic. For a trajectory, Etot is the
    # pass/fail trace: comparing only PotEng would mistake ordinary exchange
    # between kinetic and potential energy for integration disagreement.
    reference_energy = read_potential_energies(reference_screen)
    communication_energy = read_potential_energies(communication_screen)
    if reference_energy.shape != communication_energy.shape:
        raise AssertionError(
            f"{case}: thermo energy shapes differ: {reference_energy.shape} "
            f"and {communication_energy.shape}"
        )

    atoms = len(reference[0]["id"])
    total_energy_delta = (
        onp.abs(communication_energy - reference_energy) / atoms
    )
    total_energy_frame = int(onp.argmax(total_energy_delta))
    total_energy_error = float(total_energy_delta[total_energy_frame])

    all_atomic_energy_error_array = onp.abs(onp.asarray(atomic_energy_errors))
    atomic_energy_error_array = (
        all_atomic_energy_error_array[:1]
        if trajectory_case
        else all_atomic_energy_error_array
    )
    atomic_energy_index = onp.unravel_index(
        onp.argmax(atomic_energy_error_array), atomic_energy_error_array.shape
    )
    atomic_energy_error = float(atomic_energy_error_array[atomic_energy_index])

    force_error_array = onp.abs(onp.asarray(force_errors))
    force_index = onp.unravel_index(
        onp.argmax(force_error_array), force_error_array.shape
    )
    force_error = float(force_error_array[force_index])

    position_error_array = onp.abs(onp.asarray(position_errors))
    position_index = onp.unravel_index(
        onp.argmax(position_error_array), position_error_array.shape
    )
    position_error = float(position_error_array[position_index])

    if trajectory_case:
        reference_total_energy = read_thermo_quantity(reference_screen, "TotEng")
        communication_total_energy = read_thermo_quantity(
            communication_screen, "TotEng"
        )
        if reference_total_energy.shape != communication_total_energy.shape:
            raise AssertionError(
                f"{case}: total-energy trace shapes differ: "
                f"{reference_total_energy.shape} and "
                f"{communication_total_energy.shape}"
            )
        reference_trace = reference_total_energy - reference_total_energy[0]
        communication_trace = (
            communication_total_energy - communication_total_energy[0]
        )
        trace_delta = onp.abs(communication_trace - reference_trace) / atoms
        trace_frame = int(onp.argmax(trace_delta))
        trace_error = float(trace_delta[trace_frame])
        reference_nve_drift = maximum_segmented_nve_drift(
            reference_screen, atoms
        )
        communication_nve_drift = maximum_segmented_nve_drift(
            communication_screen, atoms
        )
    else:
        trace_frame = 0
        trace_error = 0.0
        reference_nve_drift = 0.0
        communication_nve_drift = 0.0

    # Failure messages include the largest observed error and where it occurred,
    # so CI logs identify the affected frame, atom, component, or thermo row.
    limits = {
        "atomic_energy_error_ev": (
            atomic_energy_error,
            atomic_energy_tolerance,
        ),
        "force_error_ev_per_angstrom": (
            force_error,
            MAX_FORCE_ERROR_EV_PER_ANGSTROM,
        ),
        "position_error_angstrom": (
            position_error,
            MAX_POSITION_ERROR_ANGSTROM,
        ),
        "total_energy_trace_error_ev_per_atom": (
            trace_error,
            MAX_TOTAL_ENERGY_TRACE_ERROR_EV_PER_ATOM,
        ),
    }
    if trajectory_case:
        limits["reference_nve_energy_drift_ev_per_atom"] = (
            reference_nve_drift,
            MAX_NVE_ENERGY_DRIFT_EV_PER_ATOM,
        )
        limits["communication_nve_energy_drift_ev_per_atom"] = (
            communication_nve_drift,
            MAX_NVE_ENERGY_DRIFT_EV_PER_ATOM,
        )
    locations = {
        "atomic_energy_error_ev": (
            f"frame {atomic_energy_index[0]}, atom ID "
            f"{int(reference[atomic_energy_index[0]]['id'][atomic_energy_index[1]])}"
        ),
        "force_error_ev_per_angstrom": (
            f"frame {force_index[0]}, atom {force_index[1] + 1}, "
            f"component {force_index[2]}"
        ),
        "position_error_angstrom": (
            f"frame {position_index[0]}, atom {position_index[1] + 1}, "
            f"component {position_index[2]}"
        ),
        "total_energy_trace_error_ev_per_atom": f"thermo row {trace_frame}",
    }
    if trajectory_case:
        locations["reference_nve_energy_drift_ev_per_atom"] = (
            "reference NVE segment"
        )
        locations["communication_nve_energy_drift_ev_per_atom"] = (
            "communication NVE segment"
        )

    for quantity, (measured, tolerance) in limits.items():
        if measured > tolerance:
            raise AssertionError(
                f"{case}: {quantity}={measured:.6g} exceeds "
                f"{tolerance:.6g} at {locations[quantity]}"
            )

    print(
        f"PASS {case}: max atomic dE={atomic_energy_error:.3e} eV, "
        f"total dE/atom={total_energy_error:.3e} eV, "
        f"max dF={force_error:.3e} eV/Angstrom",
        flush=True,
    )

    metrics = {name: measured for name, (measured, _) in limits.items()}
    metrics["total_energy_error_ev_per_atom"] = total_energy_error
    if trajectory_case:
        metrics["trajectory_atomic_energy_error_ev"] = float(
            onp.max(all_atomic_energy_error_array)
        )
    return metrics


# ---------------------------------------------------------------------------
# Trajectory padding case
# ---------------------------------------------------------------------------

def run_trajectory_case(
    case: str,
    atom_padding: float,
    edge_padding: float,
    *,
    args: argparse.Namespace,
    script_directory: Path,
    output_directory: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    """Run one trajectory recompilation scenario with both model variants.

    ``trajectory.lmp`` first runs a two-rank bcc Ti trajectory, then compresses
    the box and continues. With low padding, the compression should exercise
    both atom and edge recompilation paths while preserving numerical agreement
    between the default and communication-enabled models.
    """
    outputs: dict[str, tuple[Path, subprocess.CompletedProcess[str]]] = {}

    # The only intended difference between the two runs is whether the
    # distributed communication path is enabled.
    for variant, communication in (("default", "off"), ("comm", "on")):
        name = f"{case}_{variant}"
        dump = output_directory / f"{name}.lammpstrj"
        log = output_directory / f"{name}.log"
        command = [
            *shlex.split(args.launcher),
            args.lmp,
            "-var",
            "model",
            str(args.model.resolve()),
            "-var",
            "comm",
            communication,
            "-var",
            "atom_padding",
            str(atom_padding),
            "-var",
            "edge_padding",
            str(edge_padding),
            "-var",
            "trajectory_dump",
            str(dump),
            "-log",
            str(log),
            "-in",
            "trajectory.lmp",
        ]
        outputs[variant] = (
            dump,
            run_command(
                name,
                command,
                working_directory=script_directory,
                output_directory=output_directory,
                environment=environment,
            ),
        )

    # trajectory.lmp writes the initial state and forty integrated steps. Check
    # this explicitly so two equally truncated jobs cannot pass by comparison.
    for variant, (dump, _) in outputs.items():
        steps = [int(frame["step"]) for frame in read_lammps_dump(dump)]
        if steps != list(range(41)):
            raise AssertionError(
                f"{case}/{variant}: expected trajectory steps 0 through 40, "
                f"got {steps}"
            )

    metrics = compare_lammps_runs(
        case,
        outputs["default"][0],
        outputs["comm"][0],
        outputs["default"][1].stdout,
        outputs["comm"][1].stdout,
    )

    # Compilation statistics verify that the padding scenario covered the
    # intended execution path, not merely that the trajectory happened to pass.
    statistics: dict[str, dict[str, int]] = {}
    for variant, (_, completed) in outputs.items():
        text = completed.stdout
        statistics[variant] = {}
        for label, key in (
            ("Initial compilations", "initial"),
            ("Atom recompilations", "atom"),
            ("Edge recompilations", "edge"),
        ):
            stable_key = {
                "Initial compilations": "initial_total",
                "Atom recompilations": "atom_total",
                "Edge recompilations": "edge_total",
            }[label]
            matches = re.findall(rf"JCN_STATS .*?{stable_key}=([0-9]+)", text)
            if not matches:
                raise RuntimeError(f"{case}/{variant}: missing {label} summary")
            statistics[variant][key] = max(int(value) for value in matches)

        if statistics[variant]["initial"] < 2:
            raise AssertionError(
                f"{case}/{variant}: expected initial compilation on both ranks, got "
                f"{statistics[variant]['initial']} total"
            )
        if statistics[variant]["atom"] < 1 or statistics[variant]["edge"] < 1:
            raise AssertionError(
                f"{case}/{variant}: compression did not demonstrate both "
                f"recompilation causes: {statistics[variant]}"
            )

    return {"metrics": metrics, "statistics": statistics}


# ---------------------------------------------------------------------------
# Newton pair-force behavior
# ---------------------------------------------------------------------------

def run_newton_cases(
    *,
    args: argparse.Namespace,
    script_directory: Path,
    output_directory: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    """Check Newton pair-force fallback and rejection behavior.

    ``newton.lmp`` is a static two-rank bcc Ti prediction. The default model
    must support both Newton-on and Newton-off execution. The communication
    model must match Newton-on execution and reject Newton-off execution because
    distributed communication requires Newton pair forces.
    """
    successful: dict[str, tuple[Path, subprocess.CompletedProcess[str]]] = {}

    # Successful runs cover the default Newton fallback and the supported
    # communication configuration.
    for name, communication, newton in (
        ("newton_on_default", "off", "on"),
        ("newton_off_default", "off", "off"),
        ("newton_on_comm", "on", "on"),
    ):
        dump = output_directory / f"{name}.lammpstrj"
        command = [
            *shlex.split(args.launcher),
            args.lmp,
            "-var",
            "model",
            str(args.model.resolve()),
            "-var",
            "comm",
            communication,
            "-var",
            "newton_setting",
            newton,
            "-var",
            "prediction_dump",
            str(dump),
            "-log",
            str(output_directory / f"{name}.log"),
            "-in",
            "newton.lmp",
        ]
        successful[name] = (
            dump,
            run_command(
                name,
                command,
                working_directory=script_directory,
                output_directory=output_directory,
                environment=environment,
            ),
        )

    # run 0 must still produce exactly one evaluated step-0 frame. Also require
    # a meaningful force signal: otherwise the Newton comparison would be weak
    # even if both paths incorrectly returned zero for the symmetric crystal.
    for name, (dump, _) in successful.items():
        frames = read_lammps_dump(dump)
        if len(frames) != 1 or int(frames[0]["step"]) != 0:
            raise AssertionError(
                f"{name}: run 0 must produce one step-0 prediction frame"
            )
    reference_frame = read_lammps_dump(
        successful["newton_on_default"][0]
    )[0]
    reference_force = onp.column_stack(
        [reference_frame[name] for name in ("fx", "fy", "fz")]
    )
    maximum_reference_force = float(onp.max(onp.abs(reference_force)))
    if maximum_reference_force < MIN_NEWTON_REFERENCE_FORCE_EV_PER_ANGSTROM:
        raise AssertionError(
            "newton_on_default: distorted crystal did not produce a useful "
            f"force signal; max |F|={maximum_reference_force:.6g} eV/Angstrom"
        )

    fallback_metrics = compare_lammps_runs(
        "newton_off_default_fallback",
        successful["newton_on_default"][0],
        successful["newton_off_default"][0],
        successful["newton_on_default"][1].stdout,
        successful["newton_off_default"][1].stdout,
        atomic_energy_tolerance=MAX_NEWTON_FALLBACK_ATOMIC_ENERGY_ERROR_EV,
    )
    communication_metrics = compare_lammps_runs(
        "newton_on_internal_comm",
        successful["newton_on_default"][0],
        successful["newton_on_comm"][0],
        successful["newton_on_default"][1].stdout,
        successful["newton_on_comm"][1].stdout,
    )

    # The communication path should fail early and explicitly when configured
    # with unsupported Newton pair-force settings.
    rejected_name = "newton_off_comm_rejected"
    rejected = run_command(
        rejected_name,
        [
            *shlex.split(args.launcher),
            args.lmp,
            "-var",
            "model",
            str(args.model.resolve()),
            "-var",
            "comm",
            "on",
            "-var",
            "newton_setting",
            "off",
            "-var",
            "prediction_dump",
            str(output_directory / f"{rejected_name}.lammpstrj"),
            "-log",
            str(output_directory / f"{rejected_name}.log"),
            "-in",
            "newton.lmp",
        ],
        working_directory=script_directory,
        output_directory=output_directory,
        environment=environment,
        expect_success=False,
    )
    expected_error = "Communication requires Newton pair forces"
    if expected_error not in rejected.stdout:
        raise AssertionError(
            f"{rejected_name}: missing documented error {expected_error!r}"
        )

    print("PASS Newton behavior", flush=True)
    return {
        "fallback_metrics": fallback_metrics,
        "communication_metrics": communication_metrics,
        "rejected_error": expected_error,
    }


# ---------------------------------------------------------------------------
# Molecular split-rank prediction case
# ---------------------------------------------------------------------------

def prepare_molecular_frames(
    samples_path: Path,
    output_directory: Path,
) -> tuple[Path, Path, Path]:
    """Create molecular rerun inputs that require cross-rank communication.

    Each molecule is centered on the x=0 rank boundary. Frames are rejected if
    they do not contain atoms on both sides with at least one cross-boundary
    neighbor inside the model cutoff, because such frames would not exercise the
    communication path.
    """
    structures = ase.io.read(samples_path, index=":")
    if not structures:
        raise ValueError(f"No molecular frames found in {samples_path}")

    centered_path = output_directory / "molecule_centered.xyz"
    data_path = output_directory / "molecule.lmpdat"
    dump_path = output_directory / "molecule_input.lammpstrj"

    # Atom order and species must stay fixed so MACE, LAMMPS, and direct dump
    # comparisons refer to the same atoms in every frame.
    first_numbers = structures[0].numbers.tolist()
    with (
        centered_path.open("w") as centered,
        data_path.open("w") as data,
        dump_path.open("w") as dump,
    ):
        first_positions = None
        for frame_index, structure in enumerate(structures):
            if structure.numbers.tolist() != first_numbers:
                raise ValueError(
                    f"Frame {frame_index} changes atom count, order, or species"
                )

            positions = structure.positions.copy()
            positions[:, 0] -= 0.5 * (
                positions[:, 0].min() + positions[:, 0].max()
            )
            positions[:, 1:] -= positions[:, 1:].mean(axis=0)
            structure.positions = positions

            left = positions[:, 0] < 0.0
            right = positions[:, 0] >= 0.0
            distances = onp.linalg.norm(
                positions[left, None, :] - positions[None, right, :],
                axis=-1,
            )
            if not left.any() or not right.any() or not onp.any(
                distances < MODEL_CUTOFF_ANGSTROM
            ):
                raise AssertionError(
                    f"Frame {frame_index} has no cutoff neighbor across x=0"
                )

            ase.io.write(centered, structure, format="extxyz")
            if first_positions is None:
                first_positions = positions.copy()

            dump.write("ITEM: TIMESTEP\n")
            dump.write(f"{frame_index}\n")
            dump.write("ITEM: NUMBER OF ATOMS\n")
            dump.write(f"{len(structure)}\n")
            dump.write("ITEM: BOX BOUNDS pp pp pp\n")
            dump.write("-25 25\n-25 25\n-25 25\n")
            dump.write("ITEM: ATOMS id x y z\n")
            for atom_id, position in enumerate(positions, start=1):
                dump.write(
                    f"{atom_id} {position[0]:.16g} {position[1]:.16g} "
                    f"{position[2]:.16g}\n"
                )

        # The first frame initializes LAMMPS; the rerun dump supplies all
        # molecular geometries that are evaluated during the prediction case.
        data.write("Centered molecular regression frame\n\n")
        data.write(f"{len(first_numbers)} atoms\n")
        data.write("90 atom types\n\n")
        data.write("-25 25 xlo xhi\n-25 25 ylo yhi\n-25 25 zlo zhi\n\n")
        data.write("Atoms\n\n")
        for atom_id, (atomic_number, position) in enumerate(
            zip(first_numbers, first_positions, strict=True),
            start=1,
        ):
            data.write(
                f"{atom_id} {atomic_number} {position[0]:.16g} "
                f"{position[1]:.16g} {position[2]:.16g}\n"
            )

    return centered_path, data_path, dump_path


def run_molecular_prediction_case(
    *,
    args: argparse.Namespace,
    script_directory: Path,
    output_directory: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    """Validate molecular predictions across communication and reference paths.

    ``predict.lmp`` reruns fixed molecular conformations centered on the x=0
    rank boundary. The direct communication/default comparison is the strict
    communication regression. The MACE CLI comparison is a broader reference
    check with looser tolerances because it uses an independent execution path.
    """
    centered, data_file, rerun_dump = prepare_molecular_frames(
        args.samples.resolve(), output_directory
    )

    # The independent MACE reference catches export or model-loading issues that
    # may not be visible in a direct default/communication comparison.
    reference_path = output_directory / "mace_reference.xyz"
    run_command(
        "mace_reference",
        [
            args.mace_eval,
            "--configs",
            str(centered),
            "--model",
            str(args.reference_model),
            "--output",
            str(reference_path),
            "--device",
            args.mace_device,
            "--default_dtype",
            "float64",
            "--batch_size",
            "16",
            "--head",
            "default",
        ],
        working_directory=script_directory,
        output_directory=output_directory,
        environment=environment,
    )

    # Both LAMMPS variants evaluate the same molecular rerun frames. Ownership
    # must be split across both ranks, otherwise this would not test distributed
    # molecular communication.
    runs: dict[
        str, tuple[list[dict[str, onp.ndarray]], onp.ndarray]
    ] = {}
    for variant, communication in (("default", "off"), ("comm", "on")):
        name = f"molecule_{variant}"
        prediction = output_directory / f"{name}.lammpstrj"
        completed = run_command(
            name,
            [
                *shlex.split(args.launcher),
                args.lmp,
                "-var",
                "model",
                str(args.model.resolve()),
                "-var",
                "comm",
                communication,
                "-var",
                "data_file",
                str(data_file),
                "-var",
                "rerun_dump",
                str(rerun_dump),
                "-var",
                "prediction_dump",
                str(prediction),
                "-log",
                str(output_directory / f"{name}.log"),
                "-in",
                "predict.lmp",
            ],
            working_directory=script_directory,
            output_directory=output_directory,
            environment=environment,
        )

        frames = read_lammps_dump(prediction)
        for frame_index, frame in enumerate(frames):
            owners = set(frame["proc"].astype(int))
            if owners != {0, 1}:
                raise AssertionError(
                    f"{name}, frame {frame_index}: expected ownership on "
                    f"ranks 0 and 1, got {sorted(owners)}"
                )
        runs[variant] = (frames, read_potential_energies(completed.stdout))

    references = ase.io.read(reference_path, index=":")
    if len(references) != len(runs["default"][0]):
        raise AssertionError("MACE and LAMMPS molecular frame counts differ")

    # The reference comparison is intentionally tolerant enough to account for
    # independent tooling while still catching wrong models, species, or units.
    result: dict[str, object] = {}
    for variant, (frames, energies) in runs.items():
        force_errors = []
        energy_errors = []
        for frame_index, (reference, frame, energy) in enumerate(
            zip(references, frames, energies, strict=True)
        ):
            force_key = next(
                (
                    key
                    for key in ("MACE_forces", "forces")
                    if key in reference.arrays
                ),
                None,
            )
            energy_key = next(
                (
                    key
                    for key in ("MACE_energy", "energy")
                    if key in reference.info
                ),
                None,
            )
            if force_key is None or energy_key is None:
                raise KeyError(
                    f"MACE frame {frame_index} lacks force keys "
                    "MACE_forces/forces or energy keys MACE_energy/energy"
                )

            reference_forces = onp.asarray(reference.arrays[force_key])
            reference_energy = float(reference.info[energy_key])
            onp.testing.assert_array_equal(
                frame["type"].astype(int),
                reference.numbers,
                err_msg=(
                    f"molecule_{variant}, frame {frame_index}: atomic "
                    "numbers differ from the independent MACE input"
                ),
            )

            predicted_forces = onp.column_stack(
                [frame[name] for name in ("fx", "fy", "fz")]
            )
            if predicted_forces.shape != reference_forces.shape:
                raise AssertionError(
                    f"molecule_{variant}, frame {frame_index}: force shapes "
                    f"differ"
                )

            force_errors.append(predicted_forces - reference_forces)
            energy_errors.append(
                abs(float(energy) - reference_energy) / len(reference)
            )

        force_error_array = onp.asarray(force_errors)
        maximum_force_index = onp.unravel_index(
            onp.argmax(onp.abs(force_error_array)), force_error_array.shape
        )
        maximum_force_error = float(abs(force_error_array[maximum_force_index]))
        maximum_energy_frame = int(onp.argmax(energy_errors))
        maximum_energy_error = float(energy_errors[maximum_energy_frame])

        if maximum_force_error > MACE_FORCE_ERROR_EV_PER_ANGSTROM:
            raise AssertionError(
                f"molecule_{variant}: MACE force error "
                f"{maximum_force_error:.6g} eV/Angstrom exceeds "
                f"{MACE_FORCE_ERROR_EV_PER_ANGSTROM:.6g} at frame "
                f"{maximum_force_index[0]}, atom {maximum_force_index[1] + 1}, "
                f"component {maximum_force_index[2]}"
            )
        if maximum_energy_error > MACE_ENERGY_ERROR_EV_PER_ATOM:
            raise AssertionError(
                f"molecule_{variant}: MACE energy error "
                f"{maximum_energy_error:.6g} eV/atom exceeds "
                f"{MACE_ENERGY_ERROR_EV_PER_ATOM:.6g} at frame "
                f"{maximum_energy_frame}"
            )

        result[f"{variant}_versus_mace"] = {
            "maximum_force_error_ev_per_angstrom": maximum_force_error,
            "maximum_energy_error_ev_per_atom": maximum_energy_error,
        }

    # The direct comparison is the most sensitive communication regression:
    # both predictions should come from the same exported model and LAMMPS path.
    default_frames, default_energies = runs["default"]
    communication_frames, communication_energies = runs["comm"]
    if len(default_frames) != len(communication_frames):
        raise AssertionError("Molecular comm/default frame counts differ")
    if default_energies.shape != communication_energies.shape:
        raise AssertionError("Molecular comm/default energy shapes differ")

    direct_force_errors = []
    direct_atomic_energy_errors = []
    for frame_index, (expected, actual) in enumerate(
        zip(default_frames, communication_frames, strict=True)
    ):
        for column in ("id", "type", "step"):
            onp.testing.assert_array_equal(
                actual[column],
                expected[column],
                err_msg=(
                    f"molecule comm/default, frame {frame_index}: "
                    f"{column} differs"
                ),
            )

        direct_force_errors.append(
            onp.column_stack(
                [actual[name] - expected[name] for name in ("fx", "fy", "fz")]
            )
        )
        direct_atomic_energy_errors.append(
            actual["c_atom_energy"] - expected["c_atom_energy"]
        )

    direct_force_error_array = onp.abs(onp.asarray(direct_force_errors))
    direct_force_index = onp.unravel_index(
        onp.argmax(direct_force_error_array), direct_force_error_array.shape
    )
    direct_force_error = float(direct_force_error_array[direct_force_index])

    direct_atomic_energy_error_array = onp.abs(
        onp.asarray(direct_atomic_energy_errors)
    )
    direct_atomic_energy_index = onp.unravel_index(
        onp.argmax(direct_atomic_energy_error_array),
        direct_atomic_energy_error_array.shape,
    )
    direct_atomic_energy_error = float(
        direct_atomic_energy_error_array[direct_atomic_energy_index]
    )

    if direct_force_error > MAX_FORCE_ERROR_EV_PER_ANGSTROM:
        raise AssertionError(
            "molecule communication/default force error "
            f"{direct_force_error:.6g} eV/Angstrom exceeds "
            f"{MAX_FORCE_ERROR_EV_PER_ANGSTROM:.6g} at frame "
            f"{direct_force_index[0]}, atom {direct_force_index[1] + 1}, "
            f"component {direct_force_index[2]}"
        )
    if direct_atomic_energy_error > MAX_ATOMIC_ENERGY_ERROR_EV:
        raise AssertionError(
            "molecule communication/default atomic energy error "
            f"{direct_atomic_energy_error:.6g} eV exceeds "
            f"{MAX_ATOMIC_ENERGY_ERROR_EV:.6g} at frame "
            f"{direct_atomic_energy_index[0]}, atom ID "
            f"{int(default_frames[direct_atomic_energy_index[0]]['id'][direct_atomic_energy_index[1]])}"
        )

    result["comm_versus_default"] = {
        "maximum_force_error_ev_per_angstrom": direct_force_error,
        "maximum_atomic_energy_error_ev": direct_atomic_energy_error,
    }

    print(
        f"PASS molecular predictions: frames={len(references)}, "
        f"comm/default atomic dE={direct_atomic_energy_error:.3e} eV, "
        f"dF={direct_force_error:.3e} eV/Angstrom",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Regression artifacts
# ---------------------------------------------------------------------------

def write_summary(summary: dict[str, object], output_directory: Path) -> Path:
    """Write machine-readable metrics for CI and post-processing."""
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path


def _markdown_table(data: object) -> str:
    """Format nested dictionaries as a compact Markdown table."""
    rows: list[tuple[str, object]] = []

    def collect(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                collect(f"{prefix}.{key}" if prefix else key, nested)
        else:
            rows.append((prefix, value))

    collect("", data)
    if not rows:
        return "_No metrics recorded._"

    table = ["| Metric | Value |", "|---|---:|"]
    for key, value in rows:
        if isinstance(value, float):
            rendered = f"{value:.8g}"
        else:
            rendered = str(value)
        table.append(f"| `{key}` | {rendered} |")
    return "\n".join(table)


def render_report(summary: dict[str, object]) -> str:
    """Render a human-readable Markdown report for CI artifacts."""
    lines = [
        "# Communication Regression Report",
        "",
        "## Trajectory recompilation",
        "",
        "Input: `trajectory.lmp`",
        "",
        (
            "A two-rank bcc Ti trajectory is run with deliberately low atom and "
            "edge padding. The first segment establishes the compiled shapes; "
            "the box compression then increases halo occupancy and edge count. "
            "The case passes only if atom and edge recompilation both occur and "
            "the default and communication-enabled trajectories remain within "
            "the strict comparison tolerances."
        ),
        "",
        "### Numerical metrics",
        "",
        _markdown_table(summary["low_padding"]["metrics"]),
        "",
        "### Compilation statistics",
        "",
        _markdown_table(summary["low_padding"]["statistics"]),
        "",
        "## Newton pair-force behavior",
        "",
        "Input: `newton.lmp`",
        "",
        (
            "A static two-rank bcc Ti prediction checks the Newton contract. "
            "Atoms near the rank boundary receive deterministic random "
            "displacements to produce nonzero cross-rank forces. The default "
            "model must agree with itself for Newton pair forces on and off. "
            "The communication-enabled model must match Newton-on execution "
            "and reject Newton off with the documented error."
        ),
        "",
        "### Default Newton on/off metrics",
        "",
        _markdown_table(summary["newton"]["fallback_metrics"]),
        "",
        "### Internal communication Newton-on metrics",
        "",
        _markdown_table(summary["newton"]["communication_metrics"]),
        "",
        f"Rejected configuration error: `{summary['newton']['rejected_error']}`",
        "",
        "## Molecular split-rank prediction",
        "",
        "Input: `predict.lmp`",
        "",
        (
            "Fixed molecular conformations are centered on the x=0 rank boundary "
            "and rerun with ownership split across two MPI ranks. The strict "
            "check compares communication-enabled LAMMPS directly against the "
            "default LAMMPS path. A looser reference check compares both LAMMPS "
            "variants against an independent MACE CLI prediction."
        ),
        "",
        "### Metrics",
        "",
        _markdown_table(summary["molecule"]),
        "",
    ]
    return "\n".join(lines)


def write_report(summary: dict[str, object], output_directory: Path) -> Path:
    """Write a human-readable regression report for CI artifacts."""
    report_path = output_directory / "report.md"
    report_path.write_text(render_report(summary))
    return report_path


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full communication regression and write JSON and Markdown output."""
    parser = argparse.ArgumentParser(
        description=(
            "Export and test chemtrain's default and distributed MACE variants."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Model bundle path; defaults to OUTPUT_DIRECTORY/model.ptb.",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        help="Molecular XYZ input; defaults to the bundled samples.xyz.",
    )
    parser.add_argument(
        "--reference_model",
        type=Path,
        help=(
            "Torch reference model; defaults to "
            "OUTPUT_DIRECTORY/reference.model."
        ),
    )
    parser.add_argument("--lmp", default="lmp", help="LAMMPS executable.")
    parser.add_argument(
        "--launcher",
        default="mpirun -np 2",
        help="Two-rank MPI launcher; the tests require exactly two ranks.",
    )
    parser.add_argument(
        "--mace_eval", default="mace_eval_configs", help="MACE CLI executable."
    )
    parser.add_argument(
        "--mace_device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used only by the independent MACE CLI reference.",
    )
    parser.add_argument(
        "--output_directory",
        type=Path,
        default=Path("results"),
        help="Directory for the model, logs, dumps, and JSON summary.",
    )
    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="Use the model files supplied by --model and --reference_model.",
    )
    args = parser.parse_args()

    # Resolve paths up front so command logs and failure messages refer to the
    # actual artifacts used by the regression.
    script_directory = Path(__file__).resolve().parent
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    args.model = (
        (output_directory / "model.ptb")
        if args.model is None
        else args.model.resolve()
    )
    args.reference_model = (
        (output_directory / "reference.model")
        if args.reference_model is None
        else args.reference_model.resolve()
    )
    args.samples = (
        (script_directory / "samples.xyz")
        if args.samples is None
        else args.samples.resolve()
    )

    # Keep validation explicit and make plotting/cache behavior suitable for
    # headless CI environments.
    environment = os.environ.copy()
    environment["JCN_VALIDATE_COMMUNICATION"] = "1"
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")

    # Export fresh artifacts unless the caller intentionally supplied existing
    # model files, then fail early if any required input is unavailable.
    if not args.skip_export:
        run_command(
            "export",
            [
                sys.executable,
                "export_model.py",
                "--output",
                str(args.model),
                "--reference_output",
                str(args.reference_model),
            ],
            working_directory=script_directory,
            output_directory=output_directory,
            environment=environment,
        )
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if not args.samples.exists():
        raise FileNotFoundError(args.samples)
    if not args.reference_model.exists():
        raise FileNotFoundError(args.reference_model)

    summary = {
        "low_padding": run_trajectory_case(
            "low_padding",
            1.01,
            1.01,
            args=args,
            script_directory=script_directory,
            output_directory=output_directory,
            environment=environment,
        ),
        "newton": run_newton_cases(
            args=args,
            script_directory=script_directory,
            output_directory=output_directory,
            environment=environment,
        ),
        "molecule": run_molecular_prediction_case(
            args=args,
            script_directory=script_directory,
            output_directory=output_directory,
            environment=environment,
        ),
    }
    summary_path = write_summary(summary, output_directory)
    report_path = write_report(summary, output_directory)
    print(
        f"\nPASS communication regression; summary: {summary_path}; "
        f"report: {report_path}"
    )

if __name__ == "__main__":
    main()
