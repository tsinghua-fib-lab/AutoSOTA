#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the complete two-rank communication regression.

set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Runtime configuration. Each value can be overridden in the environment.
# Two MPI ranks and two visible GPUs are required because every LAMMPS rank
# creates one PJRT client and is expected to see one GPU after rank isolation.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# Avoid having two rank-local XLA allocators reserve nearly all GPU memory.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# Ask the connector to verify that runtime forward/reverse exchanges match the
# communication sites recorded in the exported model.
export JCN_VALIDATE_COMMUNICATION="${JCN_VALIDATE_COMMUNICATION:-1}"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
LAMMPS_EXECUTABLE="${LAMMPS_EXECUTABLE:-lmp}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun -np 2}"
MACE_EVAL_EXECUTABLE="${MACE_EVAL_EXECUTABLE:-mace_eval_configs}"
MACE_REFERENCE_DEVICE="${MACE_REFERENCE_DEVICE:-cpu}"
OUTPUT_DIRECTORY="${OUTPUT_DIRECTORY:-${SCRIPT_DIRECTORY}/results}"

echo "Communication regression configuration"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "  XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}"
echo "  JCN_VALIDATE_COMMUNICATION=${JCN_VALIDATE_COMMUNICATION}"
echo "  PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
echo "  LAMMPS_EXECUTABLE=${LAMMPS_EXECUTABLE}"
echo "  MPI_LAUNCHER=${MPI_LAUNCHER}"
echo "  MACE_EVAL_EXECUTABLE=${MACE_EVAL_EXECUTABLE}"
echo "  MACE_REFERENCE_DEVICE=${MACE_REFERENCE_DEVICE}"
echo "  OUTPUT_DIRECTORY=${OUTPUT_DIRECTORY}"

cd "${SCRIPT_DIRECTORY}"
# run_regression.py performs export unless --skip_export is supplied, launches
# every LAMMPS case, applies all pass/fail criteria, and writes summary.json and
# report.md below OUTPUT_DIRECTORY.
exec "${PYTHON_EXECUTABLE}" run_regression.py \
  --lmp "${LAMMPS_EXECUTABLE}" \
  --launcher "${MPI_LAUNCHER}" \
  --mace_eval "${MACE_EVAL_EXECUTABLE}" \
  --mace_device "${MACE_REFERENCE_DEVICE}" \
  --output_directory "${OUTPUT_DIRECTORY}" \
  "$@"
