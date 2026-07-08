#!/usr/bin/env bash
# SOTA iter-5: 9 variants (baseline + j0,j1,j2,j3,j4,j5,j6,j8)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

POP=200
BUDGET=10000
BOUNDS_LOW=-5
BOUNDS_HIGH=5
DIM=30
FIDS=( $(seq 1 24) )

VARIANTS=(
  baseline=checkpoints/baselines/pom_original.pt
  md_j0=checkpoints/metadistill/bbob/pom.pt
  md_j1=checkpoints/metadistill/bbob/pom.pt
  md_j2=checkpoints/metadistill/bbob/pom.pt
  md_j3=checkpoints/metadistill/bbob/pom.pt
  md_j4=checkpoints/metadistill/bbob/pom.pt
  md_j5=checkpoints/metadistill/bbob/pom.pt
  md_j6=checkpoints/metadistill/bbob/pom.pt
  md_j8=checkpoints/metadistill/bbob/pom.pt
)

VARIANT_CONFIGS=(
  baseline=configs/pom_config.json
  md_j0=configs/pom_d10_pop200.json
  md_j1=configs/pom_d10_pop200.json
  md_j2=configs/pom_d10_pop200.json
  md_j3=configs/pom_d10_pop200.json
  md_j4=configs/pom_d10_pop200.json
  md_j5=configs/pom_d10_pop200.json
  md_j6=configs/pom_d10_pop200.json
  md_j8=configs/pom_d10_pop200.json
)

SSFT_VARIANTS=(
  md_j1=1
  md_j2=2
  md_j3=3
  md_j4=4
  md_j5=5
  md_j6=6
  md_j8=8
)

build_args() {
  for v in "${VARIANTS[@]}"; do echo -n "--variant ${v} "; done
  for v in "${VARIANT_CONFIGS[@]}"; do echo -n "--variant-config ${v} "; done
  for v in "${SSFT_VARIANTS[@]}"; do echo -n "--ssft-variant ${v} "; done
}

RUN_EVAL() {
  local outdir="${1}"; local summary="${2}"; shift 2
  # shellcheck disable=SC2046
  "${PYTHON_BIN}" scripts/eval_compare_frameworks.py \
    --optimizer pom --dims "${DIM}" --popsize "${POP}" --budget "${BUDGET}" \
    --bounds "${BOUNDS_LOW}" "${BOUNDS_HIGH}" --fids "${FIDS[@]}" \
    --seeds "$@" --bbob-offsets-dir offsets \
    --outdir "${outdir}" --summary "${summary}" --curve best-gen \
    --adapt-lr 1e-4 --loss-eps 1e-12 \
    $(build_args)
}

echo "=== Running POM BBOB evaluation at d=${DIM} ==="
echo "Window 1: seeds 0-6"
RUN_EVAL images/bbob_seedwin_0to6/pom artifacts/eval_summaries/bbob_seedwin_0to6 0 1 2 3 4 5 6

echo "Window 2: seeds 1-7"
RUN_EVAL images/bbob_seedwin_1to7/pom artifacts/eval_summaries/bbob_seedwin_1to7 1 2 3 4 5 6 7

echo "Window 3: seeds 2-8"
RUN_EVAL images/bbob_seedwin_2to8/pom artifacts/eval_summaries/bbob_seedwin_2to8 2 3 4 5 6 7 8

echo "=== Computing LAD ==="
"${PYTHON_BIN}" scripts/compute_lad_shifted.py \
  artifacts/eval_summaries/bbob_seedwin_0to6/pom_framework_compare_*_B10000_pop200_dims30_seeds0-1-2-3-4-5-6.json \
  artifacts/eval_summaries/bbob_seedwin_1to7/pom_framework_compare_*_B10000_pop200_dims30_seeds1-2-3-4-5-6-7.json \
  artifacts/eval_summaries/bbob_seedwin_2to8/pom_framework_compare_*_B10000_pop200_dims30_seeds2-3-4-5-6-7-8.json

echo "=== Done ==="
