#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_ID="${1:-}"
SEED_START="${2:-}"

if [ -z "${RUN_ID}" ] || [ -z "${SEED_START}" ]; then
  echo "Usage: bash scripts/repro_bbob_4alg_seedwindow.sh <RUN_ID> <SEED_START>" >&2
  echo "Example: bash scripts/repro_bbob_4alg_seedwindow.sh bbob_seedwin_0to6 0" >&2
  exit 2
fi

if ! [[ "${SEED_START}" =~ ^-?[0-9]+$ ]]; then
  echo "SEED_START must be an integer, got: ${SEED_START}" >&2
  exit 2
fi

POP=200
BUDGET=10000
BOUNDS_LOW=-5
BOUNDS_HIGH=5
DIMS=(30 100 200 300 500)
FIDS=( $(seq 1 24) )

SEEDS=()
for ((i=0; i<7; i++)); do
  SEEDS+=( $((SEED_START + i)) )
done

OFFSETS_DIR="offsets"

BASE_IMG="images/${RUN_ID}"
BASE_SUM="artifacts/eval_summaries/${RUN_ID}"
BASE_LOG="logs/${RUN_ID}"
mkdir -p "${BASE_IMG}" "${BASE_SUM}" "${BASE_LOG}"

run_eval() {
  local optimizer="$1"   # les|lde|lga|pom
  local tag="$2"
  local base_ckpt="$3"
  local base_cfg="$4"
  local md_ckpt="$5"
  local md_cfg="$6"

  local outdir="${BASE_IMG}/${optimizer}_${tag}"
  local sumdir="${BASE_SUM}/${optimizer}_${tag}"
  local logfile="${BASE_LOG}/${optimizer}_${tag}.out"
  mkdir -p "${outdir}" "${sumdir}"

  echo "[RUN] ${optimizer}_${tag} -> ${logfile}"
  "${PYTHON_BIN}" "scripts/eval_compare_frameworks.py" \
    --optimizer "${optimizer}" \
    --dims "${DIMS[@]}" \
    --popsize "${POP}" \
    --budget "${BUDGET}" \
    --bounds "${BOUNDS_LOW}" "${BOUNDS_HIGH}" \
    --fids "${FIDS[@]}" \
    --seeds "${SEEDS[@]}" \
    --bbob-offsets-dir "${OFFSETS_DIR}" \
    --outdir "${outdir}" \
    --summary "${sumdir}" \
    --curve best-gen \
    --adapt-lr 1e-4 \
    --loss-eps 1e-12 \
    --variant "baseline=${base_ckpt}" \
    --variant "md_j0=${md_ckpt}" \
    --variant "md_j1=${md_ckpt}" \
    --variant "md_j3=${md_ckpt}" \
    --variant "md_j5=${md_ckpt}" \
    --variant-config "baseline=${base_cfg}" \
    --variant-config "md_j0=${md_cfg}" \
    --variant-config "md_j1=${md_cfg}" \
    --variant-config "md_j3=${md_cfg}" \
    --variant-config "md_j5=${md_cfg}" \
    --ssft-variant md_j1=1 \
    --ssft-variant md_j3=3 \
    --ssft-variant md_j5=5 \
    >"${logfile}" 2>&1
}

# BBOB four-optimizer evaluation.
run_eval \
  "les" "metadistill_bbob" \
  "checkpoints/baselines/les_metabbo.pt" \
  "configs/les_config.json" \
  "checkpoints/metadistill/bbob/les.pt" \
  "configs/les_d10_pop200.json"

run_eval \
  "lde" "metadistill_bbob" \
  "checkpoints/baselines/lde_policy_gradient.pt" \
  "configs/lde_pg_pop200_d10_sample_h64.json" \
  "checkpoints/metadistill/bbob/lde.pt" \
  "configs/lde_d10_pop200.json"

run_eval \
  "lga" "metadistill_bbob" \
  "checkpoints/baselines/lga_metabbo.pt" \
  "configs/lga_config.json" \
  "checkpoints/metadistill/bbob/lga.pt" \
  "configs/lga_d10_pop200_parentsoft_tau1.json"

run_eval \
  "pom" "metadistill_bbob" \
  "checkpoints/baselines/pom_original.pt" \
  "configs/pom_config.json" \
  "checkpoints/metadistill/bbob/pom.pt" \
  "configs/pom_d10_pop200.json"

echo "[OK] Finished: RUN_ID=${RUN_ID} seeds=${SEEDS[*]}"
