#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NETCFG_DIR="${NETCFG_DIR:-${REPO_ROOT}/tasks/neuroevolution/config}"

TASKS=(2)
SEEDS=(0 1 2)
POPSIZE=200
GENERATIONS=500
STEPS=256
ADAPT_LR=1e-4
MAX_JOBS="${MAX_JOBS:-4}"

TAG="${TAG:-control_lunarlander_seed0-2}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/artifacts/neuroevo/${TAG}}"
LOG_DIR="${OUT_ROOT}/logs"

if [ -e "${OUT_ROOT}" ]; then
  echo "[ERROR] OUT_ROOT exists: ${OUT_ROOT}" >&2
  echo "[HINT] remove it or set OUT_ROOT/TAG before rerun." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

run_bg() {
  local name="$1"; shift
  local outdir="$1"; shift

  mkdir -p "${outdir}"
  echo "[LAUNCH] ${name} -> ${outdir}"

  "${PYTHON_BIN}" "$@" > "${LOG_DIR}/${name}.out" 2>&1 &

  while [ "$(jobs -pr | wc -l)" -ge "${MAX_JOBS}" ]; do
    sleep 10
  done
}

# -------------------- CKPTs --------------------
BASE_LES_CKPT="checkpoints/baselines/les_metabbo.pt"
MD_LES_CKPT="checkpoints/metadistill/control/les.pt"

BASE_LDE_CKPT="checkpoints/baselines/lde_policy_gradient.pt"
MD_LDE_CKPT="checkpoints/metadistill/control/lde.pt"

BASE_LGA_CKPT="checkpoints/baselines/lga_metabbo.pt"
MD_LGA_CKPT="checkpoints/metadistill/control/lga.pt"

BASE_POM_CKPT="checkpoints/baselines/pom_original.pt"
MD_POM_CKPT="checkpoints/metadistill/control/pom.pt"

# -------------------- LES --------------------
run_bg \
  "les_baseline" \
  "${OUT_ROOT}/les_baseline" \
  scripts/run_neuroevo.py \
    --algo GradFreeLES \
    --algo-config configs/les_d10_pop200.json \
    --ckpt "${BASE_LES_CKPT}" \
    --tasks "${TASKS[@]}" \
    --net-config-dir "${NETCFG_DIR}" \
    --steps "${STEPS}" \
    --generations "${GENERATIONS}" \
    --seeds "${SEEDS[@]}" \
    --popsize "${POPSIZE}" \
    --adapt-lr 0 \
    --outdir "${OUT_ROOT}/les_baseline"

run_bg \
  "les_md_j0" \
  "${OUT_ROOT}/les_md_j0" \
  scripts/run_neuroevo.py \
    --algo GradBasedLES \
    --algo-config configs/les_d10_pop200.json \
    --ckpt "${MD_LES_CKPT}" \
    --tasks "${TASKS[@]}" \
    --net-config-dir "${NETCFG_DIR}" \
    --steps "${STEPS}" \
    --generations "${GENERATIONS}" \
    --seeds "${SEEDS[@]}" \
    --popsize "${POPSIZE}" \
    --adapt-lr 0 \
    --outdir "${OUT_ROOT}/les_md_j0"

for J in 1 3 5; do
  run_bg \
    "les_md_ssft_j${J}" \
    "${OUT_ROOT}/les_md_ssft_j${J}" \
    scripts/run_neuroevo.py \
      --algo GradBasedLES \
      --algo-config configs/les_d10_pop200.json \
      --ckpt "${MD_LES_CKPT}" \
      --tasks "${TASKS[@]}" \
      --net-config-dir "${NETCFG_DIR}" \
      --steps "${STEPS}" \
      --generations "${GENERATIONS}" \
      --seeds "${SEEDS[@]}" \
      --popsize "${POPSIZE}" \
      --adapt-lr "${ADAPT_LR}" \
      --ss-interval "${J}" \
      --ssft-mode mu \
      --outdir "${OUT_ROOT}/les_md_ssft_j${J}"
done

# -------------------- LDE --------------------
run_bg \
  "lde_baseline" \
  "${OUT_ROOT}/lde_baseline" \
  scripts/run_neuroevo.py \
    --algo LDE \
    --algo-config configs/lde_pg_pop200_d10_sample_h64.json \
    --ckpt "${BASE_LDE_CKPT}" \
    --tasks "${TASKS[@]}" \
    --net-config-dir "${NETCFG_DIR}" \
    --steps "${STEPS}" \
    --generations "${GENERATIONS}" \
    --seeds "${SEEDS[@]}" \
    --popsize "${POPSIZE}" \
    --adapt-lr 0 \
    --outdir "${OUT_ROOT}/lde_baseline"

run_bg \
  "lde_md_j0" \
  "${OUT_ROOT}/lde_md_j0" \
  scripts/run_neuroevo.py \
    --algo LDE \
    --algo-config configs/lde_d10_pop200.json \
    --ckpt "${MD_LDE_CKPT}" \
    --tasks "${TASKS[@]}" \
    --net-config-dir "${NETCFG_DIR}" \
    --steps "${STEPS}" \
    --generations "${GENERATIONS}" \
    --seeds "${SEEDS[@]}" \
    --popsize "${POPSIZE}" \
    --adapt-lr 0 \
    --outdir "${OUT_ROOT}/lde_md_j0"

for J in 1 3 5; do
  run_bg \
    "lde_md_ssft_j${J}" \
    "${OUT_ROOT}/lde_md_ssft_j${J}" \
    scripts/run_neuroevo.py \
      --algo LDE \
      --algo-config configs/lde_d10_pop200.json \
      --ckpt "${MD_LDE_CKPT}" \
      --tasks "${TASKS[@]}" \
      --net-config-dir "${NETCFG_DIR}" \
      --steps "${STEPS}" \
      --generations "${GENERATIONS}" \
      --seeds "${SEEDS[@]}" \
      --popsize "${POPSIZE}" \
      --adapt-lr "${ADAPT_LR}" \
      --ss-interval "${J}" \
      --ssft-mode mu \
      --outdir "${OUT_ROOT}/lde_md_ssft_j${J}"
done

# -------------------- LGA --------------------
run_bg \
  "lga_baseline" \
  "${OUT_ROOT}/lga_baseline" \
  scripts/run_neuroevo.py \
    --algo LGA \
    --algo-config configs/lga_config.json \
    --ckpt "${BASE_LGA_CKPT}" \
    --tasks "${TASKS[@]}" \
    --net-config-dir "${NETCFG_DIR}" \
    --steps "${STEPS}" \
    --generations "${GENERATIONS}" \
    --seeds "${SEEDS[@]}" \
    --popsize "${POPSIZE}" \
    --adapt-lr 0 \
    --outdir "${OUT_ROOT}/lga_baseline"

run_bg \
  "lga_md_j0" \
  "${OUT_ROOT}/lga_md_j0" \
  scripts/run_neuroevo.py \
    --algo GradBasedLGA \
    --algo-config configs/lga_d10_pop200_parentsoft_tau1.json \
    --ckpt "${MD_LGA_CKPT}" \
    --tasks "${TASKS[@]}" \
    --net-config-dir "${NETCFG_DIR}" \
    --steps "${STEPS}" \
    --generations "${GENERATIONS}" \
    --seeds "${SEEDS[@]}" \
    --popsize "${POPSIZE}" \
    --adapt-lr 0 \
    --outdir "${OUT_ROOT}/lga_md_j0"

for J in 1 3 5; do
  run_bg \
    "lga_md_ssft_j${J}" \
    "${OUT_ROOT}/lga_md_ssft_j${J}" \
    scripts/run_neuroevo.py \
      --algo GradBasedLGA \
      --algo-config configs/lga_d10_pop200_parentsoft_tau1.json \
      --ckpt "${MD_LGA_CKPT}" \
      --tasks "${TASKS[@]}" \
      --net-config-dir "${NETCFG_DIR}" \
      --steps "${STEPS}" \
      --generations "${GENERATIONS}" \
      --seeds "${SEEDS[@]}" \
      --popsize "${POPSIZE}" \
      --adapt-lr "${ADAPT_LR}" \
      --ss-interval "${J}" \
      --ssft-mode mu \
      --outdir "${OUT_ROOT}/lga_md_ssft_j${J}"
done

# -------------------- POM --------------------
run_bg \
  "pom_baseline" \
  "${OUT_ROOT}/pom_baseline" \
  scripts/run_neuroevo.py \
    --algo POM \
    --algo-config configs/pom_config.json \
    --ckpt "${BASE_POM_CKPT}" \
    --tasks "${TASKS[@]}" \
    --net-config-dir "${NETCFG_DIR}" \
    --steps "${STEPS}" \
    --generations "${GENERATIONS}" \
    --seeds "${SEEDS[@]}" \
    --popsize "${POPSIZE}" \
    --adapt-lr 0 \
    --outdir "${OUT_ROOT}/pom_baseline"

run_bg \
  "pom_md_j0" \
  "${OUT_ROOT}/pom_md_j0" \
  scripts/run_neuroevo.py \
    --algo POM \
    --algo-config configs/pom_d10_pop200.json \
    --ckpt "${MD_POM_CKPT}" \
    --tasks "${TASKS[@]}" \
    --net-config-dir "${NETCFG_DIR}" \
    --steps "${STEPS}" \
    --generations "${GENERATIONS}" \
    --seeds "${SEEDS[@]}" \
    --popsize "${POPSIZE}" \
    --adapt-lr 0 \
    --outdir "${OUT_ROOT}/pom_md_j0"

for J in 1 3 5; do
  run_bg \
    "pom_md_ssft_j${J}" \
    "${OUT_ROOT}/pom_md_ssft_j${J}" \
    scripts/run_neuroevo.py \
      --algo POM \
      --algo-config configs/pom_d10_pop200.json \
      --ckpt "${MD_POM_CKPT}" \
      --tasks "${TASKS[@]}" \
      --net-config-dir "${NETCFG_DIR}" \
      --steps "${STEPS}" \
      --generations "${GENERATIONS}" \
      --seeds "${SEEDS[@]}" \
      --popsize "${POPSIZE}" \
      --adapt-lr "${ADAPT_LR}" \
      --ss-interval "${J}" \
      --ssft-mode mu \
      --outdir "${OUT_ROOT}/pom_md_ssft_j${J}"
done

echo "[INFO] waiting for ${TAG} jobs to finish..."
wait
echo "[OK] Done: ${OUT_ROOT}"
