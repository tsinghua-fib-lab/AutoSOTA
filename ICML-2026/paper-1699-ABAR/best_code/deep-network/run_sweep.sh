#!/usr/bin/env bash
set -euo pipefail

############################################
# User config
############################################
BASE_EPOCHS=5
LARGE_EPOCHS=20

LR=4.0
MOM=0
CLIP=1.0

OUT_DIR="runs_sweeps"
ML_DATA_ROOT="${ML_DATA_ROOT:-/userhome/cs3/zmsxsl/data}"

# Simulate epsilons
# EPSILONS=(0 0.1 0.2 0.3 0.4 0.5 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 12.0 15.0)
EPSILONS=(0.6 0.7 0.8 0.9 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5)
DP_DATALOADER_VALUES=(True False)

# FTRL-specific knobs (paper-style)
RESTART=1
TREE_COMPLETION=True
EFFI_NOISE=False

mkdir -p "${OUT_DIR}"
stamp() { date -u +"%Y%m%dT%H%M%SZ"; }

run_one () {
  local algo="$1"
  local eps="$2"
  local dp_dl="$3"  # New argument for dp_dataloader
  local tag="$4"
  local dataset="$5"
  local epochs="$6"
  local batch="$7"
  
  local ts
  ts="$(stamp)"
  # Added dp_dl to the log filename to distinguish runs
  local log="${OUT_DIR}/${DATA}_${algo}_${tag}_eps${eps}_dpdl${dp_dl}_${ts}.log"

  echo "==== RUN algo=${algo} eps=${eps} dp_dataloader=${dp_dl} tag=${tag} -> ${log}"

  export ML_DATA="${ML_DATA_ROOT}"
  
  python -u main.py \
    --data="${dataset}" \
    --algo="${algo}" \
    --epochs="${epochs}" \
    --batch_size="${batch}" \
    --learning_rate="${LR}" \
    --momentum="${MOM}" \
    --l2_norm_clip="${CLIP}" \
    --noise_multiplier="${eps}" \
    --dp_dataloader="${dp_dl}" \
    --restart="${RESTART}" \
    --tree_completion="${TREE_COMPLETION}" \
    --effi_noise="${EFFI_NOISE}" \
    --dir="${OUT_DIR}" \
    2>&1 | tee "${log}"
}

run_dataset () {
  local dataset="$1"
  local batch_size="$2"
  local epochs="$3"
  DATA="${dataset}"
  BATCH="${batch_size}"

  # We nest the loops: For every Algo -> For every DP_Dataloader setting -> For every Epsilon
  
  # ====== 1) DP-FTRL sweep ======
  for dp_val in "${DP_DATALOADER_VALUES[@]}"; do
    for e in "${EPSILONS[@]}"; do
      run_one "ftrl_dp" "${e}" "${dp_val}" "sweep" "${dataset}" "${epochs}" "${batch_size}"
    done
  done

  # ====== 2) DP-FTRL Matrix sweep ======
  for dp_val in "${DP_DATALOADER_VALUES[@]}"; do
    for e in "${EPSILONS[@]}"; do
      run_one "ftrl_dp_matrix" "${e}" "${dp_val}" "sweep" "${dataset}" "${epochs}" "${batch_size}"
    done
  done

  # ====== 3) DP-SGD no-amplification reporting sweep ======
  for dp_val in "${DP_DATALOADER_VALUES[@]}"; do
    for e in "${EPSILONS[@]}"; do
      run_one "sgd_noamp" "${e}" "${dp_val}" "sweep" "${dataset}" "${epochs}" "${batch_size}"
    done
  done
}

############################################
# Run all datasets
############################################

run_dataset "mnist"        250 "${BASE_EPOCHS}"
run_dataset "cifar10"      500 "${LARGE_EPOCHS}"
run_dataset "emnist_merge" 500 "${LARGE_EPOCHS}"

echo "All sweeps finished. Logs in: ${OUT_DIR}"
echo "Results JSONL should be under: ${OUT_DIR}/results.jsonl"
