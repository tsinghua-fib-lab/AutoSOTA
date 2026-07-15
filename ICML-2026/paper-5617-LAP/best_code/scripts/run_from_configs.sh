#!/usr/bin/env bash
#
# Generic data-gen + train pipeline.
#
# Usage:
#   bash scripts/run_from_configs.sh MASTER_TAG DATA_CFG TRAIN_CFG RUN_TAG SEED [extra args for train.py]
#
# Env knobs:
#   RUNS_DIR    : top-level output directory  (default: runs)
#   DEVICE      : cuda | cpu                  (default: cuda)
#   REGEN_DATA  : 1 to force re-gen of bundle (default: 0)
#   WANDB_ON    : 1 to enable wandb in train  (default: 1)
#   SKIP_DONE   : 1 to skip if final.pt exists (default: 1)
#                 set to 0 if you want re-runs into existing dirs to overwrite.
#
# Layout:
#   ${RUNS_DIR}/${MASTER_TAG}/data/${data_cfg_basename}_seed${SEED}/bundle.pt
#   ${RUNS_DIR}/${MASTER_TAG}/train/${RUN_TAG}_seed${SEED}/...
#
# `data_cfg_basename` is the basename of DATA_CFG without extension. So sharing
# the same data bundle across sweeps just means pointing at the same DATA_CFG.

set -euo pipefail

# --- 1. Positional args ---
MASTER_TAG="${1:-default}"
DATA_CFG="${2:-}"
TRAIN_CFG="${3:-}"
RUN_TAG="${4:-}"
SEED="${5:-0}"
shift 5 || true   # remaining "$@" → extra train.py args

# --- 2. Env knobs ---
RUNS_DIR="${RUNS_DIR:-runs}"
DEVICE="${DEVICE:-cuda}"
REGEN_DATA="${REGEN_DATA:-0}"
WANDB_ON="${WANDB_ON:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
DATA_TAG_SUFFIX="${DATA_TAG_SUFFIX:-}"   # appended to data dir name (e.g. "_kill_poly")

# --- 3. Validation ---
[[ -n "${DATA_CFG}"    ]] || { echo "ERROR: DATA_CFG (arg 2) is required" >&2; exit 1; }
[[ -n "${TRAIN_CFG}"   ]] || { echo "ERROR: TRAIN_CFG (arg 3) is required" >&2; exit 1; }
[[ -f "${DATA_CFG}"    ]] || { echo "ERROR: DATA_CFG not found: ${DATA_CFG}" >&2; exit 1; }
[[ -f "${TRAIN_CFG}"   ]] || { echo "ERROR: TRAIN_CFG not found: ${TRAIN_CFG}" >&2; exit 1; }

# --- 4. Directory layout ---
data_base="$(basename "${DATA_CFG%.*}")"

# Auto-derive a DATA_TAG_SUFFIX from `--set pot=...` / `--set kill_condition=...`
# in the extra args, so multiple potential/kill combinations don't clobber the
# same bundle.pt. Explicit DATA_TAG_SUFFIX (env var) wins if set.
if [[ -z "${DATA_TAG_SUFFIX}" ]]; then
    auto_suffix=""
    # walk "$@" looking for --set key=value pairs
    args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[i]}" == "--set" ]]; then
            kv="${args[i+1]:-}"
            case "${kv}" in
                pot=*)             auto_suffix+="_$(printf '%s' "${kv#pot=}")" ;;
                kill_condition=true)  auto_suffix+="_kill" ;;
                kill_condition=false) auto_suffix+="_nokill" ;;
            esac
        fi
    done
    DATA_TAG_SUFFIX="${auto_suffix}"
fi
data_tag="${data_base}${DATA_TAG_SUFFIX}_seed${SEED}"

if [[ -n "${RUN_TAG}" ]]; then
    train_tag="${RUN_TAG}_seed${SEED}"
else
    train_base="$(basename "${TRAIN_CFG%.*}")"
    train_tag="${train_base}_seed${SEED}"
fi

master_dir="${RUNS_DIR}/${MASTER_TAG}"
data_dir="${master_dir}/data/${data_tag}"
train_dir="${master_dir}/train/${train_tag}"

# --- 5. Skip if already done ---
if [[ "${SKIP_DONE}" == "1" && -f "${train_dir}/final.pt" ]]; then
    echo "[run] SKIP: final.pt already exists at ${train_dir}/final.pt"
    echo "[run] (set SKIP_DONE=0 or rm the directory to re-run)"
    exit 0
fi

mkdir -p "${data_dir}" "${train_dir}"
data_path="${data_dir}/bundle.pt"

# Copy config alongside outputs (-n: don't overwrite if a previous run saved one)
_copy_cfg_nc() { local src="$1" dst="$2"; cp -n "${src}" "${dst}" 2>/dev/null || true; }

# --- 6. Data generation ---
_copy_cfg_nc "${DATA_CFG}" "${data_dir}/data_config.yaml"

if [[ -f "${data_path}" && "${REGEN_DATA}" != "1" ]]; then
    echo "[run] data exists: ${data_path} (set REGEN_DATA=1 to force regen)"
else
    echo "[run] generating data -> ${data_path}"
    DATA_EXTRA=()
    args=("$@")
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[i]}" == "--set" ]]; then
            DATA_EXTRA+=( --set "${args[i+1]:-}" )
            ((i++)) || true
        fi
    done
    python -u data_generator.py \
        --config "${DATA_CFG}" \
        --set seed="${SEED}" \
        --set device=cpu \
        --set out="${data_path}" \
        "${DATA_EXTRA[@]}"
fi

# --- 8. Build --set overrides for train.py ---
_copy_cfg_nc "${TRAIN_CFG}" "${train_dir}/train_config.yaml"

mapfile -t KV_OVERRIDES < <(python - "${TRAIN_CFG}" "${SEED}" "${DEVICE}" "${data_path}" "${train_dir}" "${WANDB_ON}" <<'PY'
import sys, yaml
train_cfg_path = sys.argv[1]
seed, device, data_path, outdir, wandb_on = sys.argv[2:7]
wandb_val = "true" if str(wandb_on) == "1" else "false"

cfg = yaml.safe_load(open(train_cfg_path)) or {}

def pick_key(d, keys, default):
    for k in keys:
        if k in d:
            return k
    return default

pairs = [
    ("seed",   seed),
    ("device", device),
    (pick_key(cfg, ["data", "data_path", "dataset"], "data"),       data_path),
    (pick_key(cfg, ["outdir", "output_dir", "run_dir"], "outdir"),  outdir),
    (pick_key(cfg, ["wandb", "use_wandb"], "wandb"),                wandb_val),
]
for k, v in pairs:
    print(f"{k}={v}")
PY
)

SET_ARGS=()
for kv in "${KV_OVERRIDES[@]}"; do
    SET_ARGS+=( --set "${kv}" )
done

# --- 9. Run training ---
WANDB_NAME="${train_tag}"

echo "============================================================"
echo "[run] master_dir = ${master_dir}"
echo "[run] data_path  = ${data_path}"
echo "[run] train_dir  = ${train_dir}"
echo "[run] wandb_name = ${WANDB_NAME}"
echo "[run] extra args = $*"
echo "============================================================"

python -u train.py \
    --config "${TRAIN_CFG}" \
    "${SET_ARGS[@]}" \
    --wandb-name "${WANDB_NAME}" \
    "$@"

echo "[run] Done. Output: ${train_dir}"