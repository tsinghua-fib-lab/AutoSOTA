#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_ENTRY="${REPO_ROOT}/train_tidegs.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"
ROOT="${ROOT:-${REPO_ROOT}/outputs/tidegs}"
SRC="${SRC:-${MATRIXCITY_SCENE_DIR:-}}"
PLY="${PLY:-${TIDEGS_DENSE_PLY:-}}"
MANIFEST="${MANIFEST:-${TIDEGS_PREBUILT_MANIFEST:-}}"
if [[ -n "${SCHED_CACHE:-}" ]]; then
  SCHED_CACHE_USER_SET=1
else
  SCHED_CACHE_USER_SET=0
fi
SCHED_CACHE="${SCHED_CACHE:-${ROOT}/schedule_cache/oneb_bigcity}"
DECODE_DATASET_PATH="${DECODE_DATASET_PATH:-}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/output/runs}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/ssd_cache}"
RUN_TAG="${RUN_TAG:-$(date +"%Y%m%d_%H%M%S")_matrixcity_1b}"

MODE="train"
DRY_RUN=0
DEBUG_LOGGING=0
VERBOSE_TERMINAL=0
ITERATIONS=240
DEBUG_MAX_TRAIN_CAMERAS=256
DEBUG_CAMERA_SAMPLE_MODE="linspace"
DEBUG_CAMERA_SAMPLE_START=0
MAX_RAM_GB=32
NUM_CLUSTERS=64
PROJECTION_CHUNK=2
CHECKPOINT_MODE="incremental"
RESIDENT_POLICY="topc_balanced"
RESIDENT_LAMBDA_LIST="0.3"
RESIDENT_DECAY_LIST="0.95"
BALANCED_SEED_FRACTION_LIST="0.25"
BSZ_LIST="16"
CAPACITY_LIST="2048"
CHECKPOINT_ITER=500
RESUME_TO_ITER=1000
START_CHECKPOINT=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Modes:
  train       Run MatrixCity 1B training with the selected configuration.
  checkpoint  Run 1000 iterations and write an incremental checkpoint.
  resume      Resume from --start-checkpoint to --resume-to-iter.
  summary     Summarize python.log files under this RUN_TAG output tree.

Options:
  --mode MODE                 train|checkpoint|resume|summary
  --gpu ID                    GPU id (default: ${GPU})
  --run-tag TAG               Experiment tag (default: timestamped)
  --root DIR                  Large output root (default: ${ROOT})
  --src DIR                   MatrixCity source dir
  --ply PATH                  1B PLY path
  --manifest PATH             Prebuilt streaming_init_manifest.json
  --schedule-cache DIR        Camera schedule cache dir
  --decode-dataset-path DIR   Optional decoded raw image cache dir
  --iterations N              Iterations for sweep mode (default: ${ITERATIONS})
  --debug-max-train-cameras N Camera cap; -1 uses all training cameras (default: ${DEBUG_MAX_TRAIN_CAMERAS})
  --debug-camera-sample-mode M linspace|contiguous|window (default: ${DEBUG_CAMERA_SAMPLE_MODE})
  --debug-camera-sample-start N Start index for window mode (default: ${DEBUG_CAMERA_SAMPLE_START})
  --bsz N                     Single batch size for release runs
  --capacity N                Single resident block capacity for release runs
  --bsz-list "LIST"           Batch sizes for sweeps (default: "${BSZ_LIST}")
  --capacity-list "LIST"      Resident block capacities for sweeps (default: "${CAPACITY_LIST}")
  --projection-chunk N        projection_max_cameras_per_chunk (default: ${PROJECTION_CHUNK})
  --max-ram-gb N              RAM cache budget (default: ${MAX_RAM_GB})
  --checkpoint-mode MODE      incremental|snapshot (default: ${CHECKPOINT_MODE})
  --resident-policy POLICY    topc_strict|topc_balanced (default: ${RESIDENT_POLICY})
  --resident-lambda VALUE     Single resident-set mixing weight
  --resident-decay VALUE      Single resident recency decay value
  --balanced-seed-fraction VALUE
                              Single topc_balanced seed-capacity fraction
  --resident-lambda-list "LIST" resident-set mixing weights for sweeps (default: "${RESIDENT_LAMBDA_LIST}")
  --resident-decay-list "LIST"  resident recency decay values for sweeps (default: "${RESIDENT_DECAY_LIST}")
  --balanced-seed-fraction-list "LIST"
                              topc_balanced seed-capacity fractions (default: "${BALANCED_SEED_FRACTION_LIST}")
  --checkpoint-iter N         Checkpoint iteration for checkpoint mode (default: ${CHECKPOINT_ITER})
  --resume-to-iter N          Final iteration for resume mode (default: ${RESUME_TO_ITER})
  --start-checkpoint DIR      Checkpoint dir for resume mode
  --debug-logging             Enable detailed TideGS runtime logs while keeping terminal quiet
  --verbose-terminal          Stream training subprocess output to the terminal
  --dry-run                   Write commands.sh only

Environment overrides:
  PYTHON_BIN, GPU, ROOT, SRC, PLY, MANIFEST, MATRIXCITY_SCENE_DIR,
  TIDEGS_DENSE_PLY, TIDEGS_PREBUILT_MANIFEST, SCHED_CACHE, OUT_ROOT,
  CACHE_ROOT, RUN_TAG
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --root)
      ROOT="$2"
      OUT_ROOT="${ROOT}/output/runs"
      CACHE_ROOT="${ROOT}/ssd_cache"
      if [[ "${SCHED_CACHE_USER_SET}" != "1" ]]; then
        SCHED_CACHE="${ROOT}/schedule_cache/oneb_bigcity"
      fi
      shift 2
      ;;
    --src) SRC="$2"; shift 2 ;;
    --ply) PLY="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --schedule-cache) SCHED_CACHE="$2"; SCHED_CACHE_USER_SET=1; shift 2 ;;
    --decode-dataset-path) DECODE_DATASET_PATH="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --debug-max-train-cameras) DEBUG_MAX_TRAIN_CAMERAS="$2"; shift 2 ;;
    --debug-camera-sample-mode) DEBUG_CAMERA_SAMPLE_MODE="$2"; shift 2 ;;
    --debug-camera-sample-start) DEBUG_CAMERA_SAMPLE_START="$2"; shift 2 ;;
    --bsz) BSZ_LIST="$2"; shift 2 ;;
    --bsz-list) BSZ_LIST="$2"; shift 2 ;;
    --capacity) CAPACITY_LIST="$2"; shift 2 ;;
    --capacity-list) CAPACITY_LIST="$2"; shift 2 ;;
    --projection-chunk) PROJECTION_CHUNK="$2"; shift 2 ;;
    --max-ram-gb) MAX_RAM_GB="$2"; shift 2 ;;
    --checkpoint-mode) CHECKPOINT_MODE="$2"; shift 2 ;;
    --resident-policy) RESIDENT_POLICY="$2"; shift 2 ;;
    --resident-lambda) RESIDENT_LAMBDA_LIST="$2"; shift 2 ;;
    --resident-lambda-list) RESIDENT_LAMBDA_LIST="$2"; shift 2 ;;
    --resident-decay) RESIDENT_DECAY_LIST="$2"; shift 2 ;;
    --resident-decay-list) RESIDENT_DECAY_LIST="$2"; shift 2 ;;
    --balanced-seed-fraction) BALANCED_SEED_FRACTION_LIST="$2"; shift 2 ;;
    --balanced-seed-fraction-list) BALANCED_SEED_FRACTION_LIST="$2"; shift 2 ;;
    --checkpoint-iter) CHECKPOINT_ITER="$2"; shift 2 ;;
    --resume-to-iter) RESUME_TO_ITER="$2"; shift 2 ;;
    --start-checkpoint) START_CHECKPOINT="$2"; shift 2 ;;
    --debug-logging) DEBUG_LOGGING=1; shift ;;
    --verbose-terminal) VERBOSE_TERMINAL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

case "${MODE}" in
  train|checkpoint|resume|summary) ;;
  *) echo "Invalid --mode '${MODE}'" >&2; usage >&2; exit 1 ;;
esac

if [[ "${MODE}" != "summary" ]]; then
  if [[ -z "${SRC}" ]]; then
    echo "Missing dataset path. Set SRC, MATRIXCITY_SCENE_DIR, or pass --src." >&2
    exit 1
  fi
  if [[ -z "${PLY}" ]]; then
    echo "Missing dense point cloud. Set PLY, TIDEGS_DENSE_PLY, or pass --ply." >&2
    exit 1
  fi
fi
case "${CHECKPOINT_MODE}" in
  incremental|snapshot) ;;
  *) echo "Invalid --checkpoint-mode '${CHECKPOINT_MODE}'" >&2; usage >&2; exit 1 ;;
esac
case "${RESIDENT_POLICY}" in
  topc_strict|topc_balanced) ;;
  *) echo "Invalid --resident-policy '${RESIDENT_POLICY}'" >&2; usage >&2; exit 1 ;;
esac
case "${DEBUG_CAMERA_SAMPLE_MODE}" in
  linspace|contiguous|window) ;;
  *) echo "Invalid --debug-camera-sample-mode '${DEBUG_CAMERA_SAMPLE_MODE}'" >&2; usage >&2; exit 1 ;;
esac

RUN_ROOT="${OUT_ROOT}/${RUN_TAG}"
COMMANDS="${RUN_ROOT}/commands.sh"
PLAN="${RUN_ROOT}/planned_runs.tsv"
SUMMARY="${RUN_ROOT}/summary.tsv"
RECOMMEND="${RUN_ROOT}/summary_recommended.tsv"
mkdir -p "${RUN_ROOT}"

cat > "${COMMANDS}" <<'HEADER'
#!/usr/bin/env bash
set -euo pipefail
HEADER
chmod +x "${COMMANDS}"

cat > "${PLAN}" <<'HEADER'
mode	run_name	bsz	capacity	resident_lambda	resident_decay	balanced_seed_fraction	iterations	checkpoint_iter	model_path	ssd_cache_dir	start_checkpoint
HEADER

append_train_command() {
  local mode="$1"
  local run_name="$2"
  local bsz="$3"
  local capacity="$4"
  local resident_lambda="$5"
  local resident_decay="$6"
  local balanced_seed_fraction="$7"
  local iterations="$8"
  local checkpoint_iter="$9"
  local start_checkpoint="${10}"
  local model_path="${RUN_ROOT}/${run_name}"
  local cache_dir="${CACHE_ROOT}/${RUN_TAG}/${run_name}"
  local checkpoint_args=()
  local resume_args=()
  local prebuilt_args=()

  if [[ -n "${checkpoint_iter}" ]]; then
    checkpoint_args=(--checkpoint_iterations "${checkpoint_iter}")
  fi
  if [[ -n "${start_checkpoint}" ]]; then
    resume_args=(--start_checkpoint "${start_checkpoint}")
  elif [[ -n "${MANIFEST}" ]]; then
    prebuilt_args=(--pure_ssd_prebuilt_manifest "${MANIFEST}")
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${mode}" "${run_name}" "${bsz}" "${capacity}" "${resident_lambda}" "${resident_decay}" "${balanced_seed_fraction}" "${iterations}" "${checkpoint_iter}" \
    "${model_path}" "${cache_dir}" "${start_checkpoint}" >> "${PLAN}"

  {
    printf '\n# %s\n' "${run_name}"
    printf 'PYTHONDONTWRITEBYTECODE=1 \\\n'
    printf 'PYTHONWARNINGS=%q \\\n' "ignore:TORCH_CUDA_ARCH_LIST is not set:UserWarning"
    printf 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\\n'
    printf 'CUDA_VISIBLE_DEVICES=%q %q \\\n' "${GPU}" "${PYTHON_BIN}"
    printf '  %q \\\n' "${TRAIN_ENTRY}"
    printf '  -s %q \\\n' "${SRC}"
    printf '  --model_path %q \\\n' "${model_path}"
    printf '  --iterations %q \\\n' "${iterations}"
    for arg in "${checkpoint_args[@]}"; do printf '  %q \\\n' "${arg}"; done
    for arg in "${resume_args[@]}"; do printf '  %q \\\n' "${arg}"; done
    printf '  --dense_ply_file %q \\\n' "${PLY}"
    if [[ -n "${DECODE_DATASET_PATH}" ]]; then
      printf '  --decode_dataset_path %q \\\n' "${DECODE_DATASET_PATH}"
    fi
    printf '  --bsz %q \\\n' "${bsz}"
    printf '  --debug_max_train_cameras %q \\\n' "${DEBUG_MAX_TRAIN_CAMERAS}"
    printf '  --debug_camera_sample_mode %q \\\n' "${DEBUG_CAMERA_SAMPLE_MODE}"
    printf '  --debug_camera_sample_start %q \\\n' "${DEBUG_CAMERA_SAMPLE_START}"
    printf '  --disable_auto_densification \\\n'
    printf '  --sparse_adam \\\n'
    printf '  --enable_timer \\\n'
    printf '  --check_gpu_memory \\\n'
    printf '  --check_cpu_memory \\\n'
    printf '  --initial_point_cloud_downsampled_ratio 1.0 \\\n'
    printf '  --use_ssd_offload \\\n'
    printf '  --pure_ssd_offload \\\n'
    printf '  --pure_ssd_init_backend streaming \\\n'
    for arg in "${prebuilt_args[@]}"; do printf '  %q \\\n' "${arg}"; done
    printf '  --use_6plane \\\n'
    printf '  --ssd_cache_dir %q \\\n' "${cache_dir}"
    printf '  --gaussian_block_size 4096 \\\n'
    printf '  --max_ram_gb %q \\\n' "${MAX_RAM_GB}"
    printf '  --num_clusters %q \\\n' "${NUM_CLUSTERS}"
    printf '  --tide_optimizer_backend gpu_resident \\\n'
    printf '  --tide_block_reader_backend tiered_cache \\\n'
    printf '  --tide_optimizer_deferred_mode off \\\n'
    printf '  --tide_resident_selection_policy %q \\\n' "${RESIDENT_POLICY}"
    printf '  --tide_resident_lambda %q \\\n' "${resident_lambda}"
    printf '  --tide_resident_recency_decay %q \\\n' "${resident_decay}"
    printf '  --tide_balanced_seed_fraction %q \\\n' "${balanced_seed_fraction}"
    printf '  --tide_resident_capacity_blocks %q \\\n' "${capacity}"
    printf '  --tide_optimizer_state_mode resident_blocks \\\n'
    printf '  --projection_max_cameras_per_chunk %q \\\n' "${PROJECTION_CHUNK}"
    printf '  --pure_ssd_checkpoint_mode %q \\\n' "${CHECKPOINT_MODE}"
    if [[ "${DEBUG_LOGGING}" == "1" ]]; then
      printf '  --tide_debug_logging \\\n'
    fi
    if [[ "${VERBOSE_TERMINAL}" != "1" ]]; then
      printf '  --quiet \\\n'
    fi
    printf '  --tide_free_unified_params \\\n'
    printf '  --pure_ssd_schedule_cache_dir %q\n' "${SCHED_CACHE}"
  } >> "${COMMANDS}"
}

if [[ "${MODE}" == "train" ]]; then
  for bsz in ${BSZ_LIST}; do
    for capacity in ${CAPACITY_LIST}; do
      for resident_lambda in ${RESIDENT_LAMBDA_LIST}; do
        for resident_decay in ${RESIDENT_DECAY_LIST}; do
          for balanced_seed_fraction in ${BALANCED_SEED_FRACTION_LIST}; do
            fraction_tag="${balanced_seed_fraction//./p}"
            lambda_tag="${resident_lambda//./p}"
            decay_tag="${resident_decay//./p}"
            append_train_command \
              "train" \
              "train_bsz${bsz}_cap${capacity}_lam${lambda_tag}_decay${decay_tag}_seed${fraction_tag}_iter${ITERATIONS}" \
              "${bsz}" "${capacity}" "${resident_lambda}" "${resident_decay}" "${balanced_seed_fraction}" "${ITERATIONS}" "" ""
          done
        done
      done
    done
  done
elif [[ "${MODE}" == "checkpoint" ]]; then
  bsz="$(awk '{print $1}' <<< "${BSZ_LIST}")"
  capacity="$(awk '{print $1}' <<< "${CAPACITY_LIST}")"
  resident_lambda="$(awk '{print $1}' <<< "${RESIDENT_LAMBDA_LIST}")"
  resident_decay="$(awk '{print $1}' <<< "${RESIDENT_DECAY_LIST}")"
  balanced_seed_fraction="$(awk '{print $1}' <<< "${BALANCED_SEED_FRACTION_LIST}")"
  fraction_tag="${balanced_seed_fraction//./p}"
  lambda_tag="${resident_lambda//./p}"
  decay_tag="${resident_decay//./p}"
  append_train_command \
    "checkpoint" \
    "ckpt_bsz${bsz}_cap${capacity}_lam${lambda_tag}_decay${decay_tag}_seed${fraction_tag}_iter1000_ckpt${CHECKPOINT_ITER}" \
    "${bsz}" "${capacity}" "${resident_lambda}" "${resident_decay}" "${balanced_seed_fraction}" "1000" "${CHECKPOINT_ITER}" ""
elif [[ "${MODE}" == "resume" ]]; then
  if [[ -z "${START_CHECKPOINT}" ]]; then
    echo "--mode resume requires --start-checkpoint" >&2
    exit 1
  fi
  bsz="$(awk '{print $1}' <<< "${BSZ_LIST}")"
  capacity="$(awk '{print $1}' <<< "${CAPACITY_LIST}")"
  resident_lambda="$(awk '{print $1}' <<< "${RESIDENT_LAMBDA_LIST}")"
  resident_decay="$(awk '{print $1}' <<< "${RESIDENT_DECAY_LIST}")"
  balanced_seed_fraction="$(awk '{print $1}' <<< "${BALANCED_SEED_FRACTION_LIST}")"
  fraction_tag="${balanced_seed_fraction//./p}"
  lambda_tag="${resident_lambda//./p}"
  decay_tag="${resident_decay//./p}"
  append_train_command \
    "resume" \
    "resume_bsz${bsz}_cap${capacity}_lam${lambda_tag}_decay${decay_tag}_seed${fraction_tag}_to${RESUME_TO_ITER}" \
    "${bsz}" "${capacity}" "${resident_lambda}" "${resident_decay}" "${balanced_seed_fraction}" "${RESUME_TO_ITER}" "" "${START_CHECKPOINT}"
fi

if [[ "${MODE}" == "summary" ]]; then
  "${PYTHON_BIN}" tools/summarize_pure_ssd_pipeline.py "${RUN_ROOT}" \
    --output "${SUMMARY}" \
    --recommend-output "${RECOMMEND}"
  echo "[TideGS] summary: ${SUMMARY}"
  exit 0
fi

echo "[TideGS] Output: ${RUN_ROOT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[TideGS] Dry run complete."
  exit 0
fi

bash "${COMMANDS}"
