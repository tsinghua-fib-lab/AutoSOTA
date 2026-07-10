#!/usr/bin/env bash



ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:1}"
RESULTS_ROOT="${RESULTS_ROOT:-results/paper/vision_L14}"
CACHE_PATH="${CACHE_PATH:-src/.cache/single_task_acc.json}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_REBASE="${RUN_REBASE:-1}"
DRY_RUN="${DRY_RUN:-0}"
SAVE_MERGED_CHECKPOINTS="${SAVE_MERGED_CHECKPOINTS:-${SAVE_MERGED:-0}}"
SAVE_MERGED_NAME="${SAVE_MERGED_NAME:-merged.pt}"

read -r -a SUITES <<< "${SUITES:-vision8 vision14 vision20}"
read -r -a MERGE_METHODS <<< "${MERGE_METHODS:-weighted_average task_arithmetic ties_merge dare_merge tsv_merge isoc_merge isocts_merge cart_merge pcb}"
read -r -a REBASE_METHODS <<< "${REBASE_METHODS:-identity orthogonal_shift gradfix theseus}"

mkdir -p "$RESULTS_ROOT"
mkdir -p "$(dirname "$CACHE_PATH")"

log() {
  printf '[vision-sweep] %s\n' "$*"
}

run_cmd() {
  log "$*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

is_run_complete() {
  local out_dir="$1"
  local summary_path="$out_dir/summary.json"

  [[ -f "$summary_path" ]] || return 1

  "$PYTHON_BIN" - "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)

status = ((payload.get("run_logging") or {}).get("status") or "").strip().lower()
raise SystemExit(0 if status == "success" else 1)
PY
}

merge_config_for_suite() {
  case "$1" in
    vision8) echo "configs/vision8_task_arithmetic_vitl.json" ;;
    vision14) echo "configs/vision14_task_arithmetic_vitl.json" ;;
    vision20) echo "configs/vision20_task_arithmetic_vitl.json" ;;
    *)
      log "Unknown suite '$1'"
      return 1
      ;;
  esac
}

merge_method_params_json() {
  case "$1" in
    ties_merge)
      printf '{"topk":1.0,"merging_type":"mean","low_memory":true,"cache_prepared":false}\n'
      ;;
    dare_merge)
      printf '{"low_memory":true,"cache_prepared":false}\n'
      ;;
    *)
      printf '\n'
      ;;
  esac
}

write_rebase_config() {
  local suite="$1"
  local method="$2"
  local out_path="$3"

  if [[ "$method" == "theseus" ]]; then
    "$PYTHON_BIN" - "configs/vision8_theseus_all_alpha_sweep.json" "$out_path" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
payload = json.loads(src.read_text())
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(payload, indent=2) + "\n")
PY
    return 0
  fi

  local suite_cfg
  suite_cfg="$(merge_config_for_suite "$suite")"

  "$PYTHON_BIN" - "configs/vision8_gradfix_rebase.json" "$suite_cfg" "$method" "$suite" "$out_path" <<'PY'
import json
import sys
from pathlib import Path

template_path = Path(sys.argv[1])
suite_path = Path(sys.argv[2])
method = sys.argv[3]
suite = sys.argv[4]
out_path = Path(sys.argv[5])

template = json.loads(template_path.read_text())
suite_cfg = json.loads(suite_path.read_text())

template["suite"] = suite
template["tasks"] = "all"
template["method"] = method
template["tuned_ckpts"] = suite_cfg["tuned_ckpts"]
template["weights"] = suite_cfg.get("weights")

if method != "gradfix":
    template.pop("mask_mode", None)
    template.pop("vote", None)
    template["method_params"] = {}

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(template, indent=2) + "\n")
PY
}

run_merge_case() {
  local suite="$1"
  local method="$2"
  local out_dir="$RESULTS_ROOT/merge/$suite/$method"
  local cfg
  local method_params_json
  local cmd

  cfg="$(merge_config_for_suite "$suite")"
  method_params_json="$(merge_method_params_json "$method")"
  mkdir -p "$out_dir"

  if is_run_complete "$out_dir"; then
    log "Skipping merge/$suite/$method; summary.json already marked success."
    return 0
  fi

  cmd=(env PYTHONPATH=src "$PYTHON_BIN" -m merge_and_rebase.eval.vision_merge \
    --config "$cfg" \
    --method "$method" \
    --device "$DEVICE" \
    --alpha-search \
    --alpha-min 0.0 \
    --alpha-max 2.0 \
    --alpha-step 0.1 \
    --single-acc-cache "$CACHE_PATH" \
    --local-log-dir "$out_dir" \
    --run-name summary)
  if [[ -n "$method_params_json" ]]; then
    cmd+=(--method-params "$method_params_json")
  fi
  if [[ "$SAVE_MERGED_CHECKPOINTS" == "1" ]]; then
    cmd+=(--save-merged "$out_dir/$SAVE_MERGED_NAME")
  fi
  run_cmd "${cmd[@]}"
}

run_rebase_case() {
  local suite="$1"
  local method="$2"
  local out_dir="$RESULTS_ROOT/rebase/$suite/$method"
  local run_cfg="$out_dir/run_config.json"

  if [[ "$method" == "theseus" && "$suite" != "vision8" ]]; then
    log "Skipping rebase/$suite/$method; repo only has a full-suite Theseus config for vision8."
    return 0
  fi

  mkdir -p "$out_dir"

  if is_run_complete "$out_dir"; then
    log "Skipping rebase/$suite/$method; summary.json already marked success."
    return 0
  fi

  write_rebase_config "$suite" "$method" "$run_cfg"

  if [[ "$method" == "theseus" ]]; then
    run_cmd env PYTHONPATH=src "$PYTHON_BIN" -m merge_and_rebase.eval.vision_rebase \
      --config "$run_cfg" \
      --device "$DEVICE" \
      --alpha-search \
      --alpha-min 0.8 \
      --alpha-max 10.0 \
      --alpha-step 0.2 \
      --alpha-selection per_task \
      --alpha-patience 5 \
      --local-log-dir "$out_dir" \
      --run-name summary
    return 0
  fi

  run_cmd env PYTHONPATH=src "$PYTHON_BIN" -m merge_and_rebase.eval.vision_rebase \
    --config "$run_cfg" \
    --device "$DEVICE" \
    --alpha-search \
    --alpha-min 0.0 \
    --alpha-max 2.0 \
    --alpha-step 0.1 \
    --local-log-dir "$out_dir" \
    --run-name summary
}

main() {
  local suite
  local method

  if [[ "$RUN_MERGE" == "1" ]]; then
    for suite in "${SUITES[@]}"; do
      for method in "${MERGE_METHODS[@]}"; do
        run_merge_case "$suite" "$method"
      done
    done
  fi

  if [[ "$RUN_REBASE" == "1" ]]; then
    for suite in "${SUITES[@]}"; do
      for method in "${REBASE_METHODS[@]}"; do
        run_rebase_case "$suite" "$method"
      done
    done
  fi
}

main "$@"
