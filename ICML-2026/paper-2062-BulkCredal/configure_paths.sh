#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------
# configure_paths.sh
#
# Creates the directory structure under two root paths and (re)generates
# paths.details.sh with the corresponding exports.
#
# Usage:
#   ./configure_paths.sh <aggregated_info_root> <data_and_details_root>
#
# Or:
#   ./configure_paths.sh \
#     --aggregated-info-root /path/to/agg_root \
#     --data-and-details-root /path/to/data_root \
#     --output /path/to/paths.details.sh
#
# Notes:
# - data_and_details_root should have >= 50 GiB available space (checked).
# - The script is idempotent (safe to run multiple times).
# - If paths.details.sh already exists at the output path, it will be backed up.
# ---------------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage:
  ./configure_paths.sh <aggregated_info_root> <data_and_details_root>
  ./configure_paths.sh --aggregated-info-root <path> --data-and-details-root <path> [--output <path>]

Arguments:
  <aggregated_info_root>   Root for aggregated info (configs, plots, CSV summaries)
  <data_and_details_root>  Root for run data, datasets, caches (>= 50 GiB free recommended/required)

Options:
  -a, --aggregated-info-root   Same as first positional argument
  -d, --data-and-details-root  Same as second positional argument
  -o, --output                 Where to write paths.details.sh (default: alongside this script)
  -h, --help                   Show this help
EOF
}

abs_path() {
  # Resolves to an absolute path if possible; expands "~".
  local p="$1"
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$p" <<'PY'
import os, sys
p = sys.argv[1]
p = os.path.expanduser(p)
p = os.path.abspath(p)
print(p)
PY
  elif command -v python >/dev/null 2>&1; then
    python - "$p" <<'PY'
import os, sys
p = sys.argv[1]
p = os.path.expanduser(p)
p = os.path.abspath(p)
print(p)
PY
  elif command -v realpath >/dev/null 2>&1; then
    # -m allows non-existent components on GNU coreutils; if unsupported, it will just fail and fall back.
    realpath -m "$p" 2>/dev/null || realpath "$p"
  else
    # Best-effort fallback (may remain relative).
    echo "$p"
  fi
}

human_gib() {
  # Convert KiB to GiB with 2 decimals (best-effort).
  local kib="$1"
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$kib" <<'PY'
import sys
kib = int(sys.argv[1])
gib = kib / (1024**2)
print(f"{gib:.2f} GiB")
PY
  else
    # shell integer arithmetic only
    local gib_int=$(( kib / (1024 * 1024) ))
    echo "${gib_int} GiB"
  fi
}

AGG_ROOT=""
DATA_ROOT=""
OUT_FILE=""

# Parse args (supports both flags and 2 positional args).
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--aggregated-info-root)
      [[ $# -ge 2 ]] || { echo "ERROR: Missing value for $1" >&2; usage; exit 2; }
      AGG_ROOT="$2"
      shift 2
      ;;
    -d|--data-and-details-root)
      [[ $# -ge 2 ]] || { echo "ERROR: Missing value for $1" >&2; usage; exit 2; }
      DATA_ROOT="$2"
      shift 2
      ;;
    -o|--output)
      [[ $# -ge 2 ]] || { echo "ERROR: Missing value for $1" >&2; usage; exit 2; }
      OUT_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$AGG_ROOT" ]]; then
        AGG_ROOT="$1"
        shift
      elif [[ -z "$DATA_ROOT" ]]; then
        DATA_ROOT="$1"
        shift
      else
        echo "ERROR: Unexpected argument: $1" >&2
        usage
        exit 2
      fi
      ;;
  esac
done

if [[ -z "$AGG_ROOT" || -z "$DATA_ROOT" ]]; then
  echo "ERROR: You must provide <aggregated_info_root> and <data_and_details_root>." >&2
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "$OUT_FILE" ]]; then
  OUT_FILE="$SCRIPT_DIR/paths.details.sh"
fi

AGG_ROOT="$(abs_path "$AGG_ROOT")"
DATA_ROOT="$(abs_path "$DATA_ROOT")"
OUT_FILE="$(abs_path "$OUT_FILE")"

# Ensure roots exist.
mkdir -p "$AGG_ROOT" "$DATA_ROOT"

# Check available space under DATA_ROOT filesystem: require >= 50 GiB free.
MIN_AVAIL_KIB=$((50 * 1024 * 1024))  # 50 GiB in KiB
AVAIL_KIB="$(df -Pk "$DATA_ROOT" | awk 'NR==2 {print $4}')"
if [[ -z "${AVAIL_KIB:-}" ]]; then
  echo "ERROR: Could not determine free space for: $DATA_ROOT" >&2
  exit 3
fi
if (( AVAIL_KIB < MIN_AVAIL_KIB )); then
  echo "WARNING: data_and_details_root has less than 50.00 GiB free. End-to-end reproduction of all results may require more space." >&2
  echo "  Path:        $DATA_ROOT" >&2
  echo "  Free:        $(human_gib "$AVAIL_KIB")" >&2
  echo "  Recommended: 50.00 GiB" >&2
fi

# ---------------------------------------------------------------------
# Create directory structure
# ---------------------------------------------------------------------

# Aggregated info dirs
NEWSVENDOR_AGG_INFO_DIR="$AGG_ROOT/Newsvendor"
CALIFORNIA_HOUSING_AGG_INFO_DIR="$AGG_ROOT/California_housing"
CIVILCOMMENTS_AGG_INFO_DIR="$AGG_ROOT/Civilcomments"

# Data & details dirs
NEWSVENDOR_RUN_DATA_DIR="$DATA_ROOT/Newsvendor_large"

CALIFORNIA_HOUSING_BASE="$DATA_ROOT/California_housing_large"
CALIFORNIA_HOUSING_RUN_DATA_DIR="$CALIFORNIA_HOUSING_BASE/Run_data"
CALIFORNIA_HOUSING_DATASET_DIR="$CALIFORNIA_HOUSING_BASE/Dataset"

CIVILCOMMENTS_BASE="$DATA_ROOT/Civilcomments_large"
CIVILCOMMENTS_RUN_DATA_DIR="$CIVILCOMMENTS_BASE/Run_data"
CIVILCOMMENTS_DATASET_DIR="$CIVILCOMMENTS_BASE/Dataset"
CIVILCOMMENTS_CACHE_DIR="$CIVILCOMMENTS_BASE/Cache"

mkdir -p \
  "$NEWSVENDOR_AGG_INFO_DIR" \
  "$CALIFORNIA_HOUSING_AGG_INFO_DIR" \
  "$CIVILCOMMENTS_AGG_INFO_DIR" \
  "$NEWSVENDOR_RUN_DATA_DIR" \
  "$CALIFORNIA_HOUSING_RUN_DATA_DIR" \
  "$CALIFORNIA_HOUSING_DATASET_DIR" \
  "$CIVILCOMMENTS_RUN_DATA_DIR" \
  "$CIVILCOMMENTS_DATASET_DIR" \
  "$CIVILCOMMENTS_CACHE_DIR"

cat > "$OUT_FILE" <<EOF
#!/usr/bin/env bash

set -euo pipefail

# ---------------------------------------------------------------------
# AUTO-GENERATED by $(basename "$0") on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
#
# Roots used:
#   aggregated_info_root   = $AGG_ROOT
#   data_and_details_root  = $DATA_ROOT
#
# If you want to change locations, re-run:
#   ./$(basename "$0") "<new_aggregated_info_root>" "<new_data_and_details_root>"
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Newsvendor Experiment Paths
# ---------------------------------------------------------------------
# Where the aggreated info experiment.json + real_world_experiment.slurm + plots/csv are written
export NEWSVENDOR_AGG_INFO_DIR="$NEWSVENDOR_AGG_INFO_DIR"

# Where per-UUID CSVs and results.csv will be written. These files can be large (a few GB for each experiment).
export NEWSVENDOR_RUN_DATA_DIR="$NEWSVENDOR_RUN_DATA_DIR"

# ---------------------------------------------------------------------
# California Housing Experiment Paths
# ---------------------------------------------------------------------
# Where the aggreated info experiment.json + real_world_experiment.slurm + plots/csv are written
export CALIFORNIA_HOUSING_AGG_INFO_DIR="$CALIFORNIA_HOUSING_AGG_INFO_DIR"

# Where per-UUID CSVs and results.csv will be written. These files can be large (a few GB for each experiment).
export CALIFORNIA_HOUSING_RUN_DATA_DIR="$CALIFORNIA_HOUSING_RUN_DATA_DIR"

# REQUIRED: Where the California Housing dataset is stored
export CALIFORNIA_HOUSING_DATASET_DIR="$CALIFORNIA_HOUSING_DATASET_DIR"

# ---------------------------------------------------------------------
# Civilcomments Experiment Paths
# ---------------------------------------------------------------------
# Where the aggreated info experiment.json + real_world_experiment.slurm + plots/csv are written
export CIVILCOMMENTS_AGG_INFO_DIR="$CIVILCOMMENTS_AGG_INFO_DIR"

# Where per-UUID CSVs and results.csv will be written. These files can be large (a few GB).
export CIVILCOMMENTS_RUN_DATA_DIR="$CIVILCOMMENTS_RUN_DATA_DIR"

# REQUIRED: the Civilcomments dataset is stored in "\$CIVILCOMMENTS_DATASET_DIR/wilds/"
export CIVILCOMMENTS_DATASET_DIR="$CIVILCOMMENTS_DATASET_DIR"

# REQUIRED: derived cache root. The cache can be large (~10 GB).
export CIVILCOMMENTS_CACHE_DIR="$CIVILCOMMENTS_CACHE_DIR"
EOF

chmod 0644 "$OUT_FILE"

echo
echo "Done."
echo "Created directories under:"
echo "  aggregated_info_root:  $AGG_ROOT"
echo "  data_and_details_root: $DATA_ROOT"
echo
echo "Wrote: $OUT_FILE"
echo
echo "Next step (in your shell):"
echo "  source \"$OUT_FILE\""