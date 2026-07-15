#!/usr/bin/env bash
# Runs samply against examples/profile_housing and stores the JSON capture
# under docs/perf/. Pass --max-bins N to exercise the binned path (any
# missing or 0 value means exact, i.e. max_bins=None).
#
# Usage:
#   bash scripts/perf/run_profile.sh                  # exact path
#   bash scripts/perf/run_profile.sh --max-bins 255   # binned path
#   bash scripts/perf/run_profile.sh --n-iter 50 --reps 5 --max-bins 64
set -euo pipefail

cd "$(dirname "$0")/../.."

N_ITER=50
REPS=5
MAX_BINS=0
DATASET="data/housing_full.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-bins)  MAX_BINS="${2:?missing value for --max-bins}"; shift 2 ;;
    --n-iter)    N_ITER="${2:?missing value for --n-iter}";     shift 2 ;;
    --reps)      REPS="${2:?missing value for --reps}";         shift 2 ;;
    --dataset)   DATASET="${2:?missing value for --dataset}";   shift 2 ;;
    -h|--help)
      sed -n '2,9p' "$0"
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if ! command -v samply >/dev/null 2>&1; then
  echo "samply is not installed."
  echo "Install with:  cargo install samply"
  exit 1
fi

echo "Building examples/profile_housing under the 'profiling' profile..."
cargo build --profile profiling --example profile_housing

TS="$(date +%Y%m%d_%H%M%S)"
TAG="exact"
if [[ "$MAX_BINS" != "0" ]]; then TAG="bins${MAX_BINS}"; fi

OUT_DIR="docs/perf"
mkdir -p "$OUT_DIR"
OUT_JSON="${OUT_DIR}/profile_${TAG}_${TS}.json"

echo "Recording samply trace: $OUT_JSON"
samply record -r 4000 \
    --save-only \
    --unstable-presymbolicate \
    -o "$OUT_JSON" -- \
  "target/profiling/examples/profile_housing" "$DATASET" "$N_ITER" "$REPS" "$MAX_BINS"

echo
echo "Saved profile to: $OUT_JSON"
echo "Open in Firefox Profiler:  samply load $OUT_JSON"
echo "Or print the top functions:  python3 scripts/perf/analyze_profile.py $OUT_JSON"
