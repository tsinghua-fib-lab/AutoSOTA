#!/usr/bin/env bash
# NESP (paper 9003) -- full evaluation entrypoint.
#
# Step 1 regenerates the 2413-row test prediction with the vendored 7th-place
# pipeline; step 2 scores it with the official scorer (Spearman rank correlation
# against the post-competition released test_labels.csv). The metric is the last
# line of stdout:  Spearman=<value>
#
# Usage (inside the container, from anywhere):
#     bash /repo/scripts/eval.sh
#
# Threading (important on this host): LightGBM is pinned to NESP_LGBM_THREADS
# (default 8) in pipeline/run_pipeline.py::get_model, and the process CPU
# affinity is pinned to the same number of CPUs when `taskset` is available.
# Without either, LightGBM falls back to all 128 logical CPUs of a heavily
# contended node and every boosting iteration becomes ~50x slower
# (0.33 s/iter vs 0.007 s/iter), turning a 3-minute run into several hours.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

export NESP_LGBM_THREADS="${NESP_LGBM_THREADS:-8}"

if command -v taskset >/dev/null 2>&1; then
  taskset -c "0-$((NESP_LGBM_THREADS - 1))" python pipeline/run_pipeline.py \
      --paper-data /paper_data --out-dir outputs
else
  python pipeline/run_pipeline.py --paper-data /paper_data --out-dir outputs
fi

python /paper_data/eval/validate_submission.py outputs/submission.csv
python /paper_data/eval/score_test.py outputs/submission.csv
