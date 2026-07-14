#!/bin/bash
# Run a SOTA evaluation iteration and record the score.
# Usage: bash run_iteration.sh <ITER> <IDEA_ID> <TITLE> [N_RUNS] [--is-best]
set -e
cd /repo

ITER="$1"
IDEA_ID="$2"
TITLE="$3"
N_RUNS="${4:-20}"
IS_BEST_FLAG=""

if [[ "$5" == "--is-best" ]] || [[ "$5" == "true" ]]; then
    IS_BEST_FLAG="--is-best true"
fi

echo "=== Iteration $ITER: $TITLE ($IDEA_ID) ==="
echo "Config:"
python3 apply_config.py show

echo ""
echo "Running evaluation with $N_RUNS samples..."
bash eval_sota.sh "$N_RUNS"

# Parse results
RESULT=$(python3 -c "
import json
with open('eval_results_sota.json', 'r') as f:
    stats = json.load(f)
rel = stats['rel_err_a_pct_mean']
pde = stats['pde_res_mean']
print(f'{rel:.4f}|{pde:.6f}')
")

REL_ERR=$(echo "$RESULT" | cut -d'|' -f1)
PDE_RES=$(echo "$RESULT" | cut -d'|' -f2)

echo ""
echo "Results: Rel.err(a)=${REL_ERR}%, PDE res.=${PDE_RES}"

# Record score
/tools/record_score.sh \
    --scores /autosota_artifacts/paper-5167/sota/scores.jsonl \
    --iter "$ITER" \
    --idea-id "$IDEA_ID" \
    --title "$TITLE" \
    --status success \
    --primary "$REL_ERR" \
    --metrics "{\"Rel. err\": $REL_ERR, \"PDE res.\": $PDE_RES}" \
    --notes "N_RUNS=$N_RUNS, config=$(python3 apply_config.py show 2>&1 | grep -E 'zeta_|beta|lr_' | tr '\n' ' ')" \
    $IS_BEST_FLAG

echo ""
echo "=== Iteration $ITER complete ==="
