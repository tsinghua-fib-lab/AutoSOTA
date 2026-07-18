#!/usr/bin/env bash
# Run a complete evaluation iteration (all 3 models) and record score
# Usage: run_iteration.sh ITER IDEA_ID TITLE [extra_args for run_single_model.py]

set -euo pipefail

ITER="$1"
IDEA_ID="$2"
TITLE="$3"
shift 3 || true
EXTRA_ARGS="$@"

MODELS=("FEDformer" "SimpleTM" "TimesNet")
RESULTS_CLN=()
RESULTS_ATK=()
FAILED_MODELS=()

for MODEL in "${MODELS[@]}"; do
    echo "=== Iter $ITER: $MODEL ==="
    python3 /repo/run_single_model.py --model "$MODEL" $EXTRA_ARGS 2>&1 | tee "/tmp/iter_${ITER}_${MODEL}.log"

    # Parse result
    RESULT_LINE=$(grep "^RESULT:" "/tmp/iter_${ITER}_${MODEL}.log" | tail -1)
    STATUS=$(echo "$RESULT_LINE" | python3 -c "import sys,json; d=json.loads(sys.stdin.read().split('RESULT: ',1)[1]); print(d.get('status','error'))" 2>/dev/null || echo "error")

    if [ "$STATUS" = "success" ]; then
        CLN=$(echo "$RESULT_LINE" | python3 -c "import sys,json; d=json.loads(sys.stdin.read().split('RESULT: ',1)[1]); print(d['cln_mae'])" 2>/dev/null || echo "0")
        ATK=$(echo "$RESULT_LINE" | python3 -c "import sys,json; d=json.loads(sys.stdin.read().split('RESULT: ',1)[1]); print(d['atk_mae'])" 2>/dev/null || echo "0")
        RESULTS_CLN+=("$CLN")
        RESULTS_ATK+=("$ATK")
        echo "  $MODEL: MAEc=$CLN, MAEp=$ATK"
    else
        FAILED_MODELS+=("$MODEL")
        echo "  $MODEL: FAILED ($STATUS)"
    fi
done

# Compute averages if all models succeeded
if [ ${#RESULTS_CLN[@]} -eq 3 ]; then
    AVG_CLN=$(python3 -c "print((${RESULTS_CLN[0]}+${RESULTS_CLN[1]}+${RESULTS_CLN[2]})/3)")
    AVG_ATK=$(python3 -c "print((${RESULTS_ATK[0]}+${RESULTS_ATK[1]}+${RESULTS_ATK[2]})/3)")

    # FDER computation
    FDER=$(python3 -c "
maec_undef=17.607; maep_undef=14.201
avg_cln=$AVG_CLN; avg_atk=$AVG_ATK
rho_maep=max(0,1-maep_undef/avg_atk)
rho_maec=max(0,1-maec_undef/avg_cln)
fder=(rho_maep-rho_maec+1)/2
print(f'{fder:.4f}')
")

    echo "=== Iter $ITER Complete ==="
    echo "  MAEc=$AVG_CLN, MAEp=$AVG_ATK, FDER=$FDER"

    # Record score
    METRICS=$(python3 -c "
import json
print(json.dumps({'MAEc': float($AVG_CLN), 'MAEp': float($AVG_ATK), 'FDER': float($FDER)}))
")

    /tools/record_score.sh \
        --scores /autosota_artifacts/paper-1920/sota/scores.jsonl \
        --iter "$ITER" \
        --idea-id "$IDEA_ID" \
        --title "$TITLE" \
        --status success \
        --primary "$AVG_CLN" \
        --metrics "$METRICS" \
        --notes "t2 override args: $EXTRA_ARGS | Individual: F=${RESULTS_CLN[0]}/${RESULTS_ATK[0]}, S=${RESULTS_CLN[1]}/${RESULTS_ATK[1]}, T=${RESULTS_CLN[2]}/${RESULTS_ATK[2]}"

    echo "Score recorded."
else
    echo "=== Iter $ITER FAILED (${#RESULTS_CLN[@]}/3 models) ==="
    FAILED_STR=$(IFS=,; echo "${FAILED_MODELS[*]}")

    /tools/record_score.sh \
        --scores /autosota_artifacts/paper-1920/sota/scores.jsonl \
        --iter "$ITER" \
        --idea-id "$IDEA_ID" \
        --title "$TITLE" \
        --status failed \
        --primary 0.0 \
        --metrics '{}' \
        --notes "Only ${#RESULTS_CLN[@]}/3 models completed. Failed: $FAILED_STR. Args: $EXTRA_ARGS"

    echo "Failed score recorded."
fi
