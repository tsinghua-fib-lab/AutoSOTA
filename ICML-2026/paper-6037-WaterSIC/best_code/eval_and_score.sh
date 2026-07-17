#!/bin/bash
set -e
RUN_DIR="$1"; ITER="$2"; IDEA_ID="$3"; TITLE="$4"; NOTES="${5:-}"
export QUANT_BUCKET=/autosota_cache HF_ENDPOINT=https://hf-mirror.com HF_HOME=/autosota_cache/hf CUDA_VISIBLE_DEVICES=0
unset ALL_PROXY all_proxy
cd /repo
echo "=== Evaluating $RUN_DIR ==="
python -m scripts.run_eval_job --run_dir "$RUN_DIR" --seqlen 2048 --ppl_only --init_dist 2>&1
PPL=$(python3 -c "import json; d=json.load(open('$RUN_DIR/eval.json')); print(d['eval']['ppl_quant'])")
echo "PPL=$PPL"
/tools/record_score.sh --scores /autosota_artifacts/paper-6037/sota/scores.jsonl --iter "$ITER" --idea-id "$IDEA_ID" --title "$TITLE" --status success --primary "$PPL" --metrics "{\"PPL\": $PPL}" --notes "$NOTES"
echo "Score recorded iter=$ITER"
