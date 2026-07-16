#!/bin/bash
# Record baseline iteration with record_score.sh
set -e

METRICS_FILE="${1:-/autosota_cache/eval_results/numina-cot-ranktuner-Qwen2.5-Math-7B/math_oai_metrics.json}"

source /autosota_cache/paper-4041-venv/bin/activate

PASS1=$(python3 -c "import json; d=json.load(open()); print(d[pass_at_k][pass@1])")
PASS16=$(python3 -c "import json; d=json.load(open()); print(d[pass_at_k][pass@16])")

echo "Parsed metrics: Pass@1=$PASS1, Pass@16=$PASS16"

METRICS_JSON="{\"Pass@1\": $PASS1, \"Pass@16\": $PASS16}"

/tools/record_score.sh \
  --paper_id 4041 \
  --iteration 0 \
  --primary "$PASS16" \
  --metrics "$METRICS_JSON" \
  --commit "$(cd /repo && git rev-parse HEAD)" \
  --status success \
  --notes "Baseline reproduction: RankTuner lr=5e-5, batch=256, 1 epoch, use_liger=False, cosine schedule"
