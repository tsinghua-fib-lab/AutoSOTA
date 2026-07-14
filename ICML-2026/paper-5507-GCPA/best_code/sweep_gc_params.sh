#!/bin/bash
set -e
SWEEP_DIR="/repo/sweep_results"
mkdir -p "$SWEEP_DIR"
ALIGN="/repo/config/alignment.yaml"
RES="/repo/results/9_multiret_m3.json"

COMBOS=(
  "0.04 0.7 tighter_no_lam_change"
  "0.12 0.7 looser_no_lam_change"
  "0.08 0.5 lower_consensus"
  "0.08 0.9 higher_consensus"
  "0.04 0.9 tight_and_high_consensus"
  "0.12 0.5 loose_and_low_consensus"
  "0.16 0.7 very_loose"
  "0.02 0.7 very_tight"
)

for combo in "${COMBOS[@]}"; do
  read -r TAU LAM TAG <<< "$combo"
  echo "=== Sweep: gc_tau=$TAU gc_lam=$LAM ($TAG) ==="
  cp "$ALIGN" "$ALIGN.bak"
  python3 -c "
import yaml
with open(\"$ALIGN\") as f:
    cfg = yaml.safe_load(f)
cfg[\"procrustes\"][\"gc_tau\"] = $TAU
cfg[\"procrustes\"][\"gc_lam\"] = $LAM
with open(\"$ALIGN\", \"w\") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print(f\"Updated gc_tau={cfg[chr(112)+chr(114)+chr(111)+chr(99)+chr(114)+chr(117)+chr(115)+chr(116)+chr(101)+chr(115)][chr(103)+chr(99)+chr(95)+chr(116)+chr(97)+chr(117)]}, gc_lam={cfg[chr(112)+chr(114)+chr(111)+chr(99)+chr(114)+chr(117)+chr(115)+chr(116)+chr(101)+chr(115)][chr(103)+chr(99)+chr(95)+chr(108)+chr(97)+chr(109)]}\")
"
  cd /repo
  unset HF_ENDPOINT
  export DATA_PATH=/autosota_cache/data
  uv run python -m scripts.exps.9_multiret --config-name 9_multiret_m3 2>&1 | tail -30
  if [ -f "$RES" ]; then
    cp "$RES" "$SWEEP_DIR/results_tau${TAU}_lam${LAM}.json"
    echo "Saved: $SWEEP_DIR/results_tau${TAU}_lam${LAM}.json"
  fi
  cp "$ALIGN.bak" "$ALIGN"
  echo "=== Done: gc_tau=$TAU gc_lam=$LAM ==="
  echo ""
done
echo "All sweeps done."
