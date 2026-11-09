#!/usr/bin/env bash
set -euo pipefail
cd /repo
SE=$(python3 -c "import json;print(json.load(open('outputs/iter4/metrics.json'))['F1_fold_SE'])")
NOTES="FUSE-04: smooth each tempered component's posterior separately, then pool (instead of pool-then-smooth). Ran the unmodified official eval command with --smooth-components --smooth-k 160 --smooth-alpha 0.65. F1_fold_mean 0.7956694 -> 0.7980419 (+0.00237) and F1_overall 0.7922172 -> 0.7973823; guardrails unchanged (LGB 0.7623462, CNN 0.7254360, npz sha256 36291c39...). NOT ADOPTED, deliberately: (1) positive on only 3 of 5 folds vs iter3 (deltas [+0.00613, +0.01129, +0.00474, -0.00276, -0.00753]) and also only 3 of 5 vs the round-2 baseline, so it fails the objective success policy's >=4/5-fold rule; (2) decisive LOO referee over the union of both smoothing families (mode x k in {80,120,160,200} x alpha in {0.60,0.65,0.70,0.75}, 32 candidates) returns 0.7927, BELOW the incumbent 0.7957 -- an honest selector re-picking these parameters loses 0.003, so the +0.0024 in-sample gain is selection noise. The variant is kept behind a flag that is inert by default; the default path was re-verified to reproduce iter3 exactly (F1=0.7957). Recorded because the evaluation command completed with trustworthy metrics."
bash /tools/record_score.sh \
  --scores /autosota_artifacts/paper-9002/sota/scores.jsonl \
  --iter 4 \
  --idea-id "FUSE-04" \
  --title "Component-wise smoothing then pool (k=160, alpha=0.65) - evaluated, NOT adopted" \
  --status success \
  --primary 0.7980418903774515 \
  --metrics "{\"F1\": 0.7980418903774515, \"F1_overall\": 0.7973822583873442, \"F1_SE\": ${SE}, \"LGB_tabular_F1\": 0.7623462038814838, \"CNN_ensemble_F1\": 0.7254360012304671}" \
  --notes "$NOTES"
