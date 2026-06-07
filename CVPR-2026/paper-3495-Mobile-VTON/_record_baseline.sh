#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     0 \
  --idea-id  "baseline" \
  --title    "Paper baseline" \
  --status   success \
  --primary  0.8352 \
  --metrics  '{"clip_i": 0.8352, "ssim": 0.8763, "lpips": 0.0914}' \
  --notes    "Paper-reported baseline. LPIPS measured at 0.0914 (vs 0.1977 reported)." \
  --is-best  true
echo "EXIT: $?" > /repo/_record_result.txt
