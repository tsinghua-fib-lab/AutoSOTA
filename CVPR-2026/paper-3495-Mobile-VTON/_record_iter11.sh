#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     11 \
  --idea-id  "IDEA-003" \
  --title    "20 Inference Steps (vs 28)" \
  --status   success \
  --primary  0.8617 \
  --metrics  '{"clip_i": 0.8617, "ssim": 0.8908, "lpips": 0.0896}' \
  --notes    "20 steps > 28 steps: CLIP-I +0.37%. Fewer steps reduce noise accumulation, confirming QoS-Diff sweet spot." \
  --is-best  true
echo "EXIT: $?" > /repo/_record_r11.txt
