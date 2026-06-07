#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     12 \
  --idea-id  "IDEA-003b" \
  --title    "16 Inference Steps" \
  --status   success \
  --primary  0.8647 \
  --metrics  '{"clip_i": 0.8647, "ssim": 0.8907, "lpips": 0.0902}' \
  --notes    "16 steps > 20 steps > 28 steps. CLIP-I improves as steps decrease. Trend continues." \
  --is-best  true
echo "EXIT: $?" > /repo/_record_r12.txt
