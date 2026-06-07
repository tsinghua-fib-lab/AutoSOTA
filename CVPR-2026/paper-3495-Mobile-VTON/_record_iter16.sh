#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     16 \
  --idea-id  "IDEA-014d" \
  --title    "Guidance Decay (2.0->1.0) + 16 Steps" \
  --status   success \
  --primary  0.8736 \
  --metrics  '{"clip_i": 0.8736, "ssim": 0.9056, "lpips": 0.0801}' \
  --notes    "CLIP-I +0.25%, SSIM 0.9056. Weaker guidance continues to improve all metrics. Just 0.39% from target." \
  --is-best  true
echo "EXIT: $?" > /repo/_record_r16.txt
