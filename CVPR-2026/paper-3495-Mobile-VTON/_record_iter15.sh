#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     15 \
  --idea-id  "IDEA-014c" \
  --title    "Weaker Guidance Decay (2.5->1.0) + 16 steps" \
  --status   success \
  --primary  0.8714 \
  --metrics  '{"clip_i": 0.8714, "ssim": 0.9006, "lpips": 0.0839}' \
  --notes    "BREAKTHROUGH: CLIP-I +0.77%, SSIM breaks 0.90! Weaker guidance is key — lower CFG reduces over-conditioning." \
  --is-best  true
echo "EXIT: $?" > /repo/_record_r15.txt
