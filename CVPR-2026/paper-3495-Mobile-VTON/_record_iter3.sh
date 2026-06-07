#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     3 \
  --idea-id  "IDEA-008" \
  --title    "TTA Horizontal Flip Ensemble" \
  --status   success \
  --primary  0.8566 \
  --metrics  '{"clip_i": 0.8566, "ssim": 0.8903, "lpips": 0.0892}' \
  --notes    "TTA flip ensemble: +1.81% CLIP-I, +1.26% SSIM, -0.3% LPIPS. All metrics improved significantly." \
  --is-best  true
echo "EXIT: $?" > /repo/_record_r3.txt
