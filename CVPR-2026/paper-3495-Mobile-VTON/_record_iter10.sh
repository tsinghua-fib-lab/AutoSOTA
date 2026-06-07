#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     10 \
  --idea-id  "IDEA-006" \
  --title    "Multi-Scale Garment Feature Weighting" \
  --status   success \
  --primary  0.8585 \
  --metrics  '{"clip_i": 0.8585, "ssim": 0.8903, "lpips": 0.0893}' \
  --notes    "Per-scale timestep-dependent weighting: fine scales get higher weight late in denoising. CLIP-I +0.21%, SSIM maintained." \
  --is-best  true
echo "EXIT: $?" > /repo/_record_r10.txt
