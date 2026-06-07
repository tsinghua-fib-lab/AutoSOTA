#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     17 \
  --idea-id  "IDEA-014e" \
  --title    "Guidance Decay (1.5->1.0) + 16 Steps + TTA + Multi-Scale" \
  --status   success \
  --primary  0.8783 \
  --metrics  '{"clip_i": 0.8783, "ssim": 0.9098, "lpips": 0.0763}' \
  --notes    "TARGET EXCEEDED! CLIP-I 0.8783 (+5.16%), SSIM 0.9098, LPIPS 0.0763. Breakthrough formula: weak guidance + 16 steps + TTA + multi-scale weighting." \
  --is-best  true
echo "EXIT: $?" > /repo/_record_r17.txt
