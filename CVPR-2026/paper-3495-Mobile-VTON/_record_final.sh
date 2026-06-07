#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     final \
  --idea-id  "final" \
  --title    "Final best state" \
  --status   success \
  --primary  0.8783 \
  --metrics  '{"clip_i": 0.8783, "ssim": 0.9098, "lpips": 0.0763}' \
  --notes    "Final evaluation after restoring _best. Target exceeded: CLIP-I 0.8783 vs 0.877 target." \
  --is-best  false
echo "EXIT: $?" > /repo/_record_final.txt
