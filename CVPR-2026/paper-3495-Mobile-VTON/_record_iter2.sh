#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     2 \
  --idea-id  "IDEA-005" \
  --title    "Non-Zero Garment CFG (0.01 baseline)" \
  --status   failed \
  --primary  0.8414 \
  --metrics  '{}' \
  --notes    "Non-zero garment CFG hurt all metrics: CLIP-I 0.8379, SSIM 0.8707, LPIPS 0.0995. Rolled back." \
  --is-best  false
echo "EXIT: $?" > /repo/_record_r2.txt
