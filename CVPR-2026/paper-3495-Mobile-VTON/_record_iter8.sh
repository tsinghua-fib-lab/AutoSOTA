#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     8 \
  --idea-id  "IDEA-010" \
  --title    "Improved Prompt Engineering" \
  --status   failed \
  --primary  0.8567 \
  --metrics  '{}' \
  --notes    "Better prompts improved SSIM+LPIPS but hurt CLIP-I (0.8524). Text prompt distribution shift affects CLIP embedding alignment." \
  --is-best  false
echo "EXIT: $?" > /repo/_record_r8.txt
