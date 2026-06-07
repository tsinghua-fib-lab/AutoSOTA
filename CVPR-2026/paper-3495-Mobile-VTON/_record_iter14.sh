#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     14 \
  --idea-id  "IDEA-003d" \
  --title    "14 Inference Steps" \
  --status   failed \
  --primary  0.8647 \
  --metrics  '{}' \
  --notes    "14 steps worse than 16: CLIP-I 0.8628. Optimum confirmed at 16 steps." \
  --is-best  false
echo "EXIT: $?" > /repo/_record_r14.txt
