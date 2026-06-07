#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     13 \
  --idea-id  "IDEA-003c" \
  --title    "12 Inference Steps" \
  --status   failed \
  --primary  0.8647 \
  --metrics  '{}' \
  --notes    "12 steps worse than 16: CLIP-I 0.8619. Sweet spot confirmed at 16 steps." \
  --is-best  false
echo "EXIT: $?" > /repo/_record_r13.txt
