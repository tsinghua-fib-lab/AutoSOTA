#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     9 \
  --idea-id  "IDEA-002" \
  --title    "Scheduler Shift 2.0" \
  --status   failed \
  --primary  0.8567 \
  --metrics  '{}' \
  --notes    "scheduler_shift 2.0 hurt all metrics (CLIP-I 0.8512). Default 3.0 is optimal." \
  --is-best  false
echo "EXIT: $?" > /repo/_record_r9.txt
