#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     7 \
  --idea-id  "IDEA-014b" \
  --title    "Stronger Guidance Decay (4.0->2.0)" \
  --status   failed \
  --primary  0.8567 \
  --metrics  '{}' \
  --notes    "Stronger guidance hurt all metrics (CLIP-I 0.8456). 3.0->1.5 is the sweet spot." \
  --is-best  false
echo "EXIT: $?" > /repo/_record_r7.txt
