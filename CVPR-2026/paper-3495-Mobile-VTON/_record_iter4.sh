#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     4 \
  --idea-id  "IDEA-009" \
  --title    "Garment Injection (70% GT blend)" \
  --status   failed \
  --primary  0.8566 \
  --metrics  '{}' \
  --notes    "Garment injection targets discarded latent portion. No effect (results identical to iter 3). Need image-space blending or different approach." \
  --is-best  false
echo "EXIT: $?" > /repo/_record_r4.txt
