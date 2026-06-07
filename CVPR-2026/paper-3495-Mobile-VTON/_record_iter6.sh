#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     6 \
  --idea-id  "IDEA-012" \
  --title    "IP-Adapter Scale 1.5x Boost" \
  --status   failed \
  --primary  0.8567 \
  --metrics  '{}' \
  --notes    "IP-adapter 1.5x hurt CLIP-I (0.8510). DINOv2 features at higher scale disrupt garment conditioning. Rolled back." \
  --is-best  false
echo "EXIT: $?" > /repo/_record_r6.txt
