#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     5 \
  --idea-id  "IDEA-004" \
  --title    "Triangular Beta-CFG Schedule (1.5->3.5->1.5)" \
  --status   success \
  --primary  0.8567 \
  --metrics  '{"clip_i": 0.8567, "ssim": 0.8895, "lpips": 0.0900}' \
  --notes    "Triangular schedule vs linear decay: essentially same (CLIP-I +0.0001). The TTA dominates; schedule shape has minimal incremental effect." \
  --is-best  true
echo "EXIT: $?" > /repo/_record_r5.txt
