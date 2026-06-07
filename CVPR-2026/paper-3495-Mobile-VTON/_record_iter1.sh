#!/bin/bash
cd /repo
bash /tools/record_score.sh \
  --scores   "/vepfs-mlp2/queue014/public/chenzhibin/AutoSota-14/optimizer/papers/paper-3495/runs/run_20260605_094651/results/scores.jsonl" \
  --iter     1 \
  --idea-id  "IDEA-014" \
  --title    "Guidance Decay Annealing (3.0->1.5)" \
  --status   success \
  --primary  0.8414 \
  --metrics  '{"clip_i": 0.8414, "ssim": 0.8792, "lpips": 0.0895}' \
  --notes    "Guidance decays from 3.0 to 1.5 over denoising. All metrics improved: CLIP-I +0.74%, SSIM +0.33%, LPIPS -2.1%." \
  --is-best  true
echo "EXIT: $?" > /repo/_record_r1.txt
