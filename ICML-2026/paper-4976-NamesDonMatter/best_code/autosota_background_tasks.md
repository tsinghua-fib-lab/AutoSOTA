# Background Tasks Ledger

## Task: iter3-algo04 training
- **Task ID**: baf3hwzw7
- **Iteration**: 3
- **Idea**: ALGO-04 — Architecture improvement (d_embed=128, num_heads=4, d_ff=1024, epochs=96)
- **Command**: `cd /repo && PYTHONPATH=/repo python3 -m autoregltl.main --model-path=/repo/models/iter3-algo04 --seed=42 train-ted --data-dir=data-prop --ds-name=ltl-35 --train-max-samples=80000 --val-max-samples=1000 --trace-max-samples=100 --epochs=96 --batch-size=256 --grad-acc-steps=4 --learning-rate=1e-3 --lr-scheduler-type=cosine --warmup-steps=1000 --eval-steps=3000 --logging-steps=500 --d-embed-enc=128 --num-heads=4 --d-ff=1024 --num-layers=6 --dropout=0.1 --ff-activation=relu --cross-attn=per --tree-pos-enc --dec-pe=rope --loss-fct=adacos`
- **Log path**: /repo/models/iter3-algo04.log
- **PID**: 68880
- **Start**: 2026-07-12T15:08 UTC
- **Deadline**: 2026-07-12T23:08 UTC (8 hour timeout)
- **Expected output**: /repo/models/iter3-algo04/pytorch_model.bin, checkpoints
- **Score row**: Will record after evaluation
