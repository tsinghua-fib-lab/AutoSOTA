#!/bin/bash

# SACHS
# PYTHONPATH=. python prior/train_dag_gfn.py --data_npz data/sachs_observational.npz --out_ckpt sachs_gfn_ckpt.pt --steps 5000

# CAUSALBENCH
PYTHONPATH=. python prior/train_dag_gfn.py --data_npz data/causalbench/exports/weissmann_k562_50.npz --out_ckpt cb50_gfn_ckpt_std.pt --steps 5000 \
  --batch_size 8 \
  --tau 50.0 \
  --max_edges 200 \
  --lr 1e-3 \
  --hidden 256 \

