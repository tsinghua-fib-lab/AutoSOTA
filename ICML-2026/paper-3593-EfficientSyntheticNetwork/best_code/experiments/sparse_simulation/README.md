# Sparse simulation experiment

Reproduces Appendix `tab:sparse-modern-baselines` of the paper.

Setting: `n = 500`, `r ∈ {2, 3, 4}`, sparse LSM-generated graphs, 20
Monte Carlo replications per `(r, method)` cell.

## 1. Generate input data

```bash
for r in 2 3 4; do
  python data/generate_sparse_sim.py --r $r --seed_start 0 --seed_end 20
done
```

## 2. Train + sample each method

```bash
# SyNGLER (SyNG-R + SyNG-D)
python experiments/sparse_simulation/run_syngler.py --r 2 --seeds 0-19

# Baselines (after cloning each baseline; see baselines/<name>/README.md)
for r in 2 3 4; do
  for s in $(seq 0 19); do
    python baselines/gran/run.py  --config baselines/gran/configs/sparse_sim.yaml  --r $r --seed $s --output runs/gran/r=$r/seed=$s
    python baselines/edge/run.py  --config baselines/edge/configs/sparse_sim.yaml  --r $r --seed $s --output runs/edge/r=$r/seed=$s
    python baselines/vgae/run.py  --config baselines/vgae/configs/sparse_sim.yaml  --r $r --seed $s --output runs/vgae/r=$r/seed=$s
  done
done
```

## 3. Evaluate with paper metrics

```bash
for r in 2 3 4; do
  for m in gran edge vgae; do
    SEEDS=$(seq -s, 0 19)
    python scripts/reeval_paper_metrics.py \
        --method $m --r $r --seeds $SEEDS \
        --samples_root runs --out_dir runs/eval_paper
  done
done
```

## 4. Aggregate

```bash
python scripts/aggregate_paper_metrics.py \
    --runs_root runs/eval_paper --out_md runs/sparse_sim_table.md
```

`runs/sparse_sim_table.md` reproduces the Appendix
`tab:sparse-modern-baselines` row-for-row.
