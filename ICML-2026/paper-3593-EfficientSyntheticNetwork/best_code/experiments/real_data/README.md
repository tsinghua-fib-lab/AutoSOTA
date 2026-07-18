# Real-data experiment

Reproduces the paper's main real-data tables on DBLP, Yelp, YouTube,
PolBlogs.

## Setup

The LSM-fitted reference adjacency for each dataset is packaged at
`data/real/<dataset>/generator/seed=0.npy`. To re-fit the LSM from raw,
use `python data/prepare_real_data.py` (see script docstring).

## SyNGLER

```bash
for ds in dblp yelp youtube polblogs; do
  python experiments/real_data/run_syngler.py --dataset $ds --output runs/syngler/$ds
done
```

This trains the LSM (or loads if cached), then produces 200 samples from
both SyNG-R and SyNG-D.

## Baselines

After cloning each baseline (see `baselines/<name>/README.md`):

```bash
for ds in dblp yelp youtube polblogs; do
  python baselines/cell/run.py  --config baselines/cell/configs/$ds.yaml  --output runs/cell/$ds
  python baselines/higen/run.py --config baselines/higen/configs/$ds.yaml --output runs/higen/$ds
done
```

GRAN, EDGE, VGAE on real data use the same runners as in the sparse
simulation; provide `--config baselines/<name>/configs/<dataset>.yaml`.

## Evaluation

```bash
python scripts/reeval_paper_metrics.py \
    --method syngler --dataset dblp \
    --samples_root runs/syngler/dblp/samples \
    --reference data/real/dblp/generator/seed=0.npy \
    --out_dir runs/eval_paper/syngler/dblp

python scripts/aggregate_paper_metrics.py --runs_root runs/eval_paper
```
