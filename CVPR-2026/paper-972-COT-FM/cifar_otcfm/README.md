# CIFAR-10, OT-CFM (COT-FM)

Unconditional CIFAR-10 image generation on the OT-CFM backbone (paper §4.2, Table 2). Built on [torchcfm](https://github.com/atong01/conditional-flow-matching) (vendored in `torchcfm/`) and [AlignFlow](https://github.com/konglk1203/AlignFlow).

## Env

Requires PyTorch and the bundled `torchcfm/`. Install dependencies following the [torchcfm](https://github.com/atong01/conditional-flow-matching) repo.

## Run

Four stages, run in order. SLURM wrappers `cluster.sh` / `reverse.sh` / `job.sh` / `evaluate.sh` hold the full flags; the core commands are:

```bash
# Stage 0: cluster targets
python clustering.py

# Stage 1: reverse ODE to estimate cluster-wise source distributions (see reverse.sh for full args)
torchrun --nproc_per_node=8 reverse.py

# Stage 2: COT-FM fine-tune
python train_cifar10.py --model cotfm --seed 10 --wandb_name cotfm_block_4 --output_dir ./results/cotfm_block_4

# Stage 3: evaluate (FID)
torchrun --nproc_per_node=8 compute_fid_multi_gpu.py --path_pattern cotfm_block_4
```

## Data & assets

**Produced by the pipeline:**
- `cifar10_{n}_cluster.pt`, `reverse_stats_cov_{n}.pth` — cluster data and per-cluster source-distribution statistics (from `clustering.py` + `reverse.py`).

**Download manually:**
- `cotfm_otcfm.pt` — COT-FM fine-tuned checkpoint, from [Google Drive](https://drive.google.com/drive/folders/1Jy7bhbI6LtwehKNY8tCx3OY1HNS6xcuO?usp=sharing).
