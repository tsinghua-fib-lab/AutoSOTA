#!/usr/bin/env bash
# Local machine/cluster configuration — TEMPLATE.
#
#   cp env.example.sh env.local.sh    # env.local.sh is gitignored
#   # edit env.local.sh with your real paths, then either:
#   source env.local.sh && python run.py black-hole --posterior meanflow ...   # interactive
#   REINF_K=512 ./submit.sh black_hole --array=0-24                            # Slurm
#
# The committed slurm/*.sbatch scripts source env.local.sh automatically if it
# exists, and submit.sh injects --partition from it — so none of these private
# paths ever need to be committed.

# Python interpreter (a conda/venv env with the experiment deps installed).
export PY="python"

# Slurm GPU partition (used by submit.sh's --partition).
export SLURM_PARTITION="gpu"

# ── Black-hole imaging ───────────────────────────────────────────────────────
export BH_PRIOR="third_party/InverseBench/checkpoints/blackhole-50k.pt"
export BH_DATA="third_party/InverseBench/data/blackhole"   # contains measure/ and test/
export MF_CKPT=""                                          # mean-flow snapshot for --posterior meanflow
export MF_ROOT="third_party/easy_meanflow"

# ── Super-resolution ─────────────────────────────────────────────────────────
export VAL_ROOT="data/imagenet_val_256"                   # ImageNet-val-256 dir (with raw/ and/or a tar)
export IMAGENET_VAL_TAR=""                                # optional ImageNet-val tar (for images missing from raw/)
export IMAGENET_TRAIN_ROOT=""                             # optional; only needed to (re)build the val manifest
export SR_CKPT="experiments/super_resolution/checkpoints/pMF-B-16.pt"

# ── Weights & Biases ─────────────────────────────────────────────────────────
export CBG_WANDB="off"                                    # "on" to log
# export CBG_WANDB_ENTITY="your-entity"
# export CBG_WANDB_PROJECT="calibrated-guidance"
