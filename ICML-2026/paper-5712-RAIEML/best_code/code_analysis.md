# Code Analysis — Paper 5712 SOTA Preparation Repair

## Original Failure

The evaluation command failed with a FileNotFoundError for the Hugging Face dataset
lmarena-ai/arena-human-preference-140k.

## Root Cause

The script removes HF_ENDPOINT (set to hf-mirror.com) because the mirror does not serve
this dataset. The first download attempt from huggingface.co failed due to a transient
network issue. On retry, the dataset downloaded successfully via proxy.

## Repair Applied

No code changes were needed. The fix was retrying the evaluation command.
The dataset (135,634 rows, 7 parquet shards) is now cached.

## Baseline Reproduction Evidence

Robust Lottery rho=0.4 win rate: 44.48% (manifest: 44.48%, paper: 44.24%)
Maximal Lottery rho=0.0 win rate: 44.03% (paper: 44.06%)
Within rubric CI [43.92, 44.56].

Config: 23 models, 4 languages (en, pl, ru, zh), 200 bootstrap, 80/20 split,
alpha=1.0, ECOS solver, seed=42.

## Corrected In-Container Evaluation Command

    cd /repo && python3 reproduce_lmarena.py

## Safe Optimization Targets

1. LP input preprocessing: alpha tuning, significance filtering, adaptive smoothing
2. LP formulation: per-group rho, entropy regularization, Wasserstein alternative
3. Bootstrap strategy: stratified sampling, multi-seed ensemble
4. Model/language selection: deduplication, K-tuning
5. Solver improvements: HiGHS, tighter tolerances
6. Parameter sweeps: finer rho grid

Red line: do not modify evaluation protocol or dataset labels.
All ideas reversible with rollback signals.

## Cache Resources

HF dataset cache: /autosota_cache/hf/datasets/lmarena-ai___arena-human-preference-140k/
No /paper_data mount available.
