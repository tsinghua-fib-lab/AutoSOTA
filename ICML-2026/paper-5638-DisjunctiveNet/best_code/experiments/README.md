# PBMC3k DNF Reproduction Experiment

## Quick Start
```bash
export JULIA_DEPOT_PATH=/autosota_cache/julia
cd /repo
julia --project=. experiments/pbmc3k_dnf.jl
```

## What it does
1. Loads preprocessed PBMC3k dataset (2643 cells, 1838 features, 8 classes)
2. Splits into train (0.5% = 13 cells) and test (10% = 264 cells)
3. Trains MLP backbone (1838→8→8) for 500 epochs with AdamW(lr=3e-3)
4. Applies DNF projection layer at inference time using marker-gene rules
5. Reports Accuracy, Macro F1, Macro Precision, Macro Recall, CSAT

## Expected runtime
~25-30 minutes on an A100 GPU (most time in base training)

## Key files
- `/repo/experiments/pbmc3k_dnf.jl` - Main experiment script
- `/datasets/pbmc3k/` - Preprocessed PBMC3k data (X.npy, y.npy)
- `/repo/src/` - DisjunctiveNet library source
- `/autosota_cache/julia/` - Julia package depot

## Known limitations
- Gradient-based fine-tuning through the projection layer is not currently functional
  due to Zygote mutation restrictions in model construction
- DNF projection is applied at inference time only
- CSAT perfect (1.0) with current single-gene rules due to guaranteed feasibility
