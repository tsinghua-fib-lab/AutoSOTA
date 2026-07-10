# Code Analysis - CauchyNet Gap-Filling (Paper 3221)

## Evaluation Path
- Script: `/repo/experiments/best_config_gap_filling.py`
- Entry: `main()` function
- Output: `/repo/experiments/results/best_config_gap_filling.json`
- Metrics: `$.CauchyNet.mae_mean`, `$.CauchyNet.mae_median`, `$.CauchyNet.mae_max`

## Key Code Sections
- **Line 98-129**: CauchyNet model definition (fixed elliptical poles, ReciprocalActivation)
- **Line 128**: SUSPICIOUS `/hidden_size` division — not present in `shared.py` canonical CauchyNet
- **Line 170-239**: `train_score_one()` — training loop with StepLR, grad clipping, best-val checkpointing
- **Line 242-260**: `run_model()` — multi-seed aggregation by concatenating per-seed errors
- **Line 263-395**: `main()` — orchestrates all 4 models, 10 seeds, 3000 epochs

## Metric Parser
- Metrics come from `run_model()` which concatenates all per-seed errors and computes numpy statistics
- Primary: `mae_mean` (lower is better)
- Guardrails: `mae_median`, `mae_max`

## Safe Modification Targets
1. `CauchyNet.__init__()` (lines 107-119): pole radii, lambda init, hidden_size
2. `CauchyNet.forward()` (lines 121-129): output computation (remove /hidden_size)
3. `train_score_one()` (lines 170-239): LR schedule, grad clipping, loss function, epochs
4. `run_model()` (lines 242-260): ensemble averaging vs concatenation

## Risky Files
- `build_data()` (lines 69-88): DO NOT modify — defines test split
- `f()` and `df()` (lines 53-66): DO NOT modify — target function
- Metric computation in `run_model()` and `train_score_one()`: DO NOT modify the math

## /paper_data
- No pre-downloaded paper data mounts. The experiment is self-contained (target function in code).
