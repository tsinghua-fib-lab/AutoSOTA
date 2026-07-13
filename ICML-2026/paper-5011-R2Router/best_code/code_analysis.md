# Code Analysis for Paper 5011 (R2-Router)

## Evaluation Path
- **Primary eval script**: `run_eval_v2.py` — KNN-based predictors (cosine distance, distance-weighted)
- **Alternative**: `run_eval_mlp.py` — 3-layer MLP predictors (paper architecture)
- **Original**: `run_eval.py` — Ridge regression predictors (sub_10 split)
- **Manifest eval command**: `python3 run_eval_v2.py --training-data /datasets/training_data.pkl --output-dir /repo/results --k-neighbors 128 --train-frac 0.8 --seed 42`

## Training Data
- `/datasets/training_data.pkl` (40MB): 8400 queries × 1024-dim Qwen3-0.6B embeddings
  - 16 models, 1 with 0 budgets (kimi-k2.5), 6 with 1 budget, 1 with 8 budgets, 8 with 9 budgets
  - 86 total (model, budget) options for routing
- `/datasets/routerarena_embeddings.pkl` (34MB): Raw embeddings (unused by eval)
- `/datasets/router_data_10.json` (860KB): Sub_10 split (used by run_eval.py)

## Metric Parser
- Metrics printed to stdout: Peak Accuracy, AUDC (norm cost), QNC
- Saved to `--output-dir/metrics.json`: keys `peak_accuracy`, `AUDC`, `QNC`
- AUDC computed via `np.trapezoid(pc, nc)` on accuracy vs normalized cost curve
- QNC = normalized cost at target accuracy (best_llm * qnc_target_rate)

## Baseline Results
| Predictor | AUDC | Peak Acc | QNC |
|-----------|------|----------|-----|
| KNN k=128 (manifest) | 0.7301 | 0.7515 | 0.4860 |
| KNN k=64 (default) | 0.7241 | 0.7437 | 0.5958 |
| MLP [256,128,64] | 0.7166 | 0.7373 | 1.0000 |
| Recorded baseline | 0.7225 | 0.7432 | 0.5427 |

**Note**: k=128 is consistently reproducible (verified 2 runs). The recorded baseline (0.7225) likely used k=64. Manifest eval command uses k=128 → working baseline is AUDC=0.7301.

## Key Observations
1. KNN outperforms MLP significantly (0.7301 vs 0.7166 AUDC)
2. MLP QNC=1.0 means routing fails to match best LLM quality at any cost
3. Per-model-budget R² is low (~0.2 for 9-budget models with KNN)
4. 6 models with single budget have degenerate R² (1.0 or 0.0) due to small test sets
5. gemma-3n-e4b has 8 budgets (missing concise)
6. 8 models with 9 budgets each are the main prediction targets

## Safe Modification Targets
- `run_eval_v2.py`: KNN hyperparams (k, weights), cost normalization, routing λ
- `run_eval_mlp.py`: MLP architecture, training loop, regularization, feature selection
- Both scripts: ensemble methods, adaptive routing

## Risky Files (do not modify)
- Metric computation (AUDC, QNC, Peak Acc formulas)
- Data loading and train/test split logic (must preserve seed determinism)
- `/tools/record_score.sh`
