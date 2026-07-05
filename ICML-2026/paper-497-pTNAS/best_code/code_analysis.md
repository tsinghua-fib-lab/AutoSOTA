# Code Analysis: pTNAS Optimization

## Evaluation Path
- **Entry point:** `scripts/ptnas_full.py` (main function, line ~550)
- **Pipeline:** Data load → EA selection (proxy-based) → Successive Halving → Final training → Test eval
- **Metric output:** CSV column `final_test_metric` in `run_outputs/ptnas_result.csv`
- **Metric parser:** `roc_auc_score(test_y, test_pred_hat)` at line ~556
- **Stdout:** `Test metric: X.XXXX` (line ~730)

## Key Modification Targets

### Safe (algorithm changes only)
1. `scripts/ptnas_full.py:312-349` — `create_evaluation_function`: proxy scoring for EA
2. `scripts/ptnas_full.py:399-484` — `successive_halving`: SH round configuration
3. `scripts/ptnas_full.py:485-540` — `train_model`: optimizer, LR schedule, loss function
4. `scripts/ptnas_full.py:674-702` — Final training warm-start section
5. `src/search_algorithm/ea.py:52-118` — EA population init, selection, mutation

### Do NOT Modify
- `scripts/ptnas_full.py:112-130` — `test()` function (metric computation)
- `scripts/ptnas_full.py:550+` — Main function structure (test split, eval protocol)
- `utils/table_data.py` — Data loading (test split definition)
- Any dataset files in `datasets/fit-medium-table/avito-user-clicks/`
- `/tools/record_score.sh` — Scoring infrastructure

## Proxy Infrastructure
- 13 proxy evaluators in `src/proxies/` — all registered in `__init__.py`
- Unified `evaluate(arch, device, batch_data, batch_labels, space_name) -> float` interface
- Currently only `ptproxy_score` is wired in the main pipeline

## Training
- Optimizer: Adam(lr=0.001) — line 497
- Loss: BCEWithLogitsLoss() — line 492
- No LR scheduler
- No weight decay
- Early stopping patience=10 on validation AUC

## Config/CLI Args
- `--given_time_budget`: search time budget (seconds)
- `--mk_ratio`: controls M/K ratio for time-aware planning
- `--seed`: random seed (default 42)
- `--final_lr`, `--final_epochs`, `--final_dropout`, etc.
