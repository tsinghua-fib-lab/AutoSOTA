# Code Analysis for AdaMeZO (Paper 2428)

## Evaluation Path
- **Eval command**: `cd /repo/MeZO/medium_models && bash /repo/eval_adaMeZO_rte.sh`
- **Eval script**: `/repo/eval_adaMeZO_rte.sh` — runs 4 seeds (13, 21, 42, 87) sequentially
- **Entry point**: `/repo/MeZO/medium_models/run.py` — main training/evaluation driver
- **Shell config**: `/repo/MeZO/medium_models/mezo.sh` — sets env vars and calls `run_fewshot.sh`
- **Shell args**: `/repo/MeZO/medium_models/run_fewshot.sh` — task-specific templates and mappings

## Train/Inference Path
- **Train loop**: `MeZO/medium_models/src/trainer.py:411` — `train()` method
- **ZO perturbation**: `efficient_perturb_parameters()` (line 245) — deterministic ZO sampling via seed+generator
- **ZO forward**: `zo_forward()` (line 229) — forward pass returning scalar loss
- **AdaMeZO update**: lines ~870-920 — Adam-style EMA with horizon h, beta1, beta2
- **Best-model tracking**: line 739-740 — saves state_dict when dev loss improves
- **Best-model restore**: line 1015-1017 — loads best checkpoint before final eval
- **Evaluation**: `evaluate()` (line 1024) — runs prediction_loop and logs metrics
- **Test evaluation**: `run.py` lines 1098-1131 — evaluates on test set, writes test_results_rte.txt

## Config Path
- **Training args**: `run.py` line ~960-980 — HuggingFace TrainingArguments
- **Data args**: `run.py` — data_dir, task_name, k-shot config
- **Model args**: `run.py` — model_name_or_path, template, mapping

## Metric Parser
- **Metric extraction**: `test_results_rte.txt` contains `eval_acc = <value>`
- The eval script greps for `eval_acc` and parses with `cut -d= -f2`
- Mean computed manually across 4 seed directories

## Key Files
| File | Role | Safe to modify? |
|------|------|-----------------|
| `src/trainer.py` | Core trainer with AdaMeZO update | YES - main target |
| `run.py` | Main entry point, args, eval | YES - but careful with eval logic |
| `src/processors.py` | Data processors including RTE | YES - data pipeline checks only |
| `src/dataset.py` | FewShotDataset tokenization | YES - data pipeline checks only |
| `mezo.sh` | Shell config with defaults | YES - hyperparameter tuning |
| `eval_adaMeZO_rte.sh` | Eval entry point | NO - red line (eval protocol) |
| `run_fewshot.sh` | Template and mapping | YES - but verify correctness first |
| `/tools/record_score.sh` | Score recording | NO - red line |

## Risky Files
- `src/trainer_ori.py` — Original trainer backup, DO NOT MODIFY
- `src/linearhead_trainer.py` — Linear head probing trainer, DO NOT MODIFY
- Anything in `large_models/` — Not used for RTE eval

## Safe Modification Targets
1. **`src/trainer.py` zo_update section (lines ~870-920)**: Change moment estimation formula
2. **`src/trainer.py` train() method**: Add gradient clipping, label smoothing, top-k checkpointing
3. **`mezo.sh`**: Change hyperparameter defaults (beta1, beta2, hw, LR, EPS)

## Current Baseline
- Mean Accuracy: 53.5% (4 seeds: 13, 21, 42, 87)
- Best seed (21): 60.3% — close to paper CI lower bound (60.8%)
- Paper target: 63.1%
- Key observation: Model peaks then diverges; best-model checkpointing is critical

## Key AdaMeZO Implementation Details
- `seed_history` and `projection_history` store the last `hw` random seeds and scalar projections
- The update reconstructs ZO gradient estimate using weighted EMA of past perturbations
- `temp_grad = sum proj[k] * beta1^(hw-1-k) * z_k`  (first moment EMA)
- `temp_hess = sum proj[k]^2 * beta2^(hw-1-k) * z_k^2`  (second moment EMA)
- `g = temp_grad / (sqrt(temp_hess) + 1e-5) * 100`  (Adam-style update)
- Multiplied by learning_rate and subtracted from parameters
