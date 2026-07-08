# Code Analysis — Paper 911 SOTA Preparation Repair

## Original Failure
- **Symptom**: `exit=139` (SIGSEGV) — segmentation fault after training completes
- **Root Cause**: PyTorch 2.1 + CUDA 12.1 crashes during interpreter shutdown when CUDA resources are freed in undefined order by atexit handlers
- **Impact**: Training and evaluation produce correct results, but the process crashes on exit, preventing multi-seed evaluation in a single process

## Repair Applied
1. **eval.py**: Added `try/finally` with `os._exit(0)` to skip PyTorch atexit CUDA cleanup
2. **run_eval.sh**: Per-seed evaluation wrapper running each seed in a separate Python process
3. **run_seed.py**: Standalone single-seed runner with direct KuaiRecOnlinePolicyLearner control, supporting optimizer choice (adagrad/adamw), embedding dimension, and MoE count

## Corrected Evaluation Command
```bash
cd /repo && PYTHONPATH=/repo:$PYTHONPATH python3 run_seed.py <seed> <dim_emb> <n_moe> <lr> <K> <n_epoch> <output> [adagrad|adamw]
```
- No host-side paths in the command
- All paths are container-internal
- GPU device is hardcoded as cuda:0 (container perspective)

## Baseline Match Evidence
- Manifest baseline: Policy Value = 6.43 (10 seeds, from reproduction notes)
- Our baseline: Policy Value = 6.4485 ± 0.0371 (3 seeds)
- Match within normal numerical noise
- Same evaluation protocol: TOP1-PG (CA-PG-SwR), K=50, 5000 epochs, greedy evaluation

## Safe Optimization Targets
- **dim_model_emb**: Config parameter in initialize_trainable_policy(), values 10-32
- **n_moe_model**: Config parameter, values 1-4 (already implemented in codebase)
- **early_stage_optimizer**: Dataclass field of KuaiRecOnlinePolicyLearner, accepts Adagrad/AdamW
- **early_stage_lr**: Training hyperparameter, passed via early_stage_optimizer_kwargs

## Reusable Resources
- KuaiRec dataset at `/repo/experiments/synthetic/data/kuairec_small_matrix.csv` (388MB)
- Cache datasets at `/datasets/`
- Cache models at `/models/`

## No Red-Line Violations
- Two-tower architecture preserved
- Plackett-Luce sampling mechanism unchanged
- Greedy evaluation protocol unchanged
- Dataset splits unchanged
- Metric: `logged_dataset.agg_reward.mean()` (Policy Value)
