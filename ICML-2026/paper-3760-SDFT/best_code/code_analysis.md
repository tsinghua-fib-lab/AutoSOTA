# Code Analysis for Paper 3760: Self-Distillation Enables Continual Learning

## Evaluation Path
- Script: /repo/eval_science.py
- Command uses vLLM for generation, evaluates exact-match accuracy on 507 chemistry questions
- Output: eval_results.json with accuracy, num_correct, num_total
- PYTHONHASHSEED=42 set but vLLM seed not explicitly controlled; minor variance possible

## Training Path
- Script: /repo/main.py
- Config: /repo/distil_config.py (extends HuggingFace TrainingArguments)
- Trainer: /repo/distil_trainer.py (DistilTrainer, extends BaseTrainer from TRL)
- Key functions:
  - _compute_loss (line 1596): KL divergence loss
  - MemoryEfficientSyncRefModelCallback (line 91): EMA teacher sync
  - _get_train_sampler (line 608): RepeatSampler
  - _generate_and_score_completions (line 1322): vLLM generation

## Key Config Parameters
- alpha (default 0.0): Forward KL; alpha=0.5 for JSD
- ref_model_mixup_alpha (default 0.01): EMA rate for teacher
- num_generations (default 1): generations per prompt
- top_entropy_quantile (default 1.0): token filtering
- learning_rate (default 2e-5), num_train_epochs (default 1)

## Data
- Train: 2674 examples in data/science_data/train_data
- Test: 507 examples in data/science_data/eval_data
- No validation split

## Models
- Base: /models/Qwen2.5-7B-Instruct (~14.2GB)
- SDFT checkpoint: /models/sdft-science (~14.2GB, 67.85 percent accuracy)

## Infrastructure
- GPUs: 2x NVIDIA A100-SXM4-80GB
- DeepSpeed 0.18.4, vLLM 0.12.0, TRL 0.24.0 available in venv
- Overlay: 64GB free; cache mounts: 6.7TB

## Red-line Constraints
- DO NOT modify: eval_science.py, test data, data splits, scoring
- Safe to modify: distil_trainer.py, distil_config.py, main.py

## Safe Modification Targets
1. distil_trainer.py _compute_loss: Loss computation (logit standardization, token weighting)
2. distil_trainer.py MemoryEfficientSyncRefModelCallback: EMA sync (cosine schedule)
3. main.py: CLI args, gradient checkpointing
4. distil_config.py: New config parameters
