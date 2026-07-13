# RLIE Code Analysis for SOTA Optimization

## Evaluation Path
- Entry: `python -m rlie.main --config configs/dreaddit_prod.yaml`
- Config: `configs/dreaddit_prod.yaml` → overrides `configs/default.yaml`
- Pipeline: `main.py` → `run_single_task()` × 3 repeats → `run_eval_only.py`
- Final metrics: `outputs/run_*/real_dreaddit_repeatN/final_test_metrics_regression_only__*.json`
- Summary: `outputs/run_*/real_dreaddit_summary.json`

## Core Pipeline (per repeat)
1. Load Dreaddit data (train/val/test split from `data/real/dreaddit/`)
2. Sample 20 seed examples, call LLM (deepseek-v4-pro) to generate 5 initial rules
3. Score rules locally (deepseek-v4-flash), filter by coverage ≥ 0.2
4. Train elastic-net logistic regression (GridSearchCV: 30 C × 30 l1_ratio = 900 combos)
5. If not converged: select hard examples → LLM generates refinement rules → score → train → repeat
6. Best model selected by validation accuracy across rounds
7. Final test evaluation: regression_only mode (LLM re-scores rules on test, elastic-net predicts)

## Key Files
- `rlie/main.py` — Orchestrator, runs 3 repeats in parallel
- `rlie/core/task_runner.py` — Single task loop: generation → scoring → training → refinement
- `rlie/core/modeling.py` — `SoftmaxTrainer` with GridSearchCV on C/l1_ratio
- `rlie/core/rule_utils.py` — `select_hard_samples` (simple top-N by margin), `filter_rules_by_coverage`
- `rlie/core/feature_manager.py` — `RuleFeatureStore`, `prune_rules` (by per-rule validation F1)
- `rlie/llm/rule_generator.py` — LLM rule generation + FALLBACK_RULES (hotel review domain!)
- `rlie/llm/prompt.py` — `BasePrompt.batched_generation()` and `batched_generation_refine()`
- `rlie/llm/rule_scorer.py` — LLM-based rule scoring (returns binary vectors)
- `rlie/run_eval_only.py` — Final evaluation from saved rule snapshots
- `rlie/task_specs/real/dreaddit/prompts.yaml` — Dreaddit-specific prompts

## Critical Config Parameters (dreaddit_prod.yaml)
- `max_rules: 10` — Rule capacity, may be restrictive
- `rule_coverage_threshold: 0.2` — Fixed threshold
- `sample_size: 20`, `rules_per_call: 5`
- `convergence: patience: 3, max_rounds: 20, min_improvement: 0.001`
- `elastic_net: max_iter: 20000, class_weight: balanced, cv_folds: 5`
- `repeat: 3`
- LLMs: rule_learning=deepseek-v4-pro, scoring=deepseek-v4-flash

## Known Issues
1. **FALLBACK_RULES in rlie/llm/rule_generator.py:18-29**: Hardcoded for hotel review deception detection. If API fails during Dreaddit, garbage features are injected.
2. **select_hard_samples**: No diversity — just top N by (misclassified, margin_error). Can pick 20 near-identical examples.
3. **No probability calibration**: Elastic-net logistic regression with class_weight=balanced may produce miscalibrated probabilities.
4. **Fixed coverage threshold**: 0.2 for all rounds, no adaptation.

## Red-Line Safe Modification Targets
- `rlie/llm/rule_generator.py` — FALLBACK_RULES replacement
- `rlie/core/rule_utils.py` — select_hard_samples diversity improvement
- `rlie/core/modeling.py` — Add calibration step, grid parameter tuning
- `rlie/llm/prompt.py` — Add context parameters for refined prompts
- `rlie/core/feature_manager.py` — Delayed pruning
- `configs/dreaddit_prod.yaml` — Hyperparameter adjustments
- `rlie/task_specs/real/dreaddit/prompts.yaml` — Prompt improvements

## Do NOT Modify
- `rlie/datasets/` — Data splits, labels, data loader
- `rlie/core/metrics.py` — Metric computation
- `rlie/llm/rule_scorer.py` — Rule scoring logic
- Test data, label mappings, evaluation protocol
