# Code Analysis for Paper 4966 - DynaSteer SOTA Optimization

## Evaluation Path
- `steer_eval.py` - Main evaluation: loads Qwen3-1.7B, applies steering via baukit hooks, evaluates on MATH500
- `compute_score()` uses math-verify with ExprExtractionConfig + LatexExtractionConfig
- Metric: Accuracy = correct / total on MATH500 test split
- Output: `Accuracy: X.XXXX (XX.XX%)` from stdout, also `summary.json`

## Training Path
1. `vllm_generate_train.py` - Generate responses on 200 MATH training problems
2. `chunk_process.py` → `chunk_sampling.py` → `filter_sampling.py` - Process into chunks
3. `collect_activations.py` - Collect activations from forks (380 samples from 110 forks)
4. `solve_steering.py` - Compute steering vectors (LDA + logistic probe), n_clusters=1

## Config Path
- Steering configs: `/repo/results/train_pipeline/Qwen3-1.7B_no_thinking/steering_configs/c1/steering_configs.pkl`
- Activations: `/repo/results/train_pipeline/Qwen3-1.7B_no_thinking/activations/activations_entropy_0.0_10.0_period_1_10.pkl`
- Chunk analysis: `/repo/results/train_pipeline/Qwen3-1.7B_no_thinking/chunk_analysis/`

## Metric Parser
- Parse `Accuracy: X.XXXX (XX.XX%)` from stdout
- Also available in `output_dir/summary.json` as `accuracy` field

## Known Levers
- steer_alpha (1.0 baseline, range 0.5-5.0)
- Steering unit selection (top-k by probe_acc, currently top-10)
- n_clusters in solve_steering.py (currently 1)
- Training data size (200 problems)
- Entropy threshold for dynamic steering
- Steering coefficient decay schedule

## Safe Modification Targets
- `steer_eval.py:build_hook_map()` - Unit selection, ranking, weighting
- `steer_eval.py:make_steer_edit_function()` - Steering application logic
- `steer_eval.py:generate_with_steering()` - Generation loop, entropy gating, decay
- `solve_steering.py` - Retraining with different n_clusters

## Risky Files (do not modify)
- `compute_score()` - Metric computation
- Dataset loading (MATH500 test split)
- `eval_math500.py` - Alternative evaluation, not used
