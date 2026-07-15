# Code Analysis — Paper 6153: Provable Training Data Identification

## Evaluation Path
- Entry point: eval_local.py (used for SOTA eval; eval.py is original for HF datasets)
- Command: python3 eval_local.py --dataset_name WikiMIA --model_path /models/pythia-6.9b --seed 42
- Data loading: get_dataset_local() loads WikiMIA from local parquet files at /datasets/WikiMIA/
- Score computation: prepare_score_dict() with disk cache
- PTDI pipeline: perform_ptdi() -> out_calibrated_sampling() -> compute_p_values() -> PowerEnhancedStableEstimator -> SelectivePrediction -> BH_matrix_multidim_optimized()
- Metrics output: flatten_dict() prints Min_20.0% Prob_Power_alpha_0.1 and Min_20.0% Prob_FDP_alpha_0.1

## SAFE to modify
1. inference() in eval_local.py — detection score computation
2. PowerEnhancedStableEstimator in PTDI.py — pi0 estimation
3. compute_p_values() — conformal p-value computation
4. out_calibrated_sampling() in PTDI.py — sampling ratios
5. random_sample_matrix_by_ratio() in PTDI.py — RNG source
6. BH_matrix_multidim_optimized() in PTDI.py — tie-breaking

## RED LINE — Do NOT modify
1. get_dataset_local() — data loading and splits
2. EvaluationCalculator.calculate_selective_metrics_extended() — metric definitions
3. flatten_dict() — metric output format
4. convert_huggingface_data_to_lists_by_label() — label assignment

## Cache
- Score cache: /repo/score_cache/scores_WikiMIA__models_pythia-6.9b.pkl
- New scores added to inference() must update or invalidate cache

## Baseline (iter 0)
- Power: 2.565, FDP: 6.842
- Commit: eb1e1be
