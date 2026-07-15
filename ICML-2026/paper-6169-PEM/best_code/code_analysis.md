# Code Analysis for Paper 6169: Depth over Fidelity

## Evaluation Path
- tools/eval_reproduce_figure2.py orchestrates evaluation
- Step 1: runs tools/run_coco_bbob_noisy.py with COCO bbob-noisy suite
- Step 2: tools/summarize_coco_noisefree_from_exdata.py
- Step 3: Computes median log10 regret from summary CSV

## Core Algorithm
- src/algorithms/berw_es.py: BERW-Hetero (Bootstrap Expected Rank Weights)
- Entry point: my_optimizer_noise_adaptive_sel_bootstrap_weights_hetero
- Class: BerwES2NoiseAdaptiveSelBootstrapWeightsHetero

## Key Defaults (BERW-Hetero at d=40)
- popsize=15, mu=7, n_start=1.2, n_end=4.0
- c_mean=1.0, mean_update=weiszfeld, restart_patience=80
- reeval_min=2, reeval_max_frac=0.3, reeval_extra_per_point=1
- bootstrap_samples=32, noise_model_ema=0.25
- noise_pool_max=512, z_clip=10.0

## Baseline
- RB-PEM: 3.24, Resample(k=5): 3.5
- B=100d=4000, lambda=15 => ~267 generations

## Safe Modification Targets
- src/algorithms/berw_es.py: hyperparameters, new algorithm variants
- tools/run_coco_bbob_noisy.py: add new algorithm entry points

## Risky Files
- tools/eval_reproduce_figure2.py (evaluation protocol)
- tools/summarize_coco_noisefree_from_exdata.py (metric computation)
- COCO library (cocoex): test functions
- /tools/record_score.sh (scoring infrastructure)
