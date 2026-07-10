# Code Analysis — Paper 3099: MNIST-USPS W2R2

## Evaluation Path
- Command: python run_reproduction.py
- Timeout: 120 minutes
- Metric parsing: stdout FINAL_METRIC acc_target and summary.json
- Primary metric: acc_target (target test accuracy, 0-1 scale)

## Train/Inference Path
1. run_reproduction.py:main() -> setup_reproduction_config() -> run_uda()
2. run_uda() -> setup_uda() (scenario init + pretrain) -> adaptation loop
3. Per run: load_model() loads pretrained MLP -> init_algorithm() creates WeightedWRR
4. alg.adapt() -> calc_loss() computes OT plan + weighted source loss -> backward + step

## Config Path
- reproduction_config.py -> setup_reproduction_config()
- Algorithm: config[weighted_wrr] with reg_m=(1.0,100.0), uot_iter_max=1000
- Model: MLP layer_sizes [256,200,100,10] in load_model.py:init_model()
- Optimizer: Adam lr=1e-3, beta2=0.98 in run_reproduction.py:init_opt()

## Key Files
- models/mlp.py — MLP model (NO BatchNorm, NO Dropout)
- adapt/weighted_wrr.py — Core W2R2 algorithm
- sinkhorn_uot.py — mm_unbalanced OT solver (POT-based)
- loss.py — MarginLoss (Crammer & Singer)
- utils.py — train, test, report_metrics
- load_model.py — model init + pretrain
- shifts/mnist_to_usps.py — MNIST to USPS scenario

## Baseline Metrics
- Best accuracy: 87.89% (run 0), second: 87.19% (run 1)
- Failed runs: ~78.2% (runs 2-4)
- Mean: 81.98%, Std: 5.08%
- Paper reports: 86.0+-0.0053

## Known Bottlenecks
1. Seed sensitivity: 3/5 runs at ~78%
2. Fixed reg_m: (1.0, 100.0) for all epochs
3. No normalization: MLP has zero BatchNorm
4. Flat LR: No scheduler during adaptation
5. Final-epoch-only eval
6. Zip-based batching: no importance weighting

## Safe Modification Targets
- models/mlp.py: Add BatchNorm1d layers (with config flag)
- adapt/weighted_wrr.py: Dynamic reg_m schedule, prototype bank, entropy, EMA
- reproduction_config.py: New config fields
- run_reproduction.py: LR scheduler, best-epoch checkpointing

## Red-line Boundaries
- Do NOT change: test data, labels, dataset splits, metric computation
- Do NOT change: utils.report_metrics() or utils.test()
- Do NOT change: scenario data loading in shifts/mnist_to_usps.py
