# Code Analysis — Paper 4636: FMCPE Flow Theta Ablation

## Evaluation Path
- Script: scripts/run_fmcpe_flow_theta_ablation.py
- Output: results/flow_theta_ablation/raw_results.json
- Metrics: jC2ST, jMMD, jWass computed on joint (theta, y) distributions

## Key Files
- training/trainers/calibration.py — FMPostTransformTrainer (optimizer L177-180, training loop L191-274)
- flow_matching/torch_flow.py — FlowMatching (compute_loss L359, _build_drift L1015)
- utils/networks.py — ResMLP (L450-535) with zero-init final layer
- scripts/run_fmcpe_flow_theta_ablation.py — eval script with metric computation

## Baseline
Seed 43 converges (jC2ST=0.609), seeds 33/53 fail (jC2ST~0.96). Architecture works, training unstable.

## Safe Modification Targets
- Optimizer config (weight_decay), loss weights, EMA, spectral norm, OT coupling
- Config files in configs/method/fm_post_transform/
