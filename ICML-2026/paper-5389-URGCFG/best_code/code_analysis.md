# Code Analysis - Paper 5389: GCF Mechanism Design

## Evaluation Path
- Script: scripts/solve_setting_a_n4.py
- Entry point: __main__ block
- Flow: Generate uniform samples -> Init Mechanism -> fit() -> Trainer.run() -> Parse metrics

## Train/Inference Path
- Model: models/finite_model.py (FiniteModel class)
- Wrapper: mech_design/mechanism.py (Mechanism, Trainer)
- Training loop: Trainer._train_step() in mechanism.py
- Output: mechanism_data[profits], mechanism_data[revenue]

## Config / Hyperparameters (solve_setting_a_n4.py)
- npoint = dim + 10 (=14) - candidate bundles
- lr = 0.05 - learning rate
- epsilon = 2e-3 - barrier tightness
- temp = 60.0 with warmup 1000 steps from 5.0
- batch_size, patience=600, cooldown=300, factor=0.85
- max_samples=1000, default_utility=1.5*epsilon

## Metric Parser
- Primary: mean_profit_per_good = profits.mean().item() / dim
- stdout: Mean Profit per Good: <float>
- JSON: tmp/mech/dim4_settingA/reproduction_result.json

## Safe Modification Targets
1. npoint - better approximation quality
2. Learning rate and scheduler params
3. Temperature annealing schedule
4. Epsilon (barrier tightness)
5. max_samples for better MC estimation
6. Y initialization in solve_setting_a_n4.py
7. Training loop best-checkpoint tracking
8. barrier() epsilon scheduling

## Risky Files (DO NOT MODIFY)
- tools/feedback.py - logging
- tools/utils.py - IR_constraint, model_min/max_constraint
- models/finite_model.py compute_mechanism() - metric computation
- /tools/record_score.sh - scoring infrastructure
