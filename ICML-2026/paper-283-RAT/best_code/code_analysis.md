# Code Analysis - RAT (Paper 283) SOTA Optimization

## Evaluation Path
- Command: `python3 train_shared.py --config rat_mlp_shared.yaml --env_name halfcheetah --device 0 --seed 42`
- Config: `configs/rat_mlp_shared.yaml`
- Main script: `train_shared.py`
- Output: `logs/shared.rat.*/halfcheetah.*/progress.csv`
- Primary metric: Column 12 (eprewmean) - running mean episode return over last 100 episodes
- Guardrail: Column 6 (entropy) - policy entropy

## Key Files
- `train_shared.py` - Main training loop with RAT_Update, diag_Update, KFAC_Update
- `utils/utils.py` - SharedActorCritic, MuSigmaNet, model_step
- `utils/popart.py` - PopArt value normalization (beta=0.99999)
- `utils/runners.py` - Experience collection, GAE computation
- `configs/rat_mlp_shared.yaml` - Default configuration

## Risky Files (must not change)
- `utils/runners.py` - GAE computation (metric-affecting)
- Scoring/parsing logic - must not modify

## Safe Modification Targets
- `configs/rat_mlp_shared.yaml` - Parameters
- `train_shared.py:32` - gamma hardcoded value
- `train_shared.py:462-467` - KL adaptive LR
- `train_shared.py:51` - SGD weight_decay
- `utils/popart.py:11` - PopArt beta
- `utils/utils.py:293` - sigma_type assert
- `utils/utils.py:35-36` - linear sigma init

## Known Levers
- ent_coef: 0.0 → 0.001-0.01 (entropy regularization)
- gamma: 0.999 → 0.99 (discount factor)
- cg_damping: 0.1 → 0.01-0.5 (regularization)
- epochs: 4 → 8 (inner-loop updates)
- lr: 0.05 → 0.1 (learning rate)
- use_kl_adaptive_lr: True → False (disable adaptive LR)
- sigma_type: vector → mu_shared/separate (state-dependent variance)
- PopArt beta: 0.99999 → 0.999 (faster normalization adaptation)
