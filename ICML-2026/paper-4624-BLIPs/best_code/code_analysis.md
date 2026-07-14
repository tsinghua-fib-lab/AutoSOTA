# Code Analysis — BLIP Paper 4624

## Evaluation Path
- `eval_combined.py` → builds `BayesGNN` model → loads checkpoint → MC dropout inference → MSE/NLL/CRPS
- Output format: `Seed N: MSE=X.XXXXXX, MSE×10⁻¹=Y.YYYY, NLL=Z.ZZZZ, CRPS=W.WWWWWW`
- Primary metric: `MSE_x10^-1` (lower is better), parsed from stdout

## Training Path
- `experiments/nbody_main.py` → `experiments/utils.py:run()` → `train()` with `regression_loss`
- Loss: MSE + KL divergence (Bayesian). Bayesian training uses `regression_loss` + `model.kl_loss()`
- Checkpoints saved at best val_loss epoch to `ckpt/nbody/{seed}/BayesGNN_nbody_ckpt.pth`

## Config Path
- Defaults in `experiments/nbody/args.py` (paper hyperparameters)
- `eval_combined.py` hardcodes defaults matching paper values in `DefaultArgs` class

## Checkpoint State
| Seed | Epoch | Val Loss | MSE×10⁻¹ |
|------|-------|----------|-----------|
| 0    | 208   | 0.01089  | 0.1036    |
| 1    | 215   | 0.01140  | ~0.10     |
| 2    | 662   | 0.01065  | 0.0957    |
| 3    | 240   | 0.01132  | ~0.10     |

Paper trains for 10000 epochs; current checkpoints at 200-662 epochs. Seed 2 has most training + best results → longer training likely helps.

## Key Levers (from manifest)
- Training epochs: paper=10000, reproduced ~200-662
- Learning rate: 5e-4 (constant)
- Batch size: 100
- Prior dropout probability: 0.5
- KL weight beta: 0.01
- Posterior network size: hidden_dim_posterior=64
- Model depth: num_layers=4
- MC steps: 100

## Safe Modification Targets
- `eval_combined.py` — evaluation only, DO NOT modify metrics/parsing
- `experiments/nbody_main.py` — training script, safe to modify training procedure
- `experiments/utils.py` — training loop, safe to modify train/run functions
- `experiments/nbody/args.py` — argument defaults, safe to add/change defaults
- `blip/bayes/kl.py` — KL divergence, safe to modify annealing
- `blip/bayes/layer.py` — Bayesian linear layer, safe to modify
- `blip/posterior.py` — posterior network architecture, safe to modify

## Risky Files (DO NOT MODIFY)
- `experiments/nbody/dataset.py` — dataset loading, would change data
- `eval_combined.py` metric computation — would change evaluation protocol
- `/tools/record_score.sh` — scoring infrastructure
- Dataset files in `data/nbody/` — test data

## Optimization Strategy
1. **Most impactful**: Train for 10000 epochs (current: 200-662)
2. **Second tier**: KL annealing, cosine LR schedule, gradient clipping
3. **Fine-tuning**: Posterior architecture, prior probability, beta tuning
