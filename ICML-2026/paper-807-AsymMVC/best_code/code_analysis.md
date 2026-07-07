# Code Analysis: HAMC (Paper 807)

## Evaluation Path
- Entry: `python3 main.py --train True --device_num 0 --seed 3`
- Output: last line `ACC: 0.XXXX | NMI: 0.XXXX | ARI: 0.XXXX`
- Metrics: ACC (Hungarian), NMI (sklearn), ARI (sklearn) — standard, unchanged

## Files
- `main.py`: Training pipeline (warmup → init prototypes → main training → inference)
- `model.py`: HAMC_Model, HyperbolicUtils, sinkhorn_knopp, losses
- `utils.py`: Data loading, cluster_acc, entropy, set_seed (already comprehensive!)

## Safe Modification Targets
- Training loop hyperparameters (tau, temp_gate, etc.)
- Loss function (add regularization terms)
- Optimizer (Adam → AdamW, gradient clipping)
- Sinkhorn implementation (epsilon schedule, convergence check)
- Prototype update mechanism
- Gradient handling

## Risky Files (DO NOT MODIFY)
- `utils.py:cluster_acc()` — metric definition (Hungarian algorithm)
- `utils.py:data_load()` — dataset loading
- `Data/CUB.mat` — test data

## Existing Best Practices Already Present
- `set_seed()` already does comprehensive deterministic setup
- Best-model checkpointing already exists (every 5 epochs)
- Momentum prototype update already uses top-50% confidence masking
