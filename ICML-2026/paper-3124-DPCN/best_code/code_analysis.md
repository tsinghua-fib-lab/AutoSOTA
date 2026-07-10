# Code Analysis for Paper 3124 (DeepPCNs)

## Evaluation Path
- Entry: `PC.py` main()
- Config: `hps/PC-SF/VGG5_CIFAR10.yaml` (VGG5, CIFAR10, T=7, α=0.001, se_flag=true)
- Seed runner: `seed.py` — 5 seeds [0,1,2,3,4], averages results
- Eval: `eval_on_batch()` — top1/top5 accuracy on test set
- Output: JSON file `<config>_accuracy.json` with avg, std, per-seed best

## Training Path
- Model: `model.py` VGG5 class (4 conv blocks + 1 linear classifier)
- Energy: `pxc.se_energy` (SE loss when se_flag=true)
- Inference: T-step iterative refinement via `pcx.predictive_coding.Vode`
- Optimizer: AdamW (weight params) + SGD with momentum (hidden state)
- Schedule: warmup_cosine_decay with init→peak=1.1×→end=0.1×, warmup=10%
- Forward type: FU (Forward Upward via `energyW_FU`)
- Precision: S (Spiking Precision weights)

## Config Path
- `hps/PC-SF/VGG5_CIFAR10.yaml` — main config for PC+S+F on VGG5/CIFAR10

## Metric Parser
- Per-epoch: stdout `Epoch N: top1=X.XXX, top5=X.XXX`
- Final: `<config>_accuracy.json` → `"avg"` field = mean best accuracy across seeds

## Reusable Resources
- `/datasets/cifar10/` — CIFAR10 dataset (torchvision cached)
- `/models/` — empty, cache mount only

## Risky Files (avoid modifying)
- `dataset.py:74-78` — test transform pipeline (hard constraint)
- `seed.py` — seed management and final score computation
- `PC.py:210-247` — early stopping logic (metric-dependent)
- `model.py` VGG5 forward() — evaluation forward pass

## Safe Modification Targets
- `PC.py:169-176` — optimizer schedule parameters
- `PC.py:194-197` — AdamW config + gradient clipping
- `PC.py:80-96` — inference loop (momentum, energy fn)
- `PC.py:122-128` — training batch processing (label smoothing, MixUp)
- `hps/PC-SF/VGG5_CIFAR10.yaml` — hyperparameter config
- `dataset.py:65-72` — training augmentations only (not test)
- `model.py` — VGG5 initialization (regularization additions)

## Eval Command (in-container)
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PC_SEEDS=[0,1,2,3,4] python3 -u PC.py hps/PC-SF/VGG5_CIFAR10.yaml
```

## Notes
- Container uses JAX 0.4.38, pcx 0.6.3, optax 0.2.5
- Uses AdamW (not SGD as paper states for standard PC)
- Schedule was tuned via TPE for SGD/25 epochs; retuning for AdamW/50 epochs is key
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` required for torch DataLoader compat
- Training time ~17 min/seed on A100
