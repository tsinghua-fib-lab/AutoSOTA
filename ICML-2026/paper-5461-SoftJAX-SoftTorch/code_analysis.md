# SOTA Preparation Repair — Paper 5461 (SoftJAX)

## Failure Diagnosis

The preparation step failed because:

1. **Git not installed**: The container image does not include git.
2. **Network proxy blocks apt/conda**: Proxy env vars pointed to unreachable proxies.
3. **Workaround**: Unsetting all proxy env vars before apt-get succeeded. Git 2.25.1 installed.

## Corrected Evaluation Command

Inside container:
```bash
cd /repo
CUDA_VISIBLE_DEVICES=0 python3 -u mnist_sort_experiment.py
```

## Baseline Verification

- sequence_accuracy: 94.0% (manifest: 94.0%)
- element_wise_accuracy: 95.9% (manifest: 95.9%)
- Config: 3-layer CNN (32-64-128), 80 epochs, batch_size=300, tau=0.1, SoftSort smooth
- Training time: ~30min on A100
- Data: MNIST from /paper_data (read-only)

## Safe Optimization Targets

- Training hyperparameters (LR schedule, gradient clipping, tau annealing)
- Model architecture (BatchNorm, skip connections)
- Data pipeline (augmentation, curriculum sampling)
- Sort operator (NeuralSort, straight-through estimator)
