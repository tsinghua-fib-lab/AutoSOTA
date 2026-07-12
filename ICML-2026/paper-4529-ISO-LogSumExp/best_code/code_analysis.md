# Code Analysis — Paper 4529 SOTA Preparation Repair

## Original Failure

The SOTA preparation failed because `python3 eval.py` timed out after 1860 seconds (31 minutes). The root cause was that the evaluation script ran entirely on CPU despite having 2x A100 GPUs available, making the 70-run grid search (5 LRs x 10 seeds proposed + 10 seeds final + 10 seeds baseline) take ~95 minutes — far exceeding the 30-minute timeout.

## Repair Applied

1. **GPU enablement**: Added `device = torch.device('cuda:0')` and moved all tensors to GPU.
2. **Pre-loaded GPU data**: Entire dataset (~660KB) moved to GPU once, eliminating per-batch transfers.
3. **Manual batching**: Replaced DataLoader with manual index-based batching to avoid Python overhead.
4. **Removed grid search**: Use the known-best LR=5e-7 from the reproduction manifest.
5. **Separated proposed/baseline**: `--baseline` flag to run each mode independently.

## Corrected In-Container Evaluation Command

```bash
cd /repo && python3 eval.py --n-seeds 10 --lr 5e-7        # Proposed evaluation
cd /repo && python3 eval.py --baseline --n-seeds 10 --lr 1e-6  # Baseline
cd /repo && python3 eval.py --quick --lr 5e-7              # Quick 3-seed check
```

## Baseline Verification

| Metric | Reproduction | This Repair | Match? |
|--------|-------------|-------------|--------|
| proposed_objective | 0.7485 +/- 0.0024 | 0.7487 +/- 0.0019 | Yes |
| baseline_objective | 0.7762 | 0.7763 +/- 0.0002 | Yes |

Both metrics are within numerical noise of the reproduction baseline.

## Reusable Resources

- `/datasets/california_housing_data.npy` — cached California Housing features (1.3 MB)
- `/datasets/california_housing_target.npy` — cached targets (162 KB)
- `/repo/linreg_weights.pt` — least-squares initialization weights (auto-generated)

## Safe Optimization Targets

- Learning rate schedule (constant to linear decay, step decay, cosine)
- Model evaluation strategy (last iterate to EMA)
- Rho annealing schedule (fixed to log-linear)
- Numerical precision of softplus computation
- Gradient clipping threshold
- Per-parameter learning rates (separate alpha LR)
- Alpha initialization value
- Number of training epochs
- NaN/inf detection and recovery

Must NOT change: data loading, preprocessing, loss formulation, metric computation, batch_size=10, lambda=5, rho_end=1e-3, 10 seeds for final eval, LS init.
