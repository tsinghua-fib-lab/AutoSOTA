# TimeGuard SOTA Preparation Repair — Code Analysis

## Original Preparation Failure

The preparation orchestrator ran `python3 eval_timeguard.py` with a 7260s (121 min) timeout. The evaluation runs 3 models (FEDformer, SimpleTM, TimesNet) sequentially, each with paper settings (t_b=10, t_1=10, t_2=90, k_nn=20, k_nn_max=64). 

**Root cause**: Sequential execution of 3 models with KNN precomputation (358 channels × O(N²)) and 90 Stage II epochs exceeds the 120-min eval timeout. The FEDformer alone requires ~80-100 min.

**Secondary issue**: The eval script used `subprocess.run(capture_output=True)`, meaning all output was lost when the orchestrator killed the timeout. No partial results were recoverable.

**Tertiary issue**: The original TimesNet model failed with an `ImportError: cannot import name DiagnosticOptions from torch.onnx._internal.exporter` under PyTorch 2.1.2. The container was updated to PyTorch 2.4.1 which resolves this.

## Repaired Evaluation Pipeline

### Corrected in-container evaluation command

Single model:
```bash
cd /repo && python3 run_parallel_eval.py --t2 15 --tb 5 --t1 5 --knn 20 --knn-max 30 --timeout 3600
```

This runs:
- FEDformer (GPU 0) and SimpleTM (GPU 1) in parallel
- TimesNet (GPU 1) after SimpleTM completes
- Per-model timeout: 3600s
- Config overrides: t_b, t_1, t_2, k_nn, k_nn_max

### Baseline metrics (reproduction)
- MAEc: 19.654, MAEp: 37.371, FDER: 0.758
- Settings: t_b=5, t_1=5, t_2=15, k_nn=10, k_nn_max=20
- Source: Manifest baseline_metrics

## Reusable /paper_data Resources
- `/paper_data/PEMS03/PEMS03.npz` (10MB): PEMS03 traffic dataset with 358 sensors
- Mounted read-only; data is copied/symlinked as needed

## Optimization Changes Applied

### CODE-001: Best Model Checkpoint
- Saves best model state during Stage II validation
- Restores best checkpoint before final evaluation
- Prevents degradation from noisy late-stage samples

### CODE-002: KNN Graph Caching
- Caches precomputed KNN graph to disk
- Cache key includes data hash, C, N, Kmax
- Saves ~10-15 min per model after first run

### CODE-004: Gradient Clipping
- max_norm=1.0 in both warm_up_backcaster and train_with_weights
- Prevents loss spikes from destabilizing weighted training

### ALGO-001: Cosine Annealing LR
- CosineAnnealingLR in warm_up_backcaster and train_with_weights
- T_max = training epochs, eta_min = lr × 0.01
- Improves convergence by annealing learning rate

## Safe Optimization Targets

1. Hyperparameters (safe, in defense_timeguard.py):
   - alpha (0.2): initial clean rate
   - beta (0.5): max clean rate
   - t_b, t_1, t_2: epoch counts
   - k_nn, k_nn_max: neighborhood sizes
   - learning_rate, learning_rate_phase_2

2. Training improvements (safe, in defense_timeguard.py):
   - LR scheduling
   - Gradient clipping
   - Loss functions
   - Pool selection heuristics

3. Config changes (safe, in YAML files):
   - Model hyperparameters
   - Training settings

## Red Lines (DO NOT CROSS)
- No changes to metric definitions or evaluation protocol
- No changes to test data, labels, or dataset splits
- No hard-coded predictions or metrics
- No edits to scores.jsonl by hand (use record_score.sh)
