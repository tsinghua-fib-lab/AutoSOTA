# Paper 1918: Few-for-Many Personalized Federated Learning (FedFewS)
# Optimization Log

**Paper**: Few-for-Many Personalized Federated Learning (FedFew)
**Repo**: https://github.com/pgg3/FedFew
**Model**: 4-layer CNN (FedAvgCNN)
**Dataset**: CIFAR-100, Dirichlet α=0.5, M=20 clients
**Target Metric**: Averaged Test Accuracy (higher is better)

---

## Summary

| Metric | Value |
|--------|-------|
| Paper Reported | 53.69% ± 4.79% |
| Reproduced Baseline (2000 rounds) | 54.01% |
| 500-round Baseline | 49.85% |
| **Optimized Best** | **56.38%** (+6.53pp vs 500r baseline, +2.37pp vs repro) |

---

## Applied Optimizations (Stacked)

### Iter 1: CosineAnnealingLR (IDEA-001) → 50.41% (+0.56pp)
**File**: `PFLlib/system/flcore/clients/clientfedfews.py`

Replaced `ExponentialLR` with `CosineAnnealingLR`:
- Initial LR: 0.01, cosine decay to eta_min=0.0001
- T_max = global_rounds (1000)
- Applied to both standard and rep-mode optimizers
- No warmup (warmup version hurt early learning)

### Iter 3: Mixup Augmentation (IDEA-003) → 51.07% (+0.66pp)
**File**: `PFLlib/system/flcore/clients/clientfedfews.py`

Added MixUp during local client training:
- β(0.2, 0.2) distribution for mixing coefficient λ
- Enforces λ ≥ 0.5 (convex combination toward first ordering)
- Mixed loss: λ·CE(pred, y_a) + (1-λ)·CE(pred, y_b)
- Helps regularize clients with skewed label distributions under non-IID

### Iter 4: Distinct Model Initialization (IDEA-005) → 54.06% (+2.99pp) 🚀
**File**: `PFLlib/system/flcore/servers/serverfedfews.py`

K=3 server models initialized with different random seeds:
- Instead of identical `deepcopy` copies, models k=1,2 are reinitialized via `reset_parameters()`
- Accelerates STCH-Set specialization — w_{ik} weights become non-uniform earlier
- This was the **breakthrough** — exceeded paper's 2000-round baseline at only 500 rounds

### Iter 5: Extended Training to 1000 rounds (IDEA-020) → 56.38% (+2.32pp) 🏆
**File**: `configs/cifar100/noniid_dir_20_a0p5/base.yaml`

Extended training with cosine schedule:
- global_rounds: 500 → 1000 (not back to 2000! 1000 with cosine is more efficient)
- CosineAnnealingLR automatically adapts to the extended horizon

---

## Failed Ideas (Rolled Back)

| Iter | Idea | Result | Delta |
|------|------|--------|-------|
| 2 | Label Smoothing (0.1) | 48.56% | -1.85pp |
| 6 | 1200 rounds | 55.78% | -0.60pp |
| 7 | Local Epochs = 2 | 53.94% | -2.44pp |

---

## Config Changes (base.yaml)

```yaml
training:
  global_rounds: 1000       # was 2000
  learning_rate: 0.01       # was 0.005 (initial LR for cosine)
  learning_rate_decay: true # was not set (needed for scheduler stepping)
  learning_rate_decay_gamma: 0.99  # kept for compatibility (not used by Cosine)
```

---

## Evaluation Command

```bash
python scripts/run_pfllib.py configs/cifar100/noniid_dir_20_a0p5/algorithms/fedfews.yaml
```

Expected output: "Averaged Test Acc: ~56.38%"
Expected runtime: ~2 hours on 2x A100-80GB (1000 rounds)
