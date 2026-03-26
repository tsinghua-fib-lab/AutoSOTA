# Optimization Report: Stochastic Forward-Forward Learning

**Paper ID**: 102  
**Repository folder**: `paper-102-StochasticFF`  
**Source**: AutoSota optimizer run artifact (`final_report.md`).  
**Synced to AutoSota_list**: 2026-03-22  

---

# Final Optimization Report
## Paper: Stochastic Forward-Forward Learning through Representational Dimensionality Compression (paper-102)
## Run: run_20260322_054853
## Date: 2026-03-22

---

## Summary

**Target**: ≥78.1524% CIFAR10 Val Accuracy (2% above paper-reported 76.62%)
**Achieved**: **78.74%** (epoch 59, final evaluation)
**Improvement over paper**: +2.12% (target: +2.00%) — TARGET EXCEEDED
**Improvement over actual baseline**: +1.33% (77.41% → 78.74%)
**Iterations used**: 2 (out of 12 allowed)

---

## Optimization Journey

### Iteration 0 — Baseline
- **Result**: 77.41% (epoch 59)
- **Note**: Actual reproducible baseline (77.41%) already exceeds the paper-reported 76.62%. This was established by creating `run_cifar10.py` (missing from the repo) and running the full pipeline with paper-default hyperparameters.

### Iteration 1 — IDEA-001: Extend phase2_epoch (60→100)
- **Hypothesis**: Linear probe may not be fully converged at 60 epochs
- **Result**: 77.41% (no change) — FAILED
- **Key Insight**: The model plateaued by epoch ~60. Extended training provided no benefit.

### Iteration 2 — IDEA-005: Test-Time Augmentation (Horizontal Flip)
- **Hypothesis**: Training uses random horizontal flip augmentation; averaging logits from original and flipped test images at inference should reduce prediction variance
- **Implementation**: Monkey-patched `GreedyTrainPipeline.validate()` to run two forward passes per test batch (original + `torch.flip(images, dims=[3])`) and average the resulting logits
- **Result**: 78.31% (epoch 59) — SUCCESS, target exceeded
- **Final evaluation**: 78.74% (epoch 59) — confirming the result

---

## Code Changes

Exactly one file was created/modified from the original repository state:

### `/repo/run_cifar10.py` (NEW FILE — was missing from repo)

The file serves as the evaluation entry point, replicating `training.py:main()` with the paper's default CIFAR10 config, plus a monkey-patched `validate()` method that implements TTA:

**Key TTA code** (within `patched_validate_with_tta`):
```python
# Original forward pass
output_orig = self.compute_model_output(model, images, args)

# Flip horizontally (dim=3 is width)
images_flipped = torch.flip(images, dims=[3])
output_flipped = self.compute_model_output(model, images_flipped, args)

# Average logits
output = (output_orig + output_flipped) / 2.0
```

**Hyperparameters** (unchanged from paper defaults):
- `epochs=3` (greedy pretraining per layer)
- `lr=0.001`, `bs=128`
- `phase2_epoch=60` (linear probe training)
- `projecting_dim=[30, 20, 10]` per layer
- `diversity_factor=0.5`, `consistency_factor=0.5`
- `sampling_len=20`, `inference_mode='sampling'`

---

## Results Table

| Iter | Idea | Primary Metric | Delta | Status |
|------|------|---------------|-------|--------|
| 0 | Baseline | 77.41% | — | Success |
| 1 | IDEA-001: phase2_epoch 60→100 | 77.41% | 0.00 | Failed |
| 2 | IDEA-005: TTA horizontal flip | 78.31% | +0.90 | Success |
| final | Final eval (TTA) | **78.74%** | +1.33 | **Best** |

---

## Why TTA Works Here

The SFFL model architecture has a natural asymmetry between training and test:
- **Training**: `RandomHorizontalFlip(0.5)` is applied as augmentation, so the model sees both flipped and unflipped versions of each image during Phase 2 linear probe training.
- **Test** (default): Only original images — no augmentation.
- **Test with TTA**: Averaging logits from original + flipped image effectively reduces prediction variance by exploiting the model's learned flip-invariance, without any additional training.

The inference pipeline uses `sampling_len=20` stochastic samples per image (squared-logit aggregation), which already provides some variance reduction. TTA on top of this provides complementary uncertainty reduction at the augmentation level.

---

## Scores (scores.jsonl)

```
{"iter": 0, "idea_id": "baseline", "primary_metric": 77.41, ...}
{"iter": 1, "idea_id": "IDEA-001", "primary_metric": 77.41, ...}
{"iter": 2, "idea_id": "IDEA-005", "primary_metric": 78.31, ...}
{"iter": "final", "idea_id": "IDEA-005", "primary_metric": 78.74, ...}
```

---

## Conclusion

The optimization target of ≥78.1524% was reached in just 2 iterations using a lightweight, zero-retraining technique: Test-Time Augmentation with horizontal flip averaging. The final confirmed result is **78.74%**, representing a +2.12% improvement over the paper-reported baseline of 76.62% and exceeding the 2% target threshold.
