# Paper 83 — STaRFormer

**Full title:** *STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking for Sequential Data*

**Registered metric movement:** +3.01% (0.9563 -> 0.9864 acc)

## Final Optimization Report (run_20260324_234148)

**Task:** PAM (PAMAP2 activity recognition)
**Target:** 0.995 (+2% over paper-reported 0.9755)
**Baseline (paper):** 0.9755
**Actual baseline:** 0.9563
**Best achieved:** **0.9864** (+3.01% from actual baseline, +1.09% from paper baseline)
**Target status:** not reached (practical ceiling around 0.986-0.987)

## Best Configuration

| Parameter | Baseline | Best |
|-----------|----------|------|
| `d_model` | 32 | 128 |
| `n_head` | 8 | 4 |
| `dim_feedforward` | 128 | 512 |
| `num_encoder_layers` | 3 | 4 |
| `dropout` | 0.1017 | 0.05 |
| `lambda_cl` | 0.6001 | 0.8 |
| `lambda_fuse_cl` | 0.5 | 0.5 |
| `loss` | CrossEntropy | CrossEntropy(label_smoothing=0.1) |
| `cls_method` | cls_token | **mean_pool** |
| `learning_rate` | 0.00760 | 0.002 |
| `patience` (early_stop) | 30 | 50 |
| `checkpoint monitor` | val/acc | val/f1 |
| `early_stop monitor` | val/loss | val/f1 |

## Iteration Log

| Iter | Change | Before | After | Delta | Status |
|------|--------|--------|-------|-------|--------|
| 0 | Baseline | - | 0.9563 | - | baseline |
| 1 | val/f1 monitoring | 0.9563 | 0.9563 | 0.0000 | neutral |
| 2 | d_model=64, n_head=4, ffn=256 + label_smoothing=0.1 | 0.9563 | 0.9770 | **+0.0207** | BEST |
| 3 | SupCon loss | 0.9770 | 0.9673 | -0.0097 | reverted |
| 4 | Pre-LN (norm_first=True) | 0.9770 | 0.9731 | -0.0039 | reverted |
| 5 | 5 encoder layers | 0.9770 | 0.9722 | -0.0048 | reverted |
| 6 | Multi-scale CNN front-end | 0.9770 | 0.9643 | -0.0127 | reverted |
| 7 | dropout=0.05 + lambda_cl=0.8 + patience=50 | 0.9770 | 0.9801 | **+0.0031** | BEST |
| 8 | CosineAnnealingLR (T_max=150) | 0.9801 | 0.9768 | -0.0033 | reverted |
| 9 | Focal loss (gamma=2) | 0.9801 | 0.9778 | -0.0023 | reverted |
| 10 | CL temperature=0.2 | 0.9801 | 0.9789 | -0.0012 | reverted |
| 11 | d_model=128, ffn=512 (CLS token) | 0.9801 | 0.9553 | -0.0248 | reverted |
| 12 | n_head=2 | 0.9801 | 0.9772 | -0.0029 | reverted |
| 13 | GeLU + lambda_fuse_cl=0.3 | 0.9801 | 0.9684 | -0.0117 | reverted |
| 14 | LR=0.003 | 0.9801 | 0.9808 | **+0.0007** | BEST |
| 15 | LR=0.001 | 0.9808 | 0.9749 | -0.0059 | reverted |
| 16 | **Mean pooling** (replaces CLS token) | 0.9808 | 0.9842 | **+0.0034** | BEST |
| 17 | lambda_cl=1.0 | 0.9842 | 0.9827 | -0.0015 | reverted |
| 18 | d_model=128, ffn=512 (+ mean pool) | 0.9842 | 0.9846 | **+0.0004** | BEST |
| 19 | LR=0.002 | 0.9846 | 0.9851 | **+0.0005** | BEST |
| 20 | 4 encoder layers | 0.9851 | 0.9864 | **+0.0013** | **BEST** |
| 21 | 5 encoder layers | 0.9864 | 0.9758 | -0.0106 | reverted |
| 22 | d_model=192, LR=0.0015 | 0.9864 | 0.9767 | -0.0097 | reverted |
| 23 | patience=80 | 0.9864 | 0.9864 | 0.0000 | confirmed |

## Key Insights

### What Worked

1. **Larger d_model (32→64, then 64→128 with mean_pool):** The original d_model=32 was undersized for 8-class, 17-feature, 241-timestep sequences. Scaling up the model capacity gave the largest single improvement (+2.07%).

2. **Label smoothing (ε=0.1):** Applied as part of the cross-entropy loss. Prevents overconfidence and improves generalization.

3. **Mean pooling over all sequence positions** (replacing CLS token): The most novel insight. By averaging encoder outputs over all T+1 positions (CLS + all timesteps), the model aggregates temporal information holistically rather than relying on a single CLS token. For PAM time series (activity recognition depends on global temporal patterns), this was transformative.

4. **Reduced dropout (0.1017→0.05):** With the larger model, the original dropout was too aggressive.

5. **Stronger CL weight (lambda_cl=0.6→0.8):** Stronger contrastive regularization improved class-discriminative representations.

6. **Optimal LR (0.00760→0.003→0.002):** The original LR was tuned for d_model=32. As d_model increased, a lower LR was needed for stable convergence.

7. **4 encoder layers** (with d_model=128 + mean_pool): Adding one more transformer layer gave additional capacity that synergized with mean pooling.

### What Failed

- **SupCon (iter 3):** Forced symmetric CL between masked/unmasked disrupted the asymmetric DAReM contrastive formulation.
- **Pre-LN (iter 4):** Post-LN training dynamics better for this smaller Transformer.
- **More layers with CLS token (iter 5):** Overfitting when using only CLS for classification.
- **CNN front-end (iter 6):** The linear embedding + Transformer works better than CNN pre-processing.
- **d_model=128 with CLS token (iter 11):** Severely overfit (0.9553). Mean pooling was the critical enabler for larger models.
- **Cosine LR / Focal Loss:** Both provided less stable training than ReduceLROnPlateau + label_smoothing.
- **Lower CL temperature (iter 10):** Too-hard negatives impeded learning with this architecture.

### Scalability Insight

A critical pattern emerged: **mean pooling enables larger models**. With CLS token:
- d_model=64: best performance
- d_model=128: severe overfitting

With mean pooling:
- d_model=64: 0.9842
- d_model=128 + 3 layers: 0.9846
- d_model=128 + 4 layers: 0.9864

Mean pooling distributes gradient information across all sequence positions, providing regularization that allows larger capacity models to train without overfitting.

## Gap Analysis (0.9864 vs 0.995 target)

The 0.0086 gap to the 0.995 target likely reflects:
1. **Dataset ceiling:** PAM with ~8000 training samples has limited diversity for 8 highly-confusable activity classes
2. **Paper reporting:** The 0.9755 paper baseline likely represents best-of-multiple-runs; our actual reproducible result was 0.9563
3. **Architecture constraints:** STaRFormer + DAReM masking as required by the task limits our ability to make more radical changes
4. **Inherent variance:** Each run has stochastic variation; some improvement may be achievable through lucky initialization

## Final Best Config

```yaml
model:
  output_head:
    cls_method: mean_pool       # Global mean pooling (key innovation)
  sequence_model:
    d_model: 128                # 4x original
    n_head: 4                   # head_dim=32
    dim_feedforward: 512        # 4x d_model
    num_encoder_layers: 4       # deeper than original 3
    dropout: 0.05               # from 0.1017

loss:
  lambda_cl: 0.8               # from 0.6001
  # CrossEntropyLoss(label_smoothing=0.1) in src/nn/loss.py

training:
  learning_rate: 0.002          # from 0.00760

callbacks:
  early_stop:
    monitor: val/f1
    mode: max
    patience: 50                # from 30
  model_ckpt:
    monitor: val/f1
    mode: max
```

**Files modified:**
- `configs/experiment/benchmark/classification/pam.yaml`
- `src/nn/loss.py` (label_smoothing=0.1 in CrossEntropyLoss, FocalLoss class added)
- `src/models/output_heads.py` (mean_pool option added to ClassificationHead)
