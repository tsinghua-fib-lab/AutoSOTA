# Optimization Idea Library: STaRFormer PAM Classification

Last updated: 2026-03-24

## Ideas

### IDEA-001: Change monitoring metric to val/f1 for checkpoint and early stopping
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Change `callbacks.model_ckpt.monitor` from `val/acc` to `val/f1` and `callbacks.early_stop.monitor` from `val/loss` to `val/f1`. This directly optimizes for the target metric rather than trying to optimize accuracy or loss as proxies. Modify `configs/callbacks/early_stop/pam.yaml` and `configs/callbacks/model_ckpt/pam.yaml`, or override in the experiment config. Also change early_stop.mode from min (for loss) to max (for f1).
- **Hypothesis**: When we optimize all training decisions around the target metric (test_f1), we should get a consistently higher test_f1 at the checkpoint that gets restored for testing.
- **Status**: DONE
- **Result**: test_f1=0.9563 (same as baseline). val/f1 monitoring alone did not help — epoch 26 was already best for val/f1=0.972.

### IDEA-002: Increase model capacity — larger d_model and more layers
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Increase `d_model` from 32 to 64, `dim_feedforward` from 128 to 256, `num_encoder_layers` from 3 to 4. PAM has 8 classes, 17 features, 241 timesteps — a larger model should better capture patterns.
- **Hypothesis**: d_model=32 is quite small. A larger representation space should improve test_f1.
- **Status**: DONE (combined with IDEA-004)
- **Result**: test_f1=0.9770 (+0.0207 from baseline). Combined with label smoothing. d_model=64, n_head=4, ffn=256, num_encoder_layers=3. This is the current best (iter 2). Note: n_head reduced to 4 to maintain head_dim=16. Deeper model (5 layers, iter 5) hurt — 3 layers optimal.

### IDEA-003: Supervised Contrastive Learning (SupCon) — replace semi-supervised CL
- **Type**: ALGO
- **Priority**: HIGH
- **Risk**: MEDIUM
- **Description**: Inspired by Khosla et al., 2020 (SupCon Loss). Replace the current `SemiSupervisedCL` module with a fully supervised contrastive loss. In the current implementation, the CL has a batch-wise component (treats diagonal as positives, same as SimCLR) and a class-wise component. The SupCon approach treats ALL samples from the same class as positives in the contrastive objective. This gives stronger class-discriminative representations. Modify `SemiSupervisedCL._forward_global_contrastive` to use SupCon: $L = \sum_{i} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \neq i} \exp(z_i \cdot z_a / \tau)}$ where $P(i)$ are all positives for sample $i$ across both masked and unmasked embeddings.
- **Hypothesis**: SupCon should produce better class-separated representations, directly improving F1.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-004: Label Smoothing in CrossEntropy Loss
- **Type**: CODE
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Replace `nn.CrossEntropyLoss()` with `nn.CrossEntropyLoss(label_smoothing=0.1)` in the DAReM loss. Label smoothing is a regularization technique that prevents the model from becoming overconfident and helps with generalization (inspired by Inception v3 training and many subsequent SOTA results). Modify `src/nn/loss.py` in `DAReMContrastiveLoss.__init__` and `DAReMLoss.__init__`.
- **Hypothesis**: Reduces overfitting, more robust predictions, ~+0.5-2% F1 improvement.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-005: Cosine Annealing LR Schedule with Warmup
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Replace ReduceLROnPlateau with a CosineAnnealingLR schedule (or CosineAnnealingWarmRestarts). The standard ReduceLROnPlateau can get stuck in local minima. Cosine annealing provides smooth LR cycles that help escape local optima. Add `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)` in `centralized.py:configure_optimizers`. Also enable in config.
- **Hypothesis**: Better optimization dynamics → higher final test_f1.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-006: Contrastive Loss with Dual-Encoder SupCon (masked + unmasked as separate views)
- **Type**: ALGO
- **Priority**: HIGH
- **Risk**: MEDIUM
- **Description**: Inspired by SimCLR (Chen et al., 2020) and SupCon (Khosla et al., 2020). Use both masked and unmasked embeddings as two explicit "views" of the same sample, then apply SupCon across all 2N samples (N unmasked + N masked embeddings). For each anchor, positives are: (1) the paired view of the same sample, (2) all same-class embeddings from both views. This creates a richer contrastive signal than the current approach. Modify `SemiSupervisedCL._forward_global_contrastive` to concatenate `unmasked_rd_norm` and `masked_rd_norm` into a 2N×D matrix, then compute SupCon with the combined positive mask.
- **Hypothesis**: More comprehensive contrastive signal → better class representations → higher F1.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-007: Focal Loss for hard examples
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Inspired by RetinaNet (Lin et al., 2017). Replace CrossEntropyLoss with Focal Loss: $FL(p_t) = -(1-p_t)^\gamma \log(p_t)$ with γ=2. Focal loss down-weights easy examples and focuses training on hard examples. For multiclass: compute softmax probs, then apply focal weighting. Implement as a custom nn.Module and substitute where CrossEntropyLoss is used.
- **Hypothesis**: PAM has 8 classes — some class boundaries may be harder to learn. Focal loss helps focus on these.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-008: Mixup augmentation for time series
- **Type**: ALGO
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: Inspired by Zhang et al., 2018 (Mixup). Apply mixup augmentation at the embedding level (after linear embedding and positional encoding). Mix pairs of sequences: $\tilde{x} = \lambda x_i + (1-\lambda) x_j$, $\tilde{y} = \lambda y_i + (1-\lambda) y_j$ with $\lambda \sim \text{Beta}(\alpha, \alpha)$, $\alpha=0.2$. Only apply during training. Modify `centralized.py:__shared_step_starformer` to apply mixup before passing to the model. Note: contrastive loss with mixed labels is tricky; apply mixup only to the CE loss component.
- **Hypothesis**: Better generalization by creating synthetic training examples between classes.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-009: Attention-based temporal pooling instead of CLS token
- **Type**: ALGO
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: Inspired by attention-weighted pooling in many BERT-style approaches. Replace the CLS token classification method with a learned soft attention pooling. Add a small attention module: `scores = linear(encoder_output).softmax(dim=0)`, then `pooled = (scores * encoder_output).sum(dim=0)`. Pass this pooled representation to the classifier head. This might provide a more informative aggregation than just the CLS token. Modify `SequenceModel.forward` and `STaRFormer.forward`.
- **Hypothesis**: CLS token may not capture all relevant information. Attention pooling aggregates the full sequence.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-010: Increase contrastive loss temperature annealing
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Inspired by SimCLR (Chen et al., 2020) which shows temperature has a significant effect on contrastive learning quality. Implement a temperature annealing schedule: start at temp=0.5 and anneal to temp=0.1 over training. Modify `DAReMContrastiveLoss` to pass a decreasing temperature to `SemiSupervisedCL`. Add a callback or modify the training step to update temperature. Lower temperature creates harder negative examples as training progresses.
- **Hypothesis**: Adaptive temperature helps the contrastive loss focus on harder negatives as representations improve.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-011: Norm-first (Pre-LN) Transformer
- **Type**: ALGO
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Inspired by "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020) and GPT-2. Change `norm_first=False` (Post-LN) to `norm_first=True` (Pre-LN) in the TransformerEncoderLayer. Pre-LN Transformers are more stable to train and often converge faster and to better solutions than Post-LN. Modify the model config.
- **Hypothesis**: Pre-LN training dynamics are more stable → better final performance.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-012: Prototype learning augmentation — class centroid regularization
- **Type**: ALGO
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: Inspired by prototypical networks (Snell et al., 2017). During training, maintain exponential moving averaged class centroids in the embedding space. Add a regularization term that pulls each sample's embedding toward its class centroid: $L_{proto} = \sum_i |z_i - c_{y_i}|^2$. This creates compact, well-separated class clusters. Maintain centroids as buffers in the loss module and update them with EMA. Add to total loss with small weight (λ=0.1).
- **Hypothesis**: Prototype regularization creates more compact class clusters → better generalization.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-013: Reduce dropout for larger representation capacity
- **Type**: PARAM
- **Priority**: LOW
- **Risk**: LOW
- **Description**: Reduce dropout from 0.1017 to 0.05. With the current small model (d_model=32), dropout may be too aggressive. Lower dropout allows more of the learned representation to pass through.
- **Hypothesis**: Less aggressive dropout → better feature utilization.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-014: Increase early stopping patience
- **Type**: PARAM
- **Priority**: LOW
- **Risk**: LOW
- **Description**: Increase patience from 30 to 50. The model may need more epochs to converge, but early stopping is cutting it off.
- **Hypothesis**: Longer training → better convergence.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-015: Contrastive loss weight tuning
- **Type**: PARAM
- **Priority**: LOW
- **Risk**: LOW
- **Description**: Increase lambda_cl from 0.6 to 0.8 or 1.0. The contrastive loss helps learn task-aware representations. A higher weight might force better class-discriminative features.
- **Hypothesis**: More contrastive regularization → better representations.
- **Status**: PENDING
- **Result**: (fill in after execution)

### IDEA-016: SupCon + Label Smoothing + F1 checkpoint monitoring (combined)
- **Type**: ALGO
- **Priority**: HIGH
- **Risk**: MEDIUM
- **Description**: Combine IDEA-001 (F1 monitoring), IDEA-004 (label smoothing), and IDEA-003 (SupCon loss). These three changes work synergistically: F1 monitoring ensures the saved model is best for our metric, label smoothing prevents overconfidence, and SupCon builds better class-discriminative representations. Implement all three in one iteration.
- **Hypothesis**: Combined effect of all three improvements should give >+2% F1.
- **Status**: PENDING
- **Result**: (fill in after execution)

## Red Line Audit

| Idea | R1 | R2 | R3 | R4 | R5 | R6 | Decision |
|------|----|----|----|----|----|----|----|
| IDEA-001 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-002 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-003 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-004 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-005 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-006 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-007 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-008 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-009 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-010 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-011 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-012 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-013 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-014 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-015 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |
| IDEA-016 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | CLEARED |

## Iteration Log

| Iter | Idea | Type | Before | After | Delta | Status | Key Takeaway |
|------|------|------|--------|-------|-------|--------|--------------|
| 0 | baseline | - | - | 0.9563 | - | success | Actual baseline lower than paper (0.9755). Run variance. |
| 1 | IDEA-001 | CODE | 0.9563 | 0.9563 | 0.0000 | success | val/f1 monitoring alone had no effect. |
| 2 | IDEA-002+004 | PARAM+CODE | 0.9563 | **0.9770** | +0.0207 | **BEST** | Larger model (d_model=64,heads=4,ffn=256) + label_smoothing=0.1 is current best. |
| 3 | IDEA-003 | ALGO | 0.9770 | 0.9673 | -0.0097 | failed | SupCon hurt — original asymmetric CL better than 2N SupCon. Rolled back. |
| 4 | IDEA-011 | ALGO | 0.9770 | 0.9731 | -0.0039 | failed | Pre-LN slightly worse. Post-LN better for this model. Rolled back. |
| 5 | IDEA-002b | PARAM | 0.9770 | 0.9722 | -0.0048 | failed | Deeper model (5 layers) hurt. 3 layers is optimal. Rolled back. |
| 6 | IDEA-006-cnn | ALGO | 0.9770 | 0.9643 | -0.0127 | failed | CNN front-end hurt. Linear embedding is better. Rolled back. |
