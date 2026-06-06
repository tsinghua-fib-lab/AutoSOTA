# Code Analysis: STaRFormer (PAM Classification)

## Pipeline Summary

1. **Input**: PAM dataset — 17 features, sequences up to ~241 time steps, 8 activity classes
2. **Embedding**: Linear(17 → 32) + Tanh, then prepend learnable CLS token → shape [242, BS, 32]
3. **Positional Encoding**: Sinusoidal PE added to embeddings
4. **Transformer Encoder**: 3 layers, 8 heads, d_model=32, feedforward=128, dropout=0.1017
5. **DAReM Masking**: In train/val modes, runs attention rollout → selects top-k attended regions → masks them to 0
6. **Second Forward Pass**: Runs the masked sequence through the same encoder → `embedding_masked`
7. **Contrastive Loss**: `SemiSupervisedCL` combines batch-wise InfoNCE + class-wise loss
8. **Total Loss**: `loss = CE_loss + lambda_cl * contrastive_loss`
9. **Test Mode**: Only uses unmasked path, CE loss only
10. **Output**: CLS token from encoder → Linear(32 → 8) with SELU activation

## Key Source Files

| File | Purpose |
|------|---------|
| `src/models/transformer.py` | STaRFormer: embedding, encoder, DAReM masking logic |
| `src/nn/transformer.py` | Custom TransformerEncoder/Layer with attention weight caching |
| `src/nn/loss.py` | DAReMLoss, DAReMContrastiveLoss, SemiSupervisedCL |
| `src/runtime/centralized.py` | CentralizedModel (Lightning): train/val/test steps |
| `src/runtime/callbacks.py` | PAMMetricLoggerCallback: F1/precision/recall computation |
| `src/models/global_modules.py` | SequenceModel wrapper: sequence_model + output_head |
| `src/models/output_heads.py` | ClassificationHead: Linear(d_model → d_out) |
| `configs/experiment/benchmark/classification/pam.yaml` | Full experiment config |
| `configs/callbacks/early_stop/pam.yaml` | early_stop monitors val/loss |
| `configs/callbacks/model_ckpt/pam.yaml` | checkpoint monitors val/acc |
| `scripts/training/train.py` | Entry point, uses Hydra config |

## Evaluation Procedure

- Command: `CUDA_VISIBLE_DEVICES=0 python scripts/training/train.py +experiment=benchmark/classification/pam.yaml logger.name=tensorboard`
- Output: Lightning test table (printed to stdout)
- Parse from `test/f1`, `test/acc`, `test/precision`, `test/recall`, `test/loss_ce`
- Estimated runtime: ~8-15 minutes per run (usually 50-80 epochs with early stopping at patience=30)
- Metrics computed by PAMMetricLoggerCallback using sklearn's f1_score/precision_score/recall_score with average='macro'

## Optimization Levers

| Parameter | Current Value | Location | Type | Notes |
|-----------|---------------|----------|------|-------|
| loss.lambda_cl | 0.6001 | pam.yaml | float | weight for contrastive loss |
| loss.temp | 0.5 | pam.yaml | float | temperature for CL similarity |
| loss.lambda_fuse_cl | 0.5 | pam.yaml | float | fusion weight for CL components |
| model.sequence_model.n_head | 8 | pam.yaml | int | attention heads |
| model.sequence_model.d_model | 32 | pam.yaml | int | transformer hidden dim |
| model.sequence_model.dropout | 0.1017 | pam.yaml | float | dropout rate |
| model.sequence_model.dim_feedforward | 128 | pam.yaml | int | FFN dimension |
| model.sequence_model.num_encoder_layers | 3 | pam.yaml | int | transformer depth |
| model.sequence_model.mask_threshold | 0.2078 | pam.yaml | float | fraction of tokens masked |
| model.sequence_model.mask_region_bound | 0.1 | pam.yaml | float | region boundary size |
| model.sequence_model.ratio_highest_attention | 0.3 | pam.yaml | float | fraction of tokens to use for masking regions |
| training.learning_rate | 0.00760 | pam.yaml | float | Adam LR |
| training.batch_size | 256 | pam.yaml | int | batch size |
| training.epochs | 300 | pam.yaml | int | max epochs |
| model.output_head.activation | selu | pam.yaml | str | classifier head activation |
| optimizer.beta1 | 0.857 | pam.yaml | float | Adam beta1 |
| optimizer.beta2 | 0.939 | pam.yaml | float | Adam beta2 |
| optimizer.weight_decay | 0.000410 | pam.yaml | float | L2 regularization |
| callbacks.lr_scheduler.factor | 0.8 | pam.yaml | float | LR reduction factor |
| callbacks.lr_scheduler.patience | 8 | pam.yaml | int | epochs before LR reduction |
| callbacks.early_stop.patience | 30 | pam.yaml (via early_stop/pam.yaml) | int | early stopping patience |
| callbacks.early_stop.monitor | val/loss | early_stop/pam.yaml | str | what to monitor |
| callbacks.model_ckpt.monitor | val/acc | model_ckpt/pam.yaml | str | checkpoint metric |

## Critical Observations

1. **Checkpoint saves based on val/acc**, but we care about **test_f1**. Changing to val/f1 monitoring could help.
2. **Early stopping on val/loss** — this means training stops when val/loss stagnates, even if val/f1 could improve.
3. **Small model**: d_model=32 with 3 layers is quite small for 8-class multivariate TS with 241 timesteps.
4. **Semi-supervised CL**: Currently uses both unmasked and masked embeddings for CL.
5. **DAReM** uses attention rollout to find important regions, then masks them to create a challenging view.
6. **Run variance**: Actual baseline was 0.9563 f1, lower than paper's 0.9755. Paper may have reported best-of-multiple runs.

## Hard Constraints / Red Lines (DO NOT CHANGE)

- [ ] **Eval metric parameters**: f1 is computed as macro average — do not change this
- [ ] **Evaluation script logic**: eval command, script, metric computation must not be modified
- [ ] **Algorithm output integrity**: never hard-code or fabricate model predictions
- [ ] **Metric-dimension trade-off**: do not sacrifice acc/precision/recall to inflate f1
- [ ] **Dataset integrity**: train/test split preserved; no test data in training
- [ ] **Pretrained weights**: no pretrained weights used (trained from scratch) — maintain this
- [ ] **Core method**: STaRFormer with DAReM masking must be preserved as core approach

## Initial Hypotheses

1. Changing checkpoint monitor from val/acc to val/f1 should give the model directly optimized for the target metric
2. Increasing d_model (32→64) with more layers (3→4) should improve representation quality
3. Label smoothing (ε=0.1) in CE loss should reduce overfitting
4. Supervised contrastive learning (SupCon) could improve class discrimination
5. Cosine annealing LR schedule may give better optimization dynamics than ReduceLROnPlateau
