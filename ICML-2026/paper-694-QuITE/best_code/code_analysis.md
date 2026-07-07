# Code Analysis — QuITE (Paper 694)

## Evaluation Path

**Entry point:** `train_forecasting.py`
- Parses args, builds Model, trains for max 1000 epochs with early stopping (patience)
- Per epoch: train on n_train_batches, validate on n_val_batches, test on n_test_batches when validation improves
- Early stops after patience epochs without validation MSE improvement
- Saves best model to `models/{dataset}_{history}history_{pred_window}pred_{model}_{mode}.pt`

**Metric computation:** `evaluation.py`
- `compute_all_losses()`: single training step, returns MSE loss, MSE, RMSE, MAE
- `evaluation()`: full pass, aggregates per-variable MSE sums, then averages across variables
- `compute_error()`: per-variable MSE/MAE/MAPE, reduces by mean (averaging across variables with available data)
- MSE is the training loss (used for backprop)

**Metric parser:** `extract_metrics.py`
- Parses log file for lines matching test best-epoch format
- Returns best (lowest MSE) epoch's metrics
- For the baseline seed=1: epoch=66, MSE=0.01852, MAE=0.07428

## Configuration

All config via argparse in `train_forecasting.py:29-68`:
- Model: --model, --mode, --irr_emb
- Architecture: --hid_dim, --nhead, --nlayer, --patch_size, --stride, --dropout
- Training: --epoch, --patience, --batch_size, --lr, --seed, --gpu
- Data: --dataset, --history, --pred_window, --quantization

## Architecture Map

### QuITE Embedding (`models/embeddings/quite.py`)
- `QuITEEmbedding`: query-based irregular time series embedding
  - Uses `SelfAttentionBlock` (from `models/modules.py`) with `MultiHeadedAttention`
  - `query_emb`: learnable query tokens (one per variable or patch)
  - `te`: `LearnableTE` (harmonic time embedding from `_base.py`)
  - `val_emb`: linear projection of scalar values to d_model
  - Forward: prepend query token to observation tokens, run masked self-attention, extract query output

### Backbone Models
- **PatchMixer** (current): `PatchMixerLayer` in `models/common.py` (depthwise conv + 1x1 conv)
- **PatchTST**: transformer encoder from `models/layers/transformer.py`
- **iTransformer/S-Mamba**: variate-level models from `models/layers/transformer.py`

### Cross-Attention Decoder (`models/quite.py:69-75`)
- `CrossAttentionFFNLayer` from `models/layers/transformer.py`
- Uses `FullAttention` from `models/layers/attention.py`
- Future-time queries attend to encoded patches/variables
- Output fed to 3-layer MLP decoder, producing scalar predictions

## Safe Modification Targets

### Low risk (no structural changes):
1. `train_forecasting.py:148` — optimizer/scheduler setup
2. `train_forecasting.py:156-204` — training loop (gradient clipping, logging)
3. `train_forecasting.py:29-68` — argparse (new flags)

### Medium risk (attention mechanism changes):
4. `models/embeddings/quite.py:40` — QuITEEmbedding attn parameterization
5. `models/modules.py:11-18` — Attention.forward (time kernel bias)
6. `models/layers/attention.py:15-26` — FullAttention.forward (sparse attention)

### Higher risk (architectural changes):
7. `models/quite.py:235` — forecasting decoder loop
8. `models/quite.py:_build_encoder` — backbone construction

## Red-Line Boundaries

- Do NOT change: evaluation.py metric computation, data loading in data/parse.py, dataset splits, PhysioNet test data
- Do NOT change: extract_metrics.py parsing logic or log format structure
- Do NOT hard-code: predictions, metric values, or dataset-specific magic numbers

## Reusable Artifacts

- No /paper_data mount
- /datasets cache: PhysioNet auto-downloaded
- /models cache: empty, available for checkpoint storage
- Baseline model: /repo/models/physionet_12history_36pred_patchmixer_quite.pt
