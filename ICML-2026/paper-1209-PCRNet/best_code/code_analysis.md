# PCRNet Code Analysis — Paper 1209 SOTA Optimization

## Evaluation Path
- **Entry**: `python3 main.py`
- **Config**: `/repo/config.py` — `dataset="DTU"`, `time_len=1`, `data_document_path="/datasets/DTU"`
- **Flow**: `main.py` → `main_DTU()` per-subject loop → `initiate()` → `train_model()` → evaluate on test set
- **Output**: stdout prints `avg_acc: X.XXXX`; log `/repo/log/result.log` last line `The average accuracy of DTU_1s avg_acc:X.XXXX std:X.XXXX`

## Train/Inference Path
- `main_DTU()`: Loads .mat data, CSP preprocessing (within-subject), sliding window (1s, 0.5 overlap), 8:1:1 split
- `initiate()`: Creates PCRNet model, AdamW optimizer (lr=4e-3, wd=3e-4), CrossEntropyLoss, CosineAnnealingWarmRestarts (T_0=10, T_mult=2, eta_min=3e-4)
- `train_model()`: Per-epoch training + validation, early stopping (patience 10 hardcoded, NOT args.patience), saves/loads best model
- 18 subjects processed sequentially (S1–S18)

## Config Path
- `/repo/config.py` — dataset, time_len, people_number, data_document_path

## Metric Parser
- `avg_acc` from `np.mean(all_test_acc)` printed to stdout
- `result.log` contains per-subject and final aggregate line
- `record_score.sh` parses stdout

## Key Bugs / Issues
1. **Early stopping patience hardcoded to 10** (line 162 in `train_model`), not `args.patience=15`
2. **scheduler.step() called per-batch** inside `train()`, should be per-epoch
3. **No DataLoader shuffle** — `shuffle=True` missing on train_loader (line 379)
4. **evaluate() calls optimizer.zero_grad()** unnecessarily (line 119)
5. **Metric denominator wrong** — uses `num_batches * args.batch_size` instead of `len(loader.dataset)` (line 128-129)
6. **torch.save(model)** instead of state_dict in `save_model()` — prevents architecture changes
7. **No min_delta** in early stopping — any tiny improvement resets patience

## Safe Modification Targets
- `main.py` `initiate()`: optimizer config, scheduler config, loss function
- `main.py` `train_model()`: training loop, early stopping logic, SWA
- `main.py` `main_DTU()`: DataLoader config, augmentation
- `model.py`: Architecture changes (kernel sizes, dimensions, expansion factors)
- `data_process.py`: Data preprocessing (no augmentation currently)
- `utils.py`: Save/load model pattern

## Risky Files (DO NOT MODIFY)
- `data_process.py` `within_data()`: Controls train/test split — changing this changes evaluation protocol
- `data_process.py` `sliding_window_csp()`: Window generation protocol
- `config.py`: Dataset config (but adding new config parameters is OK)
- Test data labels, evaluation metric computation

## Reusable Resources
- `/datasets/DTU/`: 18 subject .mat files (S1–S18_data_preproc.mat)
- `/repo/pre_trained_models/`: Baseline trained models (S1–S18.pt)

## Setup Notes
- Container: `autosota/paper-1209:reproduced` (pytorch 2.1.0+cu121)
- GPU: `0,1` (uses CUDA_VISIBLE_DEVICES=0 in code)
- Dependencies: torch, mne, scikit-learn, timm, einops, dotmap, scipy, pandas, tqdm
