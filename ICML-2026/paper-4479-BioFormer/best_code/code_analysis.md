# Code Analysis - BioFormer (Paper 4479)

## Evaluation Path
- `run.py` → `exp/exp_classification.py` → `Exp_Classification.test()`
- Test calls `vali()` which runs forward pass and computes sklearn metrics
- Metrics: Accuracy, Precision, Recall, F1, AUROC, AUPRC (macro averaging)

## Train/Inference Path
- **Training**: `run.py --is_training 1` → `Exp_Classification.train()` → `Exp_Classification.test()`
- **Inference-only**: `run.py --is_training 0` → `Exp_Classification.test(test=1)`
- Checkpoints saved to `./checkpoints/classification/APAVA-Indep/BioFormer/{setting}/checkpoint.pth`

## Config Path
- All config via `run.py` argparse arguments
- Key flags: `--use_FBD`, `--mag_learning`, `--phase_learning`, `--use_ASSLN`, `--augmentations`, `--swa`

## Metric Parser
- stdout format: `Test results --- Loss: X, Accuracy: Y, Precision: Y, Recall: Y, F1: Y, AUROC: Y, AUPRC: Y`
- Metrics computed in `vali()` using sklearn

## Key Files
- `exp/exp_classification.py` — training loop, validation, criterion selection (modifiable)
- `models/BioFormer.py` — model architecture (modifiable)
- `layers/Embed.py` — PyramidConvEmbedding with BatchNorm1d (modifiable)
- `layers/BioFormer_EncDec.py` — FBAM, encoder, LayerNorm (modifiable)
- `layers/Augmentation.py` — Jitter, Scale, Dropout augmentations (modifiable)
- `utils/losses.py` — MMD, CORAL losses (add FocalLoss here)
- `run.py` — CLI args (add new flags here)

## Reusable Resources
- APAVA data at `/repo/dataset/APAVA/` (Feature/*.npy + Label/label.npy)
- Baseline checkpoint at `/repo/checkpoints/classification/APAVA-Indep/BioFormer/classification_APAVA-Indep_BioFormer_APAVA_dm128_nh8_el6_dl1_seed41/checkpoint.pth`

## Safe Modification Targets
1. Loss function (`_select_criterion` in exp_classification.py)
2. LR schedule (training loop in exp_classification.py)
3. Normalization layers (Embed.py)
4. Augmentation parameters (run.py flags)
5. Training hyperparameters (epochs, patience, etc.)
6. Test-time augmentation (vali method)
7. SWA flag (already implemented, just needs --swa)

## Risky Changes
- Modifying eval metrics computation (red-line)
- Changing data splits or labels (red-line)
- Modifying test data (red-line)
