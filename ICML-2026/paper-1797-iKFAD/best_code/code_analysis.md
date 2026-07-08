# Code Analysis for Paper 1797 — iKFAD GPT2-Nano Shakespeare

## Evaluation Path
- Script: nano/eval.py
- Flow: eval.py -> GPTTrainer(config) -> trainer.train() -> returns best_val_loss
- Output: Last stdout line is JSON with test_loss and seed fields
- Note: "test_loss" is actually best validation loss during training (established protocol)
- Config: Hardcoded in eval.py: iKFAD h=0.4941, alpha=2.1955, mu=4.55e-06, gamma=0.0

## Training Path
- Script: nano/train.py, Class: GPTTrainer
- Data: nano/data/shakespeare-char/ (train.bin, val.bin, meta.pkl), 65-char vocabulary
- Optimizer: iKFAD in optimizers/ikfad.py (A/B/C/D step pattern)

## Model (nano/model.py)
- GPT: n_layer=4, n_head=4, n_embd=128, block_size=64 -> ~796K params
- Dropout layers exist but configured to 0.0
- Bug line 207-208: embedding dropout applied then immediately overwritten

## Safe Modification Targets
1. Gradient clipping threshold (eval.py, train.py)
2. Dropout rate (eval.py)
3. iKFAD hyperparameters (eval.py)
4. iKFAD optimizer internals (optimizers/ikfad.py)
5. Training loop scheduling (train.py)
6. Model architecture (model.py) — careful

## Risky Files (do NOT modify)
- /tools/record_score.sh
- nano/data/shakespeare-char/ (dataset)
- model.py loss computation (line 219)
