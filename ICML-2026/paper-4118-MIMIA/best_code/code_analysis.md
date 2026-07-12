# Code Analysis for Paper 4118 SOTA Optimization

## Evaluation Path
- `eval_attack_full.py` — main evaluation script
- Loads pre-computed data from `random_cremad_5client_mix/Combined_Attack_Scores_Epoch_50.xlsx`
- Trains attack models (CrossAttnGapMIA, GapGatedMIA) and evaluates
- Outputs metrics: TPR@0.1%FPR, TPR@1%FPR, AUC, Balanced_Accuracy
- Saves to `evaluation_results_full.json`

## Metric Parser
- Metrics are printed to stdout as a table and saved to JSON
- TPR@FPR computed via `get_tpr_at_fpr()` which finds the highest TPR at/below target FPR
- AUC computed via sklearn `roc_auc_score`
- Balanced Accuracy = (TPR + TNR) / 2

## Config Path
- Hard-coded in `eval_attack_full.py`:
  - DATA_PATH: `random_cremad_5client_mix/Combined_Attack_Scores_Epoch_50.xlsx`
  - FEATURE_COLUMNS: 8 columns
  - BATCH_SIZE=1024, LR=1e-4, EPOCHS=200, SEED=422134
  - Model: CrossAttnGapMIA(temperature=0.5), GapGatedMIA

## Training Path
- `train_model()` in eval_attack_full.py
- BCEWithLogitsLoss with pos_weight
- AdamW optimizer, CosineAnnealingLR scheduler
- Best model selected by lowest test loss

## Data Characteristics
- 6639 samples (1339 members, 5300 non-members — ~20% members)
- 13 columns including LOSS, LOSS_BASED, GRAD_NORM (full/audio/visual), gap, metadata
- GRAD_NORM_audio_Score is object dtype (needs pd.to_numeric conversion)
- gap column exists but is not used as a feature
- LOSS_full_Score exists and is conditionally included

## Safe Modification Targets
- `eval_attack_full.py`: loss function, learning rate, scheduler, feature columns, calibration
- `attack_models.py`: model architectures (CrossAttnGapMIA, GapGatedMIA, etc.)
- `crient_function.py`: loss function implementations (FocalLoss, AsymmetricLoss)

## Risky Files (DO NOT MODIFY)
- `random_cremad_5client_mix/Combined_Attack_Scores_Epoch_50.xlsx` — pre-computed data
- Metric computation functions in `evaluate_model()`
- Data split logic (seed-based random split)
- Score recording scripts

## Reusable Resources
- Pre-computed attack scores at epoch 50 in Excel format
- 4 attack model architectures in attack_models.py
- AsymmetricLoss and FocalLoss in crient_function.py
- fast_dataset.py for dataloading
