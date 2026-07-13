# Code Analysis - Paper 3153 (DISSOLVR)

## Evaluation Path
- Main eval: `cd /repo/regime-i && python3 train.py`
- Data: `./data/train.csv` (7480 rows), `./data/test.csv` (832 rows) — 90/10 AqSolDB split
- Labels: `LogS` column in CSV
- Output: stdout prints "FINAL RESULTS (5 seeds)" block with RMSE±std, R²±std

## Train/Inference Path
- `train.py`: Loads data → featurizes → trains 5 CatBoost models (seeds 42, 101, 123, 456, 789) → reports mean±std
- `featurizer.py`: MoleculeFeaturizer class, 168 features (168 in practice, paper claims 176 due to RDKit version)
- Model saved: Best seed model → `./model/model.joblib`

## Config Path
- Hardcoded in `train.py`: iterations=10000, lr=0.02, depth=8, l2_leaf_reg=5
- Seeds: [42, 101, 123, 456, 789]
- No external config file

## Metric Parser
- Parse stdout: look for "FINAL RESULTS (5 seeds)" block
- Extract: `RMSE: X.XXXX ± X.XXXX` and `R²: X.XXXX ± X.XXXX`
- Primary: RMSE mean (lower), Guardrail: R² mean (higher, must stay > 0.857)

## Reusable Resources
- `/autosota_cache`: Available for caching
- `/datasets`: Cache mount for datasets
- `/models`: Cache mount for model weights
- No `/paper_data` mount

## Risky Files (do not modify)
- `data/train.csv`, `data/test.csv` — test data/splits
- `data/*.bak` — backup files
- Metric computation in sklearn (standard, trusted)

## Safe Modification Targets
1. `train.py` — Model params, ensemble logic, early stopping, loss function
2. `featurizer.py` — Feature set (uncomment Morgan/MACCS/AUTOCORR2D)
3. `train_hyperopt.py` — Hyperparameter search space

## Baseline Evidence
- iter-0 commit: `639af53`
- RMSE: 0.8189 ± 0.0037, R²: 0.8614 ± 0.0014
- Best seed (789): RMSE 0.8175
- Model params: CatBoostRegressor(iterations=10000, lr=0.02, depth=8, l2_leaf_reg=5)

## Scores File
- Path: `/autosota_artifacts/paper-3153/sota/scores.jsonl`
- Record via: `bash /tools/record_score.sh --scores <path> --iter N ...`
