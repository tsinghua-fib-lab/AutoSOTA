# Code Analysis for Paper 3108 SOTA Optimization

## Evaluation Path
- **Entry point**: `/repo/reproduce.py`
- **Command**: `python3 reproduce.py --n_seeds 10 --asym_n 20000 --n_jobs 5`
- **Phases**: (1) run_seed for seeds 1..10, (2) run_asymptotic (single large-sample run), (3) compute_metrics
- **Output**: stdout prints R2 Score, MSE importance, ROC AUC per strategy (ensemble/sub-models)
- **Output files**: results/scores_*.csv, results/loco_*.csv, results/asympt_*.csv
- **Metric parser**: compute_metrics() prints mean/median for all three metrics from parsed CSV files

## Train/Inference Path
- Model: BaggingRegressor(MLPRegressor(hidden_layer_sizes=(64,32,8), max_iter=500, early_stopping=True), n_estimators=10)
- get_model() in ensemble_vim/simulation.py lines 108-152 builds the model
- get_sub_models() extracts model.estimators_ for per-sub-model evaluation
- Data: get_dataset("friedman1", n_samples=512, n_features=20, snr=1.0) from ensemble_vim/data.py
- snr parameter uses sklearn make_friedman1(noise=2/snr) -> effective SNR approx 6 at snr=1.0

## Config Path
- All relevant args in reproduce.py:parse_args() (lines 24-41)
- Model config in ensemble_vim/simulation.py:get_model() (lines 108-152)
- No external config files (pure argparse)

## Metric Parser
- **R2 Score**: compute_metrics() lines 193-220, reads from scores_*.csv, prints mean/median
- **MSE importance**: compute_metrics() lines 222-253, joins LOCO with asymptotic ground truth
- **ROC AUC**: compute_metrics() lines 255-280, binarizes asymptotic importance >1e-3
- All three metrics printed to stdout; values parsed from printed lines

## Paper Data
- No pre-downloaded data (paper_data not mounted)
- All data is synthetic via sklearn make_friedman1()

## Safe Modification Targets
1. ensemble_vim/simulation.py:get_model() - model architecture, hyperparameters
2. reproduce.py:parse_args() - default argument values
3. reproduce.py:run_seed() - training loop, prediction aggregation
4. New functions in reproduce.py - additional importance methods
5. reproduce.py:loco_one_fast() - importance estimation method

## Risky Files (do not modify)
- ensemble_vim/data.py - dataset generation (defines ground truth)
- reproduce.py:compute_metrics() - metric definitions
- /tools/record_score.sh - scoring protocol
- Test data splits, labels, scoring logic

## Red-Line Boundaries
- No changes to metric computation formulas
- No changes to test/train split logic
- No changes to data generation (Friedman 1 definition)
- No hard-coded predictions or metric values
- All optimization changes must be to model/training/ensemble architecture only
