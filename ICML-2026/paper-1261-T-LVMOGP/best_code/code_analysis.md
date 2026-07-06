# Code Analysis — Paper 1261: T-LVMOGP EEG Optimization

## Evaluation Path
- **Entry point:** `/repo/experiments/eeg/all_dkl_lvmogp_eeg.py`
- **Model:** `dkl_lvmogp_eeg` in `/repo/models/dkl_lvmogp_eeg.py`
- **Base class:** `dkl_lvmogp_base` in `/repo/models/dkl_lvmogp_base.py`
- **Config:** CLI arguments (argparse), no config file
- **Data:** `/repo/data/eeg.h5` (1492 train + 300 test, 7 EEG electrodes, 256 time points)

## Train/Inference Path
- Training: `model.train_lvmogp()` → per-epoch `_epoch_start_hook` → mini-batch ELBO loop
- ELBO: `model.elbo()` → `exp_log_lik` + KL_qU_pU + KL_qH_pH + correction_term
- Prediction: `model.predict_lvmogp_gaussian()` → MC sampling over qH → MSE/NLL computation
- Metrics parsed from stdout: `Total training time: ..., train_mse: ..., train_nll: ..., test_mse: ..., test_nll: ...`

## Metric Parser
- Parse line containing "Total training time" and metric values
- Format: `Total training time: <float>, train_mse: <float>, train_nll: <float>, test_mse: <float>, test_nll: <float>`

## Already Implemented Features (Not Enabled in Baseline)
1. **Tighter ELBO (ALGO-01):** `--tighter_elbo` flag. Code in `_setup_tighter_elbo_params()`, `correction_term()`. Uses `D_diag` cached from `variational_f_base()`.
2. **NGD (ALGO-02):** `--qU_type {standard,natural,tril-natural}`. Three variational distribution types: `Variational_inducing_dist`, `Natural_Variational_inducing_dist`, `TrilNatural_Variational_inducing_dist`. `--natural_lr` controls NGD learning rate.
3. **Spectral Norm:** `SpectralNormToConstant` with soft upper bound. `--spectral_norm --sn_ub <value>`.
4. **Gradient Clipping:** `--max_norm <value>` or `--max_norm None`.
5. **Model capacity:** `--M`, `--D_H`, `--num_blocks` all adjustable via CLI.

## Key Parameters
| Param | Baseline | Range |
|-------|----------|-------|
| D_H | 3 | 2-10 |
| M | 200 | 50-500 |
| num_blocks | 3 | 1-7 |
| sn_ub | 0.005 | 0.001-1.0 |
| lr | 0.01 | 0.001-0.1 |
| epochs | 1000 | 100-2000 |
| max_norm | None | None or positive float |
| qU_type | standard | standard, natural, tril-natural |
| tighter_elbo | False | True/False |

## Safe Modification Targets
- `/repo/experiments/eeg/all_dkl_lvmogp_eeg.py` — CLI args, optimizer setup
- `/repo/models/building_blocks/neural_nets.py` — ResNet architecture, SN schedule
- `/repo/models/dkl_lvmogp_base.py` — ELBO, training loop, LR schedule

## Risky Files (Do Not Modify)
- `/repo/utils/metrics.py` — metric computation
- `/repo/data/eeg.h5` — dataset
- `scores.jsonl` — only via `/tools/record_score.sh`

## Pre-existing Code Patterns
- The code already supports `tighter_elbo=True` with all the math (Titsias 2025 bound via `correction_term()`)
- The code already supports NGD via `--qU_type` and `--natural_lr`
- The `SpectralNormToConstant` class implements a soft UB: if sigma < sn_ub, weight unchanged; else rescale
- No LR scheduler is implemented — fixed LR for all epochs
- No gradient accumulation is implemented
- No early stopping is implemented
