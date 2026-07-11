# Code Analysis — Paper 3954: BNRDE on Simple rBergomi

## Evaluation Path
- Entry: `taming_the_ito_lyon/training/test_cli.py` → `run_test_from_run_dir()` → `run_test()`
- Key function: `experiment.run_test()` loads checkpoint, runs eval epoch, computes results
- Metric extraction: `get_rough_volatility_results()` in `results_gathering_fns.py:551-738`
  - Computes KS statistic per-batch at each valid time index using `scipy.stats.ks_2samp`
  - `eval_metric = float(np.median(ks_stats))` — median KS across time indices
  - Display value = raw × 100 (e.g., 0.0371 → 3.71)
  - For simple_rbergomi, also computes extra Itô diagnostic metrics

## Train/Inference Path
- Entry: `taming_the_ito_lyon/training/train_cli.py` → `experiment()`
- `experiment()` in `experiment.py`: builds runtime, creates model, trains with early stopping
- `run_train_epoch()` in `loops.py`: unconditional training loop with per-batch Brownian driver sampling
- Model forward: `jax.vmap(model)(control_values_b)` where model is `BNRDE` from `models/bnrde.py`

## Config Path
- Target config: `configs/simple_bergomi/bnrde.toml`
- Key params: optimizer=muon, lr=5e-3, weight_decay=1e-6, max_grad_norm=1.0, loss=sigker
- Model: hidden_size=32, vf_hidden_dim=32, mlp_depth=2, signature_depth=2, window_size=16
- Training: batch_size=512, epochs=100, early_stopping_patience=25, seed=1

## Metric Parser
- KS Score: raw value from `metrics.json` → `test.metric.value` (e.g., 0.0371)
- Training Time: from `metrics.json` → `timings.training_s` (e.g., 138.95)
- Data: 15000 paths × 129 steps, 80/10/10 train/val/test split

## Risky Files (do NOT modify)
- `taming_the_ito_lyon/training/results_gathering_fns.py` — KS computation and eval metrics
- `taming_the_ito_lyon/data/simple_rough_volatility.py` — data generation
- `/repo/data/rough_volatility/simple_rbergomi_data.npz` — test data
- `scipy.stats.ks_2samp` — scipy internals

## Safe Modification Targets
- `configs/simple_bergomi/bnrde.toml` — hyperparameters
- `taming_the_ito_lyon/training/losses.py` — loss functions (signature depth, kernel type)
- `taming_the_ito_lyon/training/factories.py` — optimizer creation, loss factory, gradient clipping
- `taming_the_ito_lyon/training/experiment.py` — training loop orchestration
- `taming_the_ito_lyon/training/loops.py` — batch iteration, logging
- `taming_the_ito_lyon/training/runtime.py` — batch construction
- `taming_the_ito_lyon/models/bnrde.py` — model architecture

## Baseline Metrics
- KS Score: 3.71 (raw 0.037109375)
- Training Time: 139.0s
- Best epoch: 6 (early stopped at epoch 32)
- Params: 6,465
