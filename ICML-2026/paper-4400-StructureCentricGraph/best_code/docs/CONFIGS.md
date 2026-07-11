# Configs

All scripts accept `--config` and optional command-line overrides such as `--device`, `--data-root`, `--output-dir`, and `--checkpoint`.

Important common fields:

- `seed`: random seed
- `device`: `cpu`, `cuda`, or a specific CUDA device such as `cuda:0`
- `data_root`: dataset directory
- `output_dir`: output directory
- `model`: geometric bases parameters
- `train`: optimizer and epoch settings
- `fewshot`: `k_shot`, `n_query`, and `n_runs`

The debug configs are intentionally small and are meant for smoke tests, not for paper-quality numbers.

