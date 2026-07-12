# Shared helpers

Bash and Python utilities used across the per-section experiments.

| File | Purpose |
|---|---|
| `common.sh` | GPU auto-selection, port-availability checks, env initialisation |
| `common_eb.sh` | GPU model detection (`detect_gpu_name`), calibration resolver (`resolve_calibration`), scheduler env setup. Sources `common.sh` automatically. |
| `dataset_utils.py` | Loaders for ShareGPT / LongBench / WildChat / NuminaMath |
| `export_dataset.py` | Convert HF datasets to the prompt JSONL format the runner expects |

Scripts in `sec*/` `source` these via `${SCRIPT_DIR}/../common/common*.sh`
(one level deep) or `${SCRIPT_DIR}/../../common/common*.sh` (two levels deep,
i.e. `multiturn/`, `non_stationary/`, `traffic/`).
