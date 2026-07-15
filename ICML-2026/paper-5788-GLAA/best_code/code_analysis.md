# Code Analysis — xFedAlign MNIST Non-IID (Paper 5788)

## Evaluation Path

- **Entry point**: `vision-experiments/mnist-non-iid.py` `__main__`
- **Command**: `python3 vision-experiments/mnist-non-iid.py`
- **Order**: `run_plain_fl` (BL-A) → `run_local_posthoc` (BL-B) → `run_server_summary` (BL-C) → `run_interpretable_only` (BL-D) → `run_xfl_from_blA` (xFL)
- **Seeds**: 5 (2025–2029)
- **Data**: MNIST via torchvision, auto-downloads to `./data`
- **Non-IID split**: Dirichlet(alpha=0.1), 8 clients, deterministic per seed
- **Output**: `mnist_noniid_outputs_fixed_5seeds/summary_mean_std.json`

## Metric Parser

All metrics computed in `run_once()`:
- **Accuracy**: `eval_accuracy()` — standard classification accuracy on test set
- **EDI**: `compute_edi()` — Jensen-Shannon divergence between per-client maps and reference
  - xFL uses `ref_from_Pi(Pi_X)` as reference
- **Deletion/Insertion AUC**: `deletion_insertion_auc()` on 128 test samples
  - xFL uses `build_imp_map(xFL, ...)` with surr_weights and xfl_gamma

## Config Path

Two dataclasses:
- `FLConfig`: n_clients=8, rounds=15, local_epochs=1, lr_cnn=0.01, batch_size=64, alpha_dirichlet=0.1
- `XFLConfig`: topk=256, quant_bits=8, clip_radius=5.5, dp_sigma=0.10, temperature=3.0,
  surrogate_epochs=1, beta_align_final=0.35, align_warmup_rounds=4, l1_lambda=8e-5, sharpen_gamma=1.8,
  xfl_rounds=8, gamma_start=1.0, gamma_end=1.8, temp_start=5.0, temp_end=1.5

## Key Functions

| Function | Lines | Role | Safe to modify |
|---|---|---|---|
| `SparseLinearSurrogate` | 93-103 | Surrogate model (784→10 linear) | Yes — model architecture |
| `fit_surrogate_teacher_student` | 307-322 | KD training of surrogate | Yes — loss function, init |
| `surrogate_to_artifact` | 325-341 | W→artifact (topk, clip, quantize, DP) | Yes — artifact pipeline |
| `robust_aggregate_artifacts` | 344-345 | Median aggregate | No — core algorithm |
| `run_xfl_from_blA` | 548-595 | Multi-round artifact mixing loop | Yes — Pi init, annealing |
| `build_imp_map` | 646-686 | Build importance maps for fidelity eval | Yes — map construction |
| `run_once` | 700+ | One full seed evaluation | No — metric definitions |
| `__main__` | 830+ | Main loop over seeds | No — evaluation protocol |
| `compute_edi` | 618-625 | EDI via JSD | No — metric definition |
| `deletion_insertion_auc` | 143-168 | Fidelity AUC | No — metric definition |

## Safe Modification Targets

1. `fit_surrogate_teacher_student` (lines 304-322): Add MMD loss, IG warmup, multi-teacher KD
2. `SparseLinearSurrogate` (lines 93-103): Upgrade to 2-layer MLP
3. `XFLConfig` dataclass: Add new hyperparameter fields
4. `run_xfl_from_blA` (lines 548-595): Add Pi initialization, VQ codebook, SAE regularizer
5. `surrogate_to_artifact` (lines 325-341): Per-class adaptive topk
6. `build_imp_map` (lines 646-686): Fix instance-level map usage

## Risky Files (Do Not Modify)

- `compute_edi()`, `deletion_insertion_auc()`, `auc_area()` — metric definitions
- `dirichlet_split_noniid()` — data splitting
- `make_loaders_mnist_noniid()` — data loading
- `run_once()` — evaluation protocol
- `__main__` loop and summary computation — output format

## Current Changes from Original

The code at HEAD includes ALGO-2 (co-annealing schedule) already integrated:
- `xfl_rounds=8` (default, multi-round Pi refinement)
- `gamma_start=1.0 → gamma_end=1.8` (linear schedule)
- `temp_start=5.0 → temp_end=1.5` (linear schedule)
- Beta warmup over `align_warmup_rounds=4`
- Baseline EDI=0.07746 was recorded with these changes active

## Baseline (iter-0)

- Commit: 982f57e / d946327
- EDI: 0.07746
- Accuracy (xFL): 0.95778
- Deletion AUC (xFL): 0.19221
- Insertion AUC (xFL): 0.91171
