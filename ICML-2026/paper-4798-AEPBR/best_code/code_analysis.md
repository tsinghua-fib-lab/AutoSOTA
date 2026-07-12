# Code Analysis — Paper 4798 SOTA Optimization

## Evaluation Path
- **Entry point**: `/repo/experiments/section_4_4_partial_escnn/experiments/medical_mnist2d.py`
- **Model**: `PenalizedSteerableApprox3DResnet` from `/repo/experiments/section_4_4_partial_escnn/networks/`
- **Dataset**: OrganMNIST3D (3D medical image classification, 11 classes)
- **Group**: O(3) with projection-based approximate equivariance regularization
- **Eval command** (in-container):
  ```
  cd /repo/experiments/section_4_4_partial_escnn && DATA_ROOT=/autosota_cache/datasets python -u experiments/medical_mnist2d.py --epochs 100 -c 1 --batch_size 32 --nr_workers 0 -kl 0 -kl_U 0 -align 0 --approx --lr 1e-4 -d organmnist3d --group O3 --conv_wd 1e-3 --basic_wd 1e-3 --data-root /autosota_cache/datasets --iterations 0
  ```

## Train/Inference Path
- **Training**: `train_nn()` at line 262 — full training loop with AMP, gradient scaling, projection penalty
- **Testing**: `test_nn()` at line 646 — evaluation with MedMNIST evaluator
- **Network**: `create_network()` at line 214 — creates PenalizedSteerableApprox3DResnet.from_group("O3", ...)
- **Projection penalty**: In `network_new.py`, computed via `projection_penalty()` — Q matrices stored as CPU attributes, chunked computation

## Config Path
- **Function**: `create_config()` at line 735
- **Arguments**: Defined at line 880-1015 (argparse)
- **Key params**: channels (-c), lr, conv_wd, basic_wd, group, approx, proj_penalty_every, dataset

## Metric Parser
- Stdout format: `Epoch N: test loss X.XXXX test acc:Y.YYYY val loss X.XXXX val acc:Y.YYYY`
- Primary metric: **Test Accuracy** (maximum across all 100 epochs)
- Parsed from `test_nn()` output, which prints `dataset metrics` dict from MedMNIST Evaluator

## Reusable Resources
- **No `/paper_data` mount** — nothing to reuse
- **Datasets**: MedMNIST auto-downloaded to `/autosota_cache/datasets/medmnist/`
- **Basis caches**: `/tmp/partial_escnn_q_cache` → `/autosota_cache/tmp/partial_escnn_q_cache`

## Risky Files (avoid modifying)
- `test_nn()` function — metric computation must stay unchanged
- MedMNIST library — dataset splits/labels
- `calc_accuracy()` in util.py — metric definition

## Safe Modification Targets
- `train_nn()` in `medical_mnist2d.py` — training loop modifications (schedulers, EMA, best-model logic)
- `create_network()` in `medical_mnist2d.py` — model config
- `projection_penalty()` in `network_new.py` — penalty computation
- `PenalizedDenseR3Conv` in `network_new.py` — per-layer penalty

## Key Bugs Found
1. **Line 420**: `if "organ" not in config.dataset:` skips best-model loading for organ datasets
   - This is likely a debugging artifact — organ datasets use final-epoch model instead of best-validation model
   - Small datasets like Organ (1,293 training samples) are prone to overfitting in final epochs
   - Fix: change condition to `if True:` or remove it entirely

## Baseline
- Commit: `a68c826` (tagged `_baseline`)
- Iter 0 commit: `a95a24e`
- Baseline Test Accuracy: 0.9272

## Container
- GPUs: 2x NVIDIA A100-SXM4-80GB (devices 0,1)
- CUDA: 13.0
- Image: autosota/paper-4798:reproduced

## Optimization Constraints
- Do NOT modify: test data, labels, splits, metric computation, scoring scripts
- Do NOT hard-code outputs
- All changes must be auditable via git commits
