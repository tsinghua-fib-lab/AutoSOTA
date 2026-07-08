# Code Analysis: Paper 888 - BTSP-MSR Decentralized Learning

## Preparation Failure Diagnosis

**Root Cause**: The preprocessed CIFAR-10 numpy arrays were missing from `/datasets/cifar10_preprocessed/`. The directory existed but was empty because:
1. The NFS-mounted `/datasets` directory is shared across containers
2. The original reproduction's data files were either not persisted or were cleaned up
3. The SOTA container started fresh without the preprocessed data

**Repair**: Downloaded CIFAR-10 dataset from HuggingFace mirror (hf-mirror.com) as parquet files via the `uoft-cs/cifar10` dataset, decoded PIL images, and saved as uint8 numpy arrays:
- `train_images.npy`: (50000, 3, 32, 32) uint8
- `train_labels.npy`: (50000,) int64
- `test_images.npy`: (10000, 3, 32, 32) uint8
- `test_labels.npy`: (10000,) int64

The data format matches the training script's expectations (uint8, channel-first, CIFAR-10 standard normalization applied on-the-fly).

## Corrected Evaluation Command

```bash
cd /repo && python3 train_decentralized.py \
  --n_rounds 200 \
  --eval_every 20 \
  --gpu 0 \
  --output_dir /repo/results \
  --save_every 200 \
  --data_dir /datasets/cifar10_preprocessed \
  --assignment_file /repo/output_exp_ebone.json
```

Note: `--n_rounds` can be adjusted. 200 rounds is used for quick SOTA iteration (~50 min per eval). The original 5000-round command is valid but takes ~21 hours.

## Baseline Reproduction Evidence

Smoke test at 2 rounds confirmed:
- Data loads correctly (50000 train, 10000 test images on GPU)
- Dirichlet(alpha=0.1) split produces non-IID distribution across 87 nodes
- BTSP-MSR BCD: 86.53 ms/round
- Training runs without errors
- Initial accuracy ~0.098 (near random, expected for extreme non-IID)

## Repository State

- Git repo at `/repo` with `_baseline` tag at commit `9691c7e`
- Baseline code: standard DSGD with LeNet-5, exponential graph mixing, BTSP-MSR assignment
- `apply_idea.py`: script to apply/undo optimization ideas
- Backup at `train_decentralized.py.baseline`

## Safe Optimization Targets

The training script is well-structured for modification:
1. **Optimizer/loss** (lines 252-256): SGD, CrossEntropyLoss, AMP scaler
2. **Data augmentation** (lines 88-122): `gpu_augment()` function
3. **Batch sampling** (lines 288-295): Random sampling from per-node indices
4. **Learning rate schedule**: Not present, can be added
5. **Model architecture** (lines 24-41): LeNet-5 - could be replaced with ResNet-18
6. **Mixing matrix** (lines 48-54): Exponential graph, could try different topologies
7. **Local updates** (lines 282-308): Sequential per-node SGD steps
