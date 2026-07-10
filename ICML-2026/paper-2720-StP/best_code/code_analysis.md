# Code Analysis: SOTA Preparation Repair — Paper 2720

## Preparation Failure Diagnosis

**Root cause**: The container filesystem (Docker overlay) had 0 bytes free (200G/200G used). This was a Docker storage pool exhaustion issue in the running environment, not a problem with the code or container image.

**Failure chain**:
1. The orchestrator tried to run `git init/add/commit/tag` inside the SOTA container
2. `git` was not installed, so it tried `apt-get update && apt-get install git`
3. `apt-get update` failed with "No space left on device"
4. The baseline commit/tag was never created
5. The orchestrator escalated to repair

**Repair actions**:
1. Cleaned `/var/lib/apt/lists/*` and `/var/cache/apt/archives/*.deb`
2. Ran `conda clean -a -y` to free conda cache
3. `apt-get update` succeeded after cleaning
4. `apt-get install git` succeeded (git 2.25.1)
5. Initialized git repo at `/repo`, created baseline commit with tag `_baseline`
6. Copied `record_score.sh` to `/tools/record_score.sh`
7. Ensured `/autosota_artifacts/paper-2720/sota/` is writable

## Corrected In-Container Evaluation Command

```bash
cd /repo/EDM && python3 eval_analyze.py --model_path outputs/edm_qm9_StP --n_samples 10000 --batch_size_gen 100
```

This is the same command as the manifest — it was already correct for in-container execution.

## Evaluation Output Format

The script `eval_analyze.py`:
1. Loads the EMA model from `outputs/edm_qm9_StP/generative_model_ema.npy`
2. Generates N samples using `qm9.sampling.sample()`
3. Calls `qm9.analyze.analyze_stability_for_molecules()` for stability metrics
4. Prints `stability_dict` with keys `mol_stable` (Molecule Stability) and `atm_stable` (Atom Stability)
5. Prints `Validity X.XXXX, Uniqueness: X.XXXX, Novelty: X.XXXX`
6. Appends results to `eval_log.txt` in the model directory

Metrics mapping:
- Atom Stability = `stability_dict[atm_stable]` × 100
- Molecule Stability = `stability_dict[mol_stable]` × 100
- Valid = Validity × 100
- Valid×Unique = Validity × Uniqueness / 100 (since both reported as decimals)

## Baseline Metrics (from Reproduction)

| Metric | Value | Paper CI |
|--------|-------|----------|
| Atom Stability | 98.84% | [98.80, 98.86] |
| Molecule Stability | 88.11% | [87.85, 88.29] |
| Valid | 93.98% | [94.33, 94.49] |
| Valid×Unique | 92.44% | [92.49, 92.77] |

Validity and Valid×Unique slightly below paper CIs, likely due to PyTorch 2.4.0+cu121 (paper used cu118) and QM9 from DeepChem SDF instead of figshare XYZ.

## Pre-trained Model

- Location: `/repo/EDM/outputs/edm_qm9_StP/`
- Files: `generative_model_ema.npy` (21.4 MB), `args.pickle` (1.1 KB)
- Epoch: 2381 (of 3000 total)
- Key config: `--StP`, `--n_layers 9`, `--nf 256`, `--diffusion_steps 1000`, `--ema_decay 0.9999`, `--normalize_factors [1, 4, 10]`, `--lr 1e-4`, `--batch_size 64`

## Safe Optimization Targets

### No-ReTraining Options (Inference-Only)
These can be tested without training, just modifying eval_analyze.py or sampling code:

1. **Temperature Tuning (ALGO-03)**: Modify `sample_p_xh_given_z0` in `equivariant_diffusion/en_diffusion.py:509` to divide logits by temperature before argmax. No training needed.

2. **Checkpoint Ensemble (ALGO-04)**: Weight-average the EMA model weights from NLL-best and Validity-best checkpoints. Need to examine what checkpoints exist.

3. **Self-Guidance (ALGO-05)**: Add self-guidance in the last K sampling steps in the `sample` method at `equivariant_diffusion/en_diffusion.py:800`.

### Training-Required Options
These require modifying training code and running training:

4. **Data Augmentation (CODE-01)**: Enable `--data_augmentation True` (code already exists in `train_test.py:36-37`).

5. **Normalize Factors Audit (CODE-04)**: Verify `normalize_factors` in the pre-trained checkpoint.

6. **Extended Training (ALGO-06)**: Continue training from epoch 2381 with cosine annealing.

7. **Per-Atom Loss Reweighting (ALGO-02)**: Modify `compute_loss` in `en_diffusion.py:583` to divide per-sample loss by `num_atoms`.

### Training From Scratch Options
8. **Learned Prior Scaling (ALGO-01)**: Replace fixed sigma_n lookup with MLP.
9. **Target Symmetrization (ALGO-07)**: Apply random rotations before loss computation.

## Key Files Map

| File | Purpose |
|------|---------|
| `EDM/main_qm9.py` | Training entry point, argparse |
| `EDM/train_test.py` | Training loop, `train_epoch`, test |
| `EDM/eval_analyze.py` | Evaluation entry point |
| `EDM/equivariant_diffusion/en_diffusion.py` | Diffusion model: forward, loss, sampling |
| `EDM/equivariant_diffusion/utils.py` | Noise sampling with StP (line 107-124) |
| `EDM/equivariant_diffusion/sigma_n_lists.py` | StP sigma_n lookup tables |
| `EDM/qm9/losses.py` | `compute_loss_and_nll` bridge |
| `EDM/qm9/sampling.py` | Molecule sampling wrapper |
| `EDM/qm9/analyze.py` | Stability/validity analysis |
| `EDM/qm9/dataset.py` | QM9 dataset loading |
| `EDM/qm9/models.py` | Model/optimizer construction |

## Hardware

- 2× A100-80GB GPUs (indices 6,7 on host → 0,1 in container)
- QM9 dataset at `/repo/processed_dataset/qm9/qm9/{train,valid,test}.npz`
- Cache mounts: `/autosota_cache`, `/datasets`, `/models` (NFS)
