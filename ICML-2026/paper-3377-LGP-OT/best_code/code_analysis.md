# Code Analysis: Paper 3377 - LGP-OT SOTA Preparation Repair

## Original Preparation Failure

The orchestrator attempted to prepare the container `autosota_sota_paper_3377` from the reproduced Docker image `autosota/paper-3377:reproduced`. Two failures occurred:

1. **First attempt in reusable container `autosota_repro_paper_3377`**: Failed to install `git` via apt-get due to proxy issues (502 Bad Gateway from proxy at 172.17.0.1:17890).

2. **Second attempt with `--network host`**: Docker rejected `--network host` due to administrative policy (OPA authorization plugin).

3. **Third attempt without `--network host`**: Container started successfully, but apt-get still failed through the proxy for git installation. The `git` package was not pre-installed in the base `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` image.

## Repair Applied

1. **Container `autosota_sota_paper_3377`**: Successfully started from `autosota/paper-3377:reproduced` without `--network host`, using bridge networking.

2. **git installation**: Installed `git` via `apt-get` with `-o Acquire::http::Proxy=false -o Acquire::https::Proxy=false` to bypass the unreliable proxy. The container has direct outbound internet access via Docker bridge networking.

3. **Data path fix**: The script expects data at `../data/drosophila_embryonic/processed` relative to `scripts/`. The actual dataset resides at `/data/drosophila_embryonic/processed/`. Created a symlink: `/repo/data/drosophila_embryonic -> /data/drosophila_embryonic`.

4. **/tools/record_score.sh**: Copied from host at `/home/dataset-assist-0/chenxinyu/autosota-v2.5-sota-icml-16/auto_sota/agents/sota/scripts/record_score.sh` into container at `/tools/record_score.sh`.

5. **Git repository**: Initialized `/repo` as a git repository with baseline commit and `_baseline` tag.

6. **Scores directory**: Created `/autosota_artifacts/paper-3377/sota/` for scores.jsonl output.

## Corrected In-Container Evaluation Command

```bash
cd /repo && python3 scripts/LGPOT.py --data_name drosophila --split_type three_interpolation --seed 42
```

This runs correctly inside `autosota_sota_paper_3377` with the following requirements:
- Data at `/data/drosophila_embryonic/processed/` (symlinked to `/repo/data/drosophila_embryonic/`)
- 2x A100-80GB GPUs (devices 0,1)
- Python 3.10.13, PyTorch 2.13.0+cu130

## Baseline Verification

The repaired evaluation reproduces the manifest baseline metrics exactly:

| Metric | Manifest Baseline | Repaired Baseline | Match |
|--------|-------------------|-------------------|-------|
| W2_t4  | 26.28             | 26.28             | ✓     |
| W2_t6  | 28.33             | 28.33             | ✓     |
| W2_t8  | 30.74             | 30.74             | ✓     |

- Seed: 42
- Training: early-stopped at epoch 389/1250
- Training time: ~383 seconds
- Inference time: ~0.09 seconds
- Peak GPU memory: 0.23 GB

## Reusable Resources

- **Dataset**: Available at `/data/drosophila_embryonic/processed/` (3.8GB extracted). Files include:
  - `three_interpolation-count_data-hvg.csv` (~111MB)
  - `three_interpolation-var_genes_list.csv` (~34KB)
  - `subsample_meta_data.csv`, `subsample_meta_data_with_celltype.csv`
  - Other split types: `three_forecasting`, `remove_recovery`, `first_five`
  
- **Model**: DGBFGP with latent_dim=32, M=6 HS basis functions, decoder [50,50]
- **Key packages**: PyTorch 2.13.0, geomloss 0.3.1, POT 0.9.7, scanpy 1.11.5

## Safe Optimization Targets

All modifications should stay within the existing DGBFGP framework:
1. **Loss function** (`optim/loss_func.py`): Sinkhorn parameters (blur, scaling), batch size
2. **Training loop** (`model/running.py`): KL coefficient schedule, learning rate schedule, optimizer, batch construction, deterministic subsampling
3. **Model architecture** (`model/models.py`, `model/layer.py`, `scripts/utils.py`): Latent dim, M basis functions, decoder architecture, IWAE samples (k)
4. **Inference** (`scripts/LGPOT.py`): Multi-seed ensemble averaging

Red lines that must NOT be crossed:
- No changes to data loading, preprocessing, or train/test splits
- No changes to evaluation metrics or `globalEvaluation` function
- No changes to the output parsing format
- No hardcoding of metrics

## Known Issues and Edge Cases

1. **LaTeX dependency**: The script sets `matplotlib.rc(text, usetex=True)` but LaTeX is not installed. However, since no plots are actually generated (matplotlib is imported but never used to create figures), this does not cause runtime errors.

2. **Proxy unreliable**: The container proxy at 172.17.0.1:17890 returns 502 errors frequently. Any package installation should bypass the proxy.

3. **Single-GPU usage**: Despite having 2 GPUs assigned, the model only uses cuda:0. Multi-GPU could be leveraged for ensemble training (ALGO-6).

4. **Determinism**: `np.random.choice` at `model/running.py:48` uses global numpy RNG without epoch-specific seeding, adding nondeterminism.
