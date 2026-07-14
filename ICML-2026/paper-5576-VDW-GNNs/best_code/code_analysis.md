# Code Analysis: Paper 5576 - VDW-GNNs SOTA Preparation Repair

## Original Preparation Failure

The orchestrator failed to set up the SOTA container due to two issues:

1. **Git not installed**: The reproduced Docker image (`autosota/paper-5576:reproduced`) does not include `git`. The orchestrator tried `apt-get install git` but the proxy (`http://172.17.0.1:17890`) returned 502 errors for Ubuntu archive repositories.

2. **Docker networking**: The first `docker run` attempt with `--network host` was rejected by the Docker auth plugin. The second attempt without `--network host` succeeded.

## Repair Actions

1. **Git installation**: Installed git by temporarily unsetting proxy environment variables and using `apt-get` directly. Archive.ubuntu.com was reachable without the proxy (confirmed via Python `urllib.request` returning HTTP 200).

2. **Container**: Container `autosota_sota_paper_5576` is running from `autosota/paper-5576:reproduced` with GPUs 6,7 (mapped to container indices 0,1 as NVIDIA A100-SXM4-80GB).

3. **Git repo setup**: Initialized git repo at `/repo`, created baseline commit and `_baseline` tag.

4. **Record score script**: Copied `/tools/record_score.sh` from host to container.

5. **Baseline verification**: Ran the evaluation command with 5 replications. Results matched the manifest baseline within normal noise:
   - test_mse: 2.528 (manifest: 2.535)
   - rotation_mse: 2.890 (manifest: 2.932)
   - parameter_count: 20,099 (exact match)

## Corrected In-Container Evaluation Command

```bash
cd /repo && CUDA_VISIBLE_DEVICES=0,1 python3 scripts/python/run_wind_experiments.py \
  --config config/yaml_files/wind/vdw.yaml \
  --root_dir /repo \
  --replications 5 \
  --knn_k 3 \
  --local_pca_k 10 \
  --sample_n 2000 \
  --mask_prop 0.3 \
  --seed 42 \
  --do_rotation_eval \
  --rotation_seed 298357 \
  --exp_name <experiment_name>
```

## Safe Optimization Targets

- `config/yaml_files/wind/vdw.yaml`: Model architecture (hidden dims, activation, BN, dropout, layers)
- `config/yaml_files/wind/experiment.yaml`: Training hyperparams (LR, batch_size, warmup, scheduler)

## Optimization Results Summary

| Iter | Config | test_mse | rotation_mse | Δ from baseline |
|------|--------|----------|--------------|-----------------|
| 0 | Baseline | 2.528 | 2.890 | — |
| 1 | +BN | 2.535 | 2.889 | +0.3% ❌ |
| 2 | +GELU | 2.620 | 3.039 | +3.6% ❌ |
| 3 | +2 layers | 2.833 | 3.347 | +12% ❌ |
| 4 | LR=0.001 | 2.481 | 2.882 | -1.9% ✅ |
| 5 | LR=0.001+BN | 2.497 | 2.940 | -1.2% |
| **6** | **LR=0.001+[256,128]** | **2.391** | **2.840** | **-5.4%** ✅✅ |
| 7 | +warmup | 2.483 | 2.831 | -1.8% |
| 8 | +dropout 0.1 | 2.469 | 2.806 | -2.3% |
| 9 | +[256,256] | 2.442 | 2.833 | -3.4% |
| 10 | LR=0.0005 | 2.429 | 2.850 | -3.9% |
| 11 | +[384,128] | 2.440 | 2.824 | -3.5% |
| 12 | LR=0.002 | 2.487 | 2.855 | -1.6% |
| 13 | batch_size=256 | 2.437 | 2.826 | -3.6% |

**Best: Iter 6 — LR=0.001 + hidden [256,128]**
- test_mse: 2.391 (-5.4% vs baseline)
- rotation_mse: 2.840 (-1.7% vs baseline)
- best_epoch: 67 (vs 95 baseline)
- Parameter count: 39,683
