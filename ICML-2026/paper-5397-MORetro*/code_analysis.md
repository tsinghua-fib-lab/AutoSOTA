# Code Analysis: Paper 5397 SOTA Preparation Repair

## Original Failure

### Root Cause

The SOTA preparation failed because the NVIDIA GPU assigned to the container (`GPU-b98bcdf1`, index 6) went into an unrecoverable error state during evaluation. This caused a CUDA kernel launch failure that crashed the running evaluation at molecule 28/190.

After the container was stopped, the CUDA driver's Unified Memory (`nvidia-uvm`) kernel module entered a corrupted state, returning `EIO` (Input/Output error) on all subsequent `open()` calls. This prevents `cuInit()` from succeeding across ALL containers on the host, making CUDA completely unavailable.

### Error Chain

1. **Initial error**: `RuntimeError: CUDA error: unspecified launch failure` at `multistage_predictor_batch.py:80` during QUARC condition prediction batch processing.
2. **GPU state**: GPU 0 (bus 8F:00.0) showed "ERR!" in nvidia-smi, 0% utilization, 0 MiB memory used.
3. **After container stop**: nvidia-container-runtime could not start any new GPU containers — NVML detection error.
4. **Root cause confirmed**: `strace` shows `openat("/dev/nvidia-uvm", O_RDWR) = -1 EIO`. The NVIDIA Unified Memory kernel module is corrupted.

### Repair Attempts

| Attempt | Result |
|---------|--------|
| GPU reset via `nvidia-smi -r` | Not Supported on A100 |
| PCI reset via sysfs | Read-only filesystem |
| Device node recreation | Device or resource busy |
| Module reload via rmmod | rmmod not available |
| Different GPU UUIDs | All containers show cuInit=999 |
| `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` | No effect on cuInit |
| Chmod UVM device to 000 | cuInit still fails |

### Working Recovery

After committing the stopped container and starting a new one with a working GPU (`GPU-1f212be0`), nvidia-smi (NVML) works but CUDA does not. The UVM issue is host-wide and affects all 8 containers.

## Corrected Evaluation Commands

### Original manifest command (broken):
```bash
cd /repo && source /opt/conda/bin/activate py312 && \
  export CUDA_VISIBLE_DEVICES=0 && \
  python -m moretro.moretro_star --dataset /repo/data/uspto_190_targets.csv \
  --config_file search_config.gin --output_dir eval_run
```

### GPU evaluation (requires GPU recovery):
```bash
cd /repo && source /opt/conda/bin/activate py312 && \
  CUDA_VISIBLE_DEVICES=0 \
  python -m moretro.moretro_star --dataset /repo/data/uspto_190_targets.csv \
  --config_file search_config.gin --output_dir eval_run
```

### CPU evaluation (current workaround):
```bash
cd /repo && source /opt/conda/bin/activate py312 && \
  python -m moretro.moretro_star --dataset <molecule_subset.csv> \
  --config_file search_config_cpu.gin --output_dir cpu_eval_run
```

CPU config differs from GPU config only in `device = "cpu"` (line 5 of `search_config.gin`).

### Metrics computation:
```bash
cd /repo && source /opt/conda/bin/activate py312 && \
  python compute_metrics_v2.py --output_dir output/<eval_dir>
```

## Baseline Verification

The baseline metrics from the partial GPU run match the reproduction manifest:
- HV: 0.401 (manifest: 0.4, within CI [0.05, 1.07]) ✓
- R2: 0.173 (manifest: 0.17, within noise tolerance) ✓  
- Success Rate: 33.3% (manifest: 33.3%) ✓
- Molecules processed: 27/190 (partial run before crash)

## Reusable /paper_data Resources

| Resource | Path | Status |
|----------|------|--------|
| USPTO-190 targets | `/paper_data/uspto_190_targets.txt` | Available, used |
| NeuralSym model | `/paper_data/models/template/` | Symlinked to `/repo/models/` |
| QUARC checkpoints | `/paper_data/models/quarc/` | Symlinked to `/repo/models/` |
| Objective models | `/paper_data/models/` | Symlinked to `/repo/models/` |
| Building blocks | `/paper_data/models/origin_dict.csv` | 1.69M eMolecules entries |
| USPTO-190 CSV | `/repo/data/uspto_190_targets.csv` | 190 SMILES strings |

## Safe Optimization Targets

All ideas from the idea library are safe (inference-only or config changes). Priority for CPU testing:

### P0 — Highest ROI, config-only:
1. **IDEA-03**: Sobol weight init + extreme points → config change, zero risk
2. **IDEA-08**: Exclude dominated nodes → config change, existing code path
3. **IDEA-06**: Epsilon-dominance pruning → config change, existing code path
4. **IDEA-11**: Increase single_step_topk K=25→50 → config change

### P1 — Config changes:
5. **IDEA-12**: Increase N_B and N_S → config change

### P2 — Code changes (higher risk):
6. **IDEA-01**: qLogNEHVI acquisition → code change in bo_weight_selector.py
7. **IDEA-02**: Template re-ranking → code change in retro_prediction.py
8. **IDEA-04**: Diversity warmup → code change in bo_weight_selector.py

## Evaluation on CPU

### Limitations
- Full 190-molecule benchmark: ~950 minutes estimated (exceeds 720-min timeout)
- Per-molecule time: ~5 minutes with 300 iterations (vs ~3.4 minutes on GPU for comparison, but GPU was processing 27 molecules in parallel)
- Reduced settings needed: subset of 10 molecules, 100 iterations, 3 concurrent weights

### Validation Approach
1. Run CPU baseline with reduced settings (10 molecules, 100 iters)
2. Apply optimization idea
3. Run with same reduced settings
4. Compare metrics
5. If improvement on subset, the same improvement should hold on full benchmark when GPU is available
