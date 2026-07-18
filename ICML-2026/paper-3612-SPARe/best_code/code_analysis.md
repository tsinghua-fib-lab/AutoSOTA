# Code Analysis: SOTA Preparation Repair — Paper 3612

## Original Preparation Failure

The orchestrator failed to initialize the git repository inside the container because:

1. **Overlay filesystem at 100% capacity**: The Docker overlay2 thin pool was fully exhausted (200G/200G used). This prevented:
   - `apt-get install git`: Failed due to "No space left on device"
   - `docker cp`: Failed due to "no space left on device" when creating pivot directory
   - Any file creation/deletion on the overlay

2. **Missing git binary**: The image `autosota/paper-3612:reproduced` (based on `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`) does not include git.

3. **Broken apt state**: Multiple gcc/g++ packages were mid-installation, compounding the space issue.

## Repair Approach

### Key Insight
The NFS-mounted paths (`/autosota_cache`, `/autosota_artifacts`) have 6.4T free space. All writes must target these paths.

### Solutions Applied

1. **Git replacement**: Python `dulwich` library installed to `/autosota_cache/tmp/python_pkgs/`. Custom git shim at `/autosota_cache/git-shim.py` handles init, add, commit, tag, config, rev-parse, log. All git data stored on NFS at `/autosota_cache/sota_git/`.

2. **File deployment**: All helper scripts written to host NFS path (`/home/dataset-assist-0/litianxing/autosota-v2.5-3/cache/`) which appears as `/autosota_cache/` inside the container.

3. **PATH setup**: `/etc/profile.d/sota_path.sh` adds `/autosota_cache:/tools` to PATH inside container.

## Corrected In-Container Evaluation Command

```bash
export PATH=/autosota_cache:/tools:$PATH
cd /repo
bash evaluate.sh
```

The `evaluate.sh` script runs the SimGrid-based SPARe DES simulator with:
- SPARe+CKPT: r=9, ckpt=15 (paper optimal)
- Rep+CKPT: r=3, ckpt=6 (paper baseline)
- Each: 3 trials with seeds 42, 123, 456
- Metrics: time-to-train/T0 (T0=660000s), Availability

## Baseline Verification

| Metric | Manifest | Reproduced | Match |
|--------|----------|------------|-------|
| time-to-train/T0 | 2.946 | 2.9460 | ✓ |
| Availability | 87.216% | 87.2157% | ✓ |

## Reusable Resources

- `/paper_data`: Not mounted (no external data needed — DES simulation)
- `/autosota_cache/spare_bin/{spare,dp,dpr}`: Compiled simulator binaries
- `/autosota_cache/tmp/platform.xml`: Platform configuration (179MB)
- `/autosota_cache/simgrid/lib/libsimgrid.so.4.0`: SimGrid runtime
- `/autosota_cache/ubuntu_toolchain/`: Ubuntu g++-9 runtime libraries

## Safe Optimization Targets

The simulator accepts command-line parameters for:
- `--replicate_level` (r): Redundancy level (1-28 valid)
- `--ckpt`: Checkpoint interval in steps
- `--workers` (N): Number of workers
- `--compute`, `--data`, `--model`: Workload parameters
- `--fail-dist`, `--weibull-shape`, `--weibull-scale`: Failure model
- `--compute-jitter`, `--recover`, `--partial-recover-time`: System parameters
- `--seed`: Random seed

These are all user-configurable via the command line and do not require code modification.

## Optimization Results Summary

Swept r ∈ {7,8,9,10,11} × ckpt ∈ {12,13,14,15,16,18}.

Best found: **r=9, ckpt=14** with time-to-train/T0=2.9074 (-1.31% vs baseline), Availability=87.41%.

See `final_report.md` for full results.
