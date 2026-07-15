# RTInfer Rebuttal Experiments

This directory contains the supplemental experiments described in the ICML
rebuttal discussion. It is intentionally separated from the original
`scripts` directory.

The scripts use transparent synthetic profiles calibrated to the reviewer
comments and author responses. They are meant to reproduce the rebuttal logic
before the real Jetson Xavier NX deployment is restored. Once measured
TorchScript/Jetson profiles are available, replace the synthetic model profiles
in `common.py` with measured latency, memory, and accuracy data.

## What Is Covered

- `transparent_configs.py`: prints the exact non-standard CNN configurations
  stated in the rebuttal, including input shapes, batch size, early exits, and
  pruning tiers.
- `modern_workloads.py`: runs memory-intensive modern workloads: YOLOv8n,
  YOLOv8-L, MobileViT-S, and ViT-L style concurrent streams.
- `completed_accuracy.py`: reports both deadline-weighted accuracy and raw
  completed-only accuracy to address the reviewer concern about assigning 0
  accuracy to missed deadlines.
- `dynamic_kv_cache.py`: emulates a generative NLP workload with a stepped KV
  cache footprint and compares conservative rectangle packing against stepped
  time-address packing.
- `npu_sram_emulation.py`: emulates the Google Coral Edge TPU style 8 MB static
  SRAM allocation experiment using RTInfer's 2D layout solver.
- `jetson_real_model_profiles.py`: measures Jetson Xavier NX latency and CUDA
  memory for pure-PyTorch YOLOv8-like, MobileViT/ViT-like, and GPT-2 KV-cache
  model graphs when official model packages or pretrained weights are not
  available on the board.
- `memory_pressure_check.py`: verifies that each revised application exceeds
  the effective Jetson Xavier NX memory budget before pruning, early exits,
  layout scheduling, or Delta-Graph loading are applied.
- `pantheon_accuracy_loss.py`: regenerates the Fig. 1(c)-style
  `Orig./Pantheon/PC-Pantheon` comparison and the Pantheon accuracy-loss probe.
- `ablation_stress.py`: reruns RTInfer ablations under a tighter dedicated
  stress setup (`kappa=0.12`, `4096 MiB`, `3 GB/s` H2D) so Fig. 11 exposes DMR,
  accuracy, and load-latency differences instead of reusing the looser overall
  setup.
- `run_all_rebuttal.sh`: runs all rebuttal experiments in sequence.

## Run

```bash
cd RTInfer
./rebuttal_experiments/run_all_rebuttal.sh
```

Individual examples:

```bash
python3 rebuttal_experiments/modern_workloads.py
python3 rebuttal_experiments/dynamic_kv_cache.py
python3 rebuttal_experiments/npu_sram_emulation.py
python3 rebuttal_experiments/jetson_real_model_profiles.py --quick
```

## Interpretation Notes

These scripts are not claiming new measured Jetson results. They encode the
rebuttal-era experimental setup and make the assumptions executable:

- Jetson Xavier NX effective GPU memory is modeled as 6 GiB after OS, sensor,
  ROS, and runtime overhead.
- High-resolution modern vision workloads are activation-memory dominated.
  We model shared weights once conceptually, while per-stream activation memory
  dominates the rectangle height.
- Delta-Graph benefits are intra-task variant-switching benefits. Heterogeneous
  models still benefit from the memory-layout scheduler, even when no cross-model
  weight deduplication exists.
- Dynamic NLP memory is represented conservatively by a worst-case rectangle
  and more efficiently by stepped time-address buffers.
- The Jetson real-model profiling script uses randomly initialized but real
  PyTorch computation graphs. It measures architecture-level latency/memory; it
  does not claim task accuracy without the original datasets and pretrained
  weights.
