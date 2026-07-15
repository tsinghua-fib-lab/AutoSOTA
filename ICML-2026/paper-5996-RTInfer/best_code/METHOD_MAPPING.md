# RTInfer method mapping

Source paper: "RTInfer: Exploiting Concurrency for Multiple Real-Time DNN
Inference on Edge GPUs" (ICML 2026). Keep the paper PDF outside the public
source tree unless redistribution is explicitly permitted.

This implementation maps the paper methods to the current server-side
reproduction layer as follows.

## Offline stage

Accuracy-calibrated lightweight variant co-optimization:

- Implemented in `rtinfer.atlas`.
- Builds the variant space `V = {(p, e) | p in P, e in E}`.
- Uses pruning-aware latency and memory scaling when measured pruned artifacts
  are absent.
- Preserves the existing Pantheon early-exit profiles and models post-pruning
  fine-tuning as calibrated accuracy recovery.
- Provides Pareto filtering and a mixed-variable genetic selector.

Delta-Graph variant representation:

- Implemented in `rtinfer.delta_graph`.
- Splits variant weights into page-aligned chunks.
- Content-addresses chunks and tracks resident chunks through an LRU residency
  cache.
- Estimates `Tload = t0 + S / beta_min` for host-to-device transfer bounds.

Input profiles:

- `rtinfer.pantheon_io` reads Pantheon `config.pbtxt` from
  `variants_for_deployment`.
- When deployment configs are missing, it falls back to
  `Pantheon_Datasets_Models/3_Exported_JIT_Models/*/profile.csv` so indoor
  traffic, robot, and UAV workloads can all be simulated.

## Online stage

Memory-layout-aware scheduler:

- Implemented in `rtinfer.layout`.
- Represents each chunk/buffer as a rectangle in time x address space.
- Enforces non-overlap for buffers whose lifetimes overlap.
- Uses the paper heuristics: area, lifetime, and size ordering with bounded
  backtracking.

Load-aware online scheduling:

- Implemented in `rtinfer.scheduler`.
- Maintains urgency/deadline ordering.
- Selects variants by latency, memory, accuracy, and load time.
- Performs admission and downgrade when deadline or memory placement fails.
- Models early-exit fallback by selecting lower-latency/lower-memory variants.

Baselines and ablations:

- `rms-p`: rate-monotonic scheduling over pruned profiles.
- `dms-p`: deadline-monotonic scheduling over pruned profiles.
- `pantheon`: serial urgency-based Pantheon-style scheduling with early exits.
- `rtinfer`: concurrent RTInfer scheduling with memory layout and Delta-Graph.
- `rtinfer-wo-alc`: disables adaptive lightweight variant co-optimization by
  restricting choices to unpruned early-exit variants.
- `rtinfer-wo-ms`: disables memory-layout-aware placement.
- `rtinfer-wo-dlp`: disables Delta-Graph reuse and pays full variant load cost.

## Current scope

Completed:

- Server-side single-GPU simulation path.
- Synthetic unit tests that do not require private Pantheon artifacts.
- Optional Pantheon-profile integration when external artifacts are available.
- Dockerfile for isolated single-card simulation.
- Unit tests for atlas construction, Delta-Graph residency, online scheduling,
  and memory placement.

Pending for Jetson/C++ integration:

- Replace estimator-based pruned variant metrics with measured profiles from
  generated pruned TorchScript artifacts.
- Add C++17/LibTorch runtime integration beside Pantheon `online`, including
  CUDA priority streams and `cudaMallocAsync` pool management.
- Add real CUPTI telemetry once Jetson Xavier NX is restored.
- Validate measured DMR, accuracy, GPU utilization, and scheduler latency
  against the paper figures.
