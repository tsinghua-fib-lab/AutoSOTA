# Jetson Nano Case Study

Recommended output: `outputs/modern_mixed_case.svg`.

This case has been updated to a rebuttal-aligned modern mixed workload. It
uses six concrete RT tasks: traffic light detection with
MobileNetv2-SSDLite-300, high-resolution object detection with YOLOv8-L-1080p,
UAV scene recognition with MobileViT-S, large scene recognition with
ViT-L-1024, edge command generation with GPT-2-small KV cache, and wildfire
detection with ResNet152-512. The SOTA/Pantheon side explicitly allows RT+BE
concurrency; the advantage of RTInfer comes from RT-RT time-address packing,
ALC variant selection, and Delta-Graph load-aware switching.
The current figure models Reserve IF as local IF hold for a preempted job, not
as a persistent global memory reservation.

The previous 3-task minimal example is retained only as historical context in
the old output files.

It uses a simulated Jetson Nano memory budget:

- Total modeled GPU-visible shared memory budget: 4096 MiB.
- Reserve IF: no global reserve; local IF hold appears only when a preempted
  RT job keeps intermediate features for continuation.
- Workload: 6 real-time tasks, second-period arrivals, and 1 best-effort task.
- The simulation is deterministic and hand-sized so the online decisions are
  easy to inspect on a time x address plot.

## What The Case Shows

Pantheon/SOTA:

- Allows RT+BE concurrency, with BE preempted/paused during the dense RT burst.
- Uses local IF hold when a preempted RT job needs to resume from intermediate
  features.
- Keeps RT tasks mostly queue-ordered instead of globally packing RT chunks.
- Later tasks wait, then choose shallower early exits or miss deadlines.
- GPT-2 KV cache is represented as one conservative worst-case rectangle.

RTInfer:

- Accuracy-Calibrated Variant Co-Optimization chooses pruned + deeper-exit
  variants that fit memory while retaining higher accuracy.
- Memory-Layout-Aware Scheduling packs the active RT tasks into the 2D
  Time x Address memory space under the 4 GiB budget.
- Delta-Graph and load-aware pipelining load only missing variant chunks before
  first use on second-period variant switches; first-use model arrivals are
  still full loads.

## Run Locally

```bash
cd RTInfer
./case_studies/jetson_nano_case/run_case.sh
```

Outputs are written to:

```text
case_studies/jetson_nano_case/outputs
```

Main modern output:

- `modern_mixed_case.svg`: visual comparison of Pantheon vs RTInfer.
- `modern_variant_table.csv`: task variants and selected exits/pruning ratios.
- `modern_pantheon_trace.csv`: SOTA schedule with RT+BE concurrency.
- `modern_rtinfer_trace.csv`: RTInfer packed concurrent schedule.
- `modern_online_decisions.md`: step-by-step online scheduler decisions.
- `modern_summary.txt`: final DMR/accuracy/latency comparison.

## Run In Docker

Docker cannot truly turn a 3090 Ti into a Jetson Nano or enforce per-container
GPU memory limits. This case therefore enforces the 4096 MiB Jetson Nano budget
inside the scheduling simulator.

```bash
cd RTInfer
./case_studies/jetson_nano_case/run_case_docker.sh
```
