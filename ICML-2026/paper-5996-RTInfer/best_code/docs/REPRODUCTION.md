# Reproduction Guide

This guide describes how to reproduce the public artifact outputs from a clean
checkout.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

The unit tests use synthetic profiles and do not require Pantheon artifacts.
If Pantheon and exported profiles are available at the local paths configured in
`tests/test_core.py`, the optional integration test runs automatically.

## 2. Core Simulator

With external Pantheon profiles:

```bash
export PANTHEON_ROOT=/path/to/Pantheon
export PROFILE_ROOT=/path/to/Pantheon_Datasets_Models/3_Exported_JIT_Models

python3 -m rtinfer.simulate \
  --deploy-json "$PANTHEON_ROOT/experiments/settings/deploy/robot.json" \
  --workload-json "$PANTHEON_ROOT/experiments/settings/workload/robot.json" \
  --profile-root "$PROFILE_ROOT" \
  --pantheon-repo "$PANTHEON_ROOT" \
  --device-preset jetson_xavier_nx \
  --duration-us 1000000
```

The simulator reports:

```text
policy,total,dmr,avg_accuracy,avg_latency_ms,avg_load_ms
```

## 3. Reviewer-Aligned Modern Workloads

```bash
./rebuttal_experiments/run_all_rebuttal.sh
```

This writes CSV and Markdown summaries under:

```text
outputs/scheduling_analysis/
outputs/runs/rebuttal_all.log
```

Key files:

- `modern_response_trace.csv`: per-job scheduling trace.
- `modern_utilization_trace.csv`: utilization proxy.
- `modern_scheduler_latency_cdf.csv`: scheduler latency CDF.
- `modern_memory_pressure_check.md`: effective memory-pressure check.
- `modern_acc_comparison.csv`: Fig. 1(c)-style modern accuracy comparison.
- `pantheon_accuracy_loss.md`: Pantheon accuracy sacrifice probe.

## 4. Paper-Style Figures

```bash
python3 paper_figures/make_revised_figures.py
```

Generated outputs:

```text
paper_figures/revised_outputs/
```

The figure script writes both SVG and PDF when `cairosvg` is installed.

## 5. Deterministic Jetson Nano Case Study

```bash
./case_studies/jetson_nano_case/run_case.sh
```

This writes an explanatory `modern_mixed_case.svg` plus CSV traces under:

```text
case_studies/jetson_nano_case/outputs/
```

The case is a deterministic simulation, not a Jetson Nano hardware measurement.

## 6. Jetson Profiling Helpers

If a Jetson Xavier NX is available:

```bash
python3 rebuttal_experiments/jetson_real_model_profiles.py --quick
python3 tools/jetson_variant_design_space.py --out-dir outputs/jetson_variant_design_space
```

These scripts measure architecture-level latency and memory behavior. They are
optional and hardware-specific.

## 7. Cleaning Generated Files

Generated outputs are ignored by Git. To remove Python build/cache files:

```bash
make clean
```

Remove experiment outputs manually when needed:

```bash
rm -rf outputs paper_figures/revised_outputs case_studies/jetson_nano_case/outputs
```
