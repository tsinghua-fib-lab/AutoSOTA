# Reproducing the EB(k̂*) paper (ICML 2026)

This directory ships the experiment scripts, analysis tools, and plot
scripts that reproduce every figure and table in the paper. The
documentation is split into three files; pick the one matching your
question:

| Document | Purpose |
|---|---|
| **[`REPRODUCE.md`](REPRODUCE.md)** | Step-by-step end-to-end recipe with wall-clock estimates on 8×RTX PRO 6000 — install, calibration, dataset export, and one section per paper artifact. **Start here to reproduce.** |
| **[`PD_SCHEDULER_ENV_VARS.md`](PD_SCHEDULER_ENV_VARS.md)** | Reference for every `VLLM_PD_*` env var read by the EB / EB(k̂\*) / EB⁺ scheduler — default, type, audience, read site. Look here when you hit an unfamiliar knob. |
| `README.md` (this file) | Map from paper section → subdirectory, plus pointers to shared infrastructure. |

## Paper section → subdirectory

| Paper section | Artifact | Subdirectory |
|---|---|---|
| §3 Cost model — linear iteration-time | Fig. `execution_time*.pdf`, `prefill_linearity_all_models.png` | [`cost_model/linear_model/`](cost_model/linear_model/) |
| §3 Cost model — kernel breakdown | Fig. `kernel_breakdown*.pdf` | [`cost_model/kernel_breakdown/`](cost_model/kernel_breakdown/) |
| §3 Hazard rate / CFR vs IFR | Fig. `CFR_IFR.pdf`, `hazard_rate_comparison.pdf` | [`hazard_rate/`](hazard_rate/) |
| §4.2 Model validation | Fig. `validation_grid.pdf` | [`validation/`](validation/) |
| §4.3.1 Synthetic e2e | Fig. `fig_synthetic_e2e.pdf` | [`synthetic_e2e/`](synthetic_e2e/) |
| §4.3.2 Real workloads + §4.3.3 latency | Tables 2-3, Figs. `ttft.pdf`, `tpot.pdf` | [`real_workloads/`](real_workloads/) |
| §4.4 EB⁺ traffic-level | Table 4 | [`eb_plus/traffic/`](eb_plus/traffic/) — paper sweep via `run_table4_sweep.sh` |
| §4.4 EB⁺ non-stationary | Table 5 | [`eb_plus/non_stationary/`](eb_plus/non_stationary/) |
| §4.4 PD disaggregation comparison | Appendix `app:disagg_2gpu` / `app:disagg_4gpu` | [`disagg/`](disagg/) — paper sweeps via `run_{2,4}gpu_paper_sweep.sh` |
| §4.4 Long-context comparison | Fig. `combined_ctx_comparison_tok1024.pdf`, Tab `e2e-128k` | [`long_context/`](long_context/) |
| §4.5.1-2 Scalability | Table 6, Fig. `scalmodel.pdf` | [`scalability/`](scalability/) |

## Shared infrastructure

- [`common/`](common/) — shared bash helpers (`common.sh`, `common_eb.sh`)
  and Python utilities (`dataset_utils.py`, `export_dataset.py`).
- [`calibration/`](calibration/) — per-(model, GPU) cost-model calibration
  JSONs. Samples for Qwen3-{8B, 30B-A3B} × {H200, RTXPRO6000} plus several
  cross-model entries are shipped; regenerate via
  `python -m vllm.v1.core.sched.calibration --model <MODEL>` for new
  (model, GPU) pairs.
- [`docker/`](docker/) — Dockerfile and build script for a reproducible
  environment.
- [`real_workloads/build_optimal_json.py`](real_workloads/build_optimal_json.py)
  — aggregates per-workload `grid_summary.json` files into the combined
  `optimal_per_scheduler.json` consumed by `plot_real_workload_latency.py`
  and `plot_scalmodel.py`.
- [`long_context/aggregate.py`](long_context/aggregate.py) — aggregates
  per-(model, context_len) bench JSONs into the `long_context_summary.csv`
  consumed by `plot_long_context.py`.

## Run order

Follow [`REPRODUCE.md`](REPRODUCE.md) §0 → §10. Each section is
self-contained and idempotent (`SKIP_EXISTING=1` is the default, so a
re-run resumes from the next missing cell).

## Notes

- Output files (`outputs/`, `__pycache__/`, etc.) are gitignored. Each
  `run_*.sh` writes results under its local `outputs/` subtree.
- Plot scripts marked **(demo)** render a synthetic illustrative version
  when no real CSV/JSON is supplied — useful for previewing figure shape
  before kicking off multi-hour experiments.
- For an internal log of reproduction-blocker patches applied while
  preparing the camera-ready snapshot, see `CHANGELOG.md` (gitignored).
