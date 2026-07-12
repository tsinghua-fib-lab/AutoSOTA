# §4.2 — Controller validation

Validates that the online (k̂\*, N̂\*) controller matches the throughput of
exhaustive fixed-k sweeps and outperforms v1 (mixed batching). Backs
Fig. `validation_grid.pdf` (3 scenarios × 3 metrics).

## Files

| File | Purpose |
|---|---|
| `run_validation.sh` | Fixed-k sweep + EB(k̂\*) + v1 on 3 synthetic CFR workloads |
| `analyze_validation.py` | Aggregate bench JSONs into `validation_summary.csv` |
| `plot_validation_grid.py` | Render the 3×3 grid (renders from CSV or `--demo`) |
| `plot_validation_grid_paper.py` | Re-render paper Figure 3 from extracted data (`paper_data/validation_grid.json`) |
| `plot_validation_comparison.py` | Overlay paper Figure 3 + your reproduced values for side-by-side diff |
| `paper_data/validation_grid.json` | Numerical data extracted from paper Figure 3 (for re-plotting + comparison) |

## Important: matching paper Figure 3 exactly

Paper Figure 3 caption: *"Validation on H200, N = 1024."* All three workloads
were run at **fixed batch size N=1024 with `VLLM_PD_AUTO_COMPUTE_N=0`** (no
memory-safe online shrinking).

If you leave `VLLM_PD_AUTO_COMPUTE_N=1` (or the older script default), the
CFR controller will aggressively shrink N̂\* (e.g. to ~315 on decode-heavy)
to satisfy the ε=0.01 OOM bound; throughput then drops by ~2.4× relative
to the paper figure. **This is not a regression in the EB algorithm** —
it's the memory-safe Proposition\,memory bound being more conservative
than what the paper figure used.

`run_validation.sh` now defaults to `VLLM_PD_AUTO_COMPUTE_N=0`. To
*exactly* match Figure 3 (same N across all three workloads), also set:

```bash
BS_DECODE_HEAVY=1024 BS_BALANCED=1024 BS_PREFILL_HEAVY=1024 \
    MODEL=Qwen/Qwen3-8B ./run_validation.sh 8
```

## Run

```bash
MODEL=Qwen/Qwen3-8B ./run_validation.sh 8        # ~30 min on 8 GPUs
python analyze_validation.py outputs/controller_validation/<GPU>_Qwen3-8B
python plot_validation_grid_paper.py                 # re-renders paper Fig. 3
python plot_validation_comparison.py                 # overlays paper + reproduction
```

`SKIP_EXISTING=1` (default) lets you re-run incrementally. See
`run_validation.sh` header for env knobs (e.g. `VLLM_PD_OOM_TOLERANCE`).

## Fixed-k sweep

The green curve in Fig. `validation_grid.pdf` is produced by sweeping a grid
of fixed switching thresholds `k` (`θ*=k/N`) per workload, with the online
controller disabled. This is now produced by `run_validation.sh` itself —
no need to back-fill from `paper_data/validation_grid.json` anymore.

| Env var | Default | Effect |
|---|---|---|
| `K_VALUES` | `"128 256 384 512 640 768 896 1024"` | k values to sweep (whitespace-separated). Match paper Fig. 3 by default. Set `K_VALUES=""` to skip the sweep. |
| `FIXED_K_N` | `1024` | Pinned N (max-num-seqs) for every sweep cell, regardless of the scenario's default `BS_*`. Matches the paper caption "N=1024". |

Per sweep cell `k`, the runner sets `VLLM_PD_K_MODE=ratio`,
`VLLM_PD_K_RATIO = k / FIXED_K_N`, and `VLLM_PD_AUTO_COMPUTE_N=0`. Output
files land at `bench_eb_fixed_k_<k>.json`, which `analyze_validation.py`
discovers automatically.

## Expected output

3 rows (decode-heavy / balanced / prefill-heavy) × 3 columns (throughput,
mean TPOT, p99 TPOT). Green points = fixed-k sweep with `k\*` annotated,
red horizontal = v1, blue dashed = EB(k̂\*).
