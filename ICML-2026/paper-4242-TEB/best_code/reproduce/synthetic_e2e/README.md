# §4.3.1 — Synthetic end-to-end

`(B, N)` grid search of v1 / EB(k̂\*) across three synthetic workloads
(decode-heavy 128/1024, balanced 512/512, prefill-heavy 1024/128). Backs
Fig. `fig_synthetic_e2e.pdf`.

> The paper's v0 column for Fig. `fig_synthetic_e2e.pdf` comes from a
> separate vLLM v0 codebase, not from this fork.

## Files

| File | Purpose |
|---|---|
| `run_grid_search.sh` | (B, N) grid over all 3 schedulers × 3 scenarios |
| `analyze_e2e.py` | Aggregate to `summary.csv` + `optimal_per_scheduler.csv` |
| `plot_synthetic_e2e.py` | Bar chart (3 scenarios × 3 metrics × schedulers) |

## Run

```bash
MODEL=Qwen/Qwen3-8B    ./run_grid_search.sh 8     # ~1.5 h on 8 GPUs
MODEL=Qwen/Qwen3-30B-A3B ./run_grid_search.sh 8   # ~7-8 h
python analyze_e2e.py outputs/e2e_grid_search/<GPU>_<MODEL>
python plot_synthetic_e2e.py \
    --grid-search-dir outputs/e2e_grid_search/<GPU>_<MODEL> \
    --output fig_synthetic_e2e.pdf
```

For a smoke test:

```bash
SCENARIOS=balanced BS_VALUES=1024 TB_VALUES=14336 NUM_PROMPTS=1000 \
    ./run_grid_search.sh 1
```

## Expected output

Bars compare schedulers per scenario for throughput, TTFT, TPOT.
Stars mark the best strategy per (scenario, metric).
