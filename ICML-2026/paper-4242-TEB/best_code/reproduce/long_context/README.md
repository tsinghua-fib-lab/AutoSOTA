# §4.4 — Long-context comparison

Compares v1 / EB(k̂\*) / EB⁺ on long-context workloads (8K–128K input
tokens, fixed 1024 output tokens). Backs Fig.
`combined_ctx_comparison_tok1024.pdf`.

## Files

| File | Purpose |
|---|---|
| `run_long_context_comparison.sh` | 4 GPUs, sweeps context length 8K-64K |
| `run_128k_all_models.sh` | Bigger model set at 128K context |
| `plot_long_context.py` | 4-panel figure (throughput, TTFT, TPOT, ITL vs context length) |

## Run

```bash
./run_long_context_comparison.sh     # ~2 h
./run_128k_all_models.sh             # ~2 h
# Aggregate the bench outputs into long_context_summary.csv (manual or via
# the helper in plot_long_context.py's docstring), then:
python plot_long_context.py long_context_summary.csv \
    --output combined_ctx_comparison_tok1024.pdf
```

The script accepts `--demo` for a synthetic illustration before real data
is available.
