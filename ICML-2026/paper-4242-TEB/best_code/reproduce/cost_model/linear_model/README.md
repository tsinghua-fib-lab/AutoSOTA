# §3 — Linear iteration-time model

Validates `T_iter = α + β_p · L_p + β_d · N_d` (linear in prefill length and
decode batch size).

## Files

| File | Purpose |
|---|---|
| `benchmark_execution_time.py`, `benchmark_batch_combinations.py` | Sweep prefill chunk × decode count, measure iteration time |
| `analyze_prefill_linearity.py`, `analyze_prefill_linearity_all.py` | Fit slopes across models |
| `plot_execution_time.py` | Render the §3 / Appendix linearity figure |
| `model_configs/` | Tested model configs (Qwen3-4B/8B/14B/30B-A3B) |
| `results.json` | Sample run on Qwen3-4B (Blackwell) — used by the plot script |

## Run

```bash
python benchmark_execution_time.py --output results.json
python analyze_prefill_linearity_all.py
python plot_execution_time.py --input results.json --output execution_time.pdf
```

## Expected outputs

`plot_execution_time.py` produces a 2-panel figure: prefill linearity
(`T_p ≈ α_p + β_p · L`) on the left, decode linearity
(`T_d ≈ α_d + β_d · N`) on the right, with measured points and the linear
fit (slope/intercept annotated in the legend).
