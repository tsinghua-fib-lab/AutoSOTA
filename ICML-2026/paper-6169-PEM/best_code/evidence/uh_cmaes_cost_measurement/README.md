# UH-CMA-ES Cost Measurement (fixed budget)

This evidence pack measures the *actual* evaluation overhead of UH-CMA-ES under the
fixed-budget protocol (bbob-noisy, B=100·d, d=40).

## Files

- `uh_cmaes_cost_measurements.csv`: per-run measurements (algorithm × function × instance)
- `uh_cmaes_cost_summary.csv`: aggregated summary used by Figure 2 (depth–fidelity)

## Reproduce

```bash
python3 tools/run_uh_cmaes_cost_measurement.py \
  --out-dir evidence/uh_cmaes_cost_measurement \
  --dims 40 \
  --functions 8,10,11,13,14,16,17,19,20,22,23,25,26,28,29 \
  --instances 1-15 \
  --budget-mult 100
```

