# §3 — Hazard rate / CFR vs IFR

Empirical hazard-rate analysis of real-workload output lengths, plus the
CFR (geometric / constant failure rate) vs IFR (increasing failure rate)
illustration. Backs `CFR_IFR.pdf` and `hazard_rate_comparison.pdf` in §3.

## Files

| File | Purpose |
|---|---|
| `run_hazard_rate_experiment.sh` | **Canonical** hazard-rate sweep used for the paper (called from `run_all_paper_artifacts.sh`). |
| `run_hazard_rate_experiment_v2.sh` | _Experimental_ — finer k* step + 5 repeats + cool-down. Not used for the camera-ready figure; kept for reference / sensitivity studies. |
| `analyze_hazard_rate.py`, `analyze_hazard_rate_robust.py` | Compute empirical hazard rates |
| `analyze_real_workload_hazard.py` | Per-dataset hazard-rate breakdown (ShareGPT, WildChat, etc.) |
| `plot_sharegpt_hazard_rate.py` | Render the empirical hazard rate of ShareGPT |
| `plot_cfr_ifr.py` | Render the CFR vs IFR theoretical comparison (3-panel) |

## Run

```bash
# Theoretical illustration (self-contained, no data needed)
python plot_cfr_ifr.py --mean 512 --gamma-shape 4.0 --output CFR_IFR.pdf

# Empirical hazard rate from real workloads
./run_hazard_rate_experiment.sh 4
python analyze_real_workload_hazard.py
python plot_sharegpt_hazard_rate.py --output hazard_rate_comparison.pdf
```

`plot_cfr_ifr.py` can optionally overlay empirical samples via
`--empirical-json <path>` (expects `{"output_lengths": [...]}` format).
