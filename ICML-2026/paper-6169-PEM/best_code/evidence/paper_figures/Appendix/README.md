# Appendix Figures

This directory contains all figures for the paper appendix (Sections A1–A10).

## File Mapping

| Paper Section | Title | File |
|---------------|-------|------|
| A1 | Mechanism validation on a controlled quadratic | `fig_a1_mechanism_quadratic.pdf` |
| A2 | RB-PEM estimator ablations | `fig_a2_ablations.pdf` |
| A3 | Residual-pool diagnostic snapshots | `fig_a3_diagnostics.pdf` |
| A4 | Interpreting the rank-disagreement probe | `fig_a4_misranking_sandwich.pdf` |
| A5 | Variance does not equal misranking | `fig_a5_probe_decoupling.pdf` |
| A6 | Probe calibration curves | `fig_a6_probe_calibration.pdf` |
| A7 | Probe reliability versus probe budget | `fig_a7_probe_budget_roc.pdf` |
| A8 | Threshold sensitivity analysis | `fig_a8_threshold_sensitivity.pdf` |
| A9 | Depth–fidelity robustness and UH-CMA-ES sensitivity | `fig_a10_depth_fidelity_tradeoff.pdf` |
| A10 | External validity on nonconvex real-data task | `fig_a12_mlp_digits0.pdf` |

**Note**: File names `fig_a10_*` and `fig_a12_*` correspond to sections A9 and A10 respectively, due to historical renumbering during the paper revision process.

## A11

Section A11 presents a table (not a figure). See `evidence/paper_tables/table_a11_high_misranking.tex`.

## Regeneration

To regenerate these figures from the evidence data:

```bash
python tools/plot_fig_a1_mechanism_quadratic.py
python tools/plot_fig_a2_ablations.py
python tools/plot_fig_a3_diagnostics.py
python tools/plot_fig_a4_misranking_sandwich.py
python tools/plot_fig_a5_probe_decoupling.py
python tools/plot_fig_a6_probe_calibration.py
python tools/plot_fig_a7_probe_budget_roc.py
python tools/plot_fig_a8_threshold_sensitivity.py
python tools/plot_fig_a10_depth_fidelity_tradeoff.py  # generates A9
python tools/plot_fig_a12_mlp_digits0.py              # generates A10
```
