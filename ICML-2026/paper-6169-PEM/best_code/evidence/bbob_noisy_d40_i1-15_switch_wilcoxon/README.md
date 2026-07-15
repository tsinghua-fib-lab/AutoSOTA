# bbob-noisy D=40 i=1–15: Wilcoxon signed-rank (noise-free)

Provides Wilcoxon signed-rank tests as an additional paired statistical test beyond sign-test.

## Setup

- Result directory: `Results/bbob_noisy_d40_i1-15_switch_probe_t012_B200/noisefree/`
- Compared:
  - `Switch-MisrankingProbe(t=0.12)`
  - `CMA-ES-sep`
- Metric: COCO noise-free `best_f` (extracted from `exdata/` via `bbob_summary.csv`)

## Files

- `pairwise_wilcoxon_switch_vs_cma_noisefree_B200.json` (from `tools/pairwise_wilcoxon.py`)
