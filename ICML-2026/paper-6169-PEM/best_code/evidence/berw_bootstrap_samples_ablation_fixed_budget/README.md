# Ablation: `bootstrap_samples` for BERW-Hetero (Fixed Budget, High Misranking)

Goal: test sensitivity of BERW-Hetero to the bootstrap-resampling count (`bootstrap_samples`).

We vary the number of bootstrap rank-resamples (`bootstrap_samples`) in **BERW-Hetero** while keeping everything else fixed.

## Setup

- Suite: COCO `bbob-noisy`
- Dimension: `D=40`
- Budget: `B=100×D`
- Instances: `1–15` (COCO standard)
- Functions: the same **high-misranking slice** used by the fixed-budget Hansen test:
  - ids `{108,110,111,113,114,116,117,119,120,122,123,125,126,128,129}`
  - indices `{8,10,11,13,14,16,17,19,20,22,23,25,26,28,29}`

Algorithms:
- Baseline: `CMA-ES-sep`
- BERW variants:
  - `BERW-Hetero` (default `bootstrap_samples=32`)
  - `BERW-Hetero(bs=16)`
  - `BERW-Hetero(bs=64)`

## Key artifacts (noise-free metric)

- Summary metrics (avg rank / win counts): `evidence/berw_bootstrap_samples_ablation_fixed_budget/summary_metrics.csv`
- Paired sign-tests (exact, two-sided): `evidence/berw_bootstrap_samples_ablation_fixed_budget/pairwise_sign_test.csv`
- Plot (avg rank): `evidence/berw_bootstrap_samples_ablation_fixed_budget/avg_rank.png`

High-level takeaway (from the paired sign-tests):
- BERW variants are **not significantly different** from each other on this slice.
- All BERW variants are **significantly better** than `CMA-ES-sep` under the same fixed budget.

## Reproduce

Run:

`python3 tools/run_coco_bbob_noisy_parallel.py --results-dir Results/_repro_berw_bs_ablation_i1-15 --dims 40 --budgets 100 --functions 8,10,11,13,14,16,17,19,20,22,23,25,26,28,29 --instances 1-15 --algorithms "CMA-ES-sep,BERW-Hetero,BERW-Hetero(bs=16),BERW-Hetero(bs=64)" --tag repro_berw_bs_ablation_i1-15 --workers 4`

Noise-free summary + stats:

`python3 tools/summarize_coco_noisefree_from_exdata.py --exdata-list Results/_repro_berw_bs_ablation_i1-15/exdata_dirs.txt --output-dir Results/_repro_berw_bs_ablation_i1-15/noisefree && python3 tools/pairwise_sign_test.py --results-dir Results/_repro_berw_bs_ablation_i1-15/noisefree && python3 tools/plot_bbob_results.py --results-dir Results/_repro_berw_bs_ablation_i1-15/noisefree`
