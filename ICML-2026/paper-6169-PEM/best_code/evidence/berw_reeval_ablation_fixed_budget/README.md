# Ablation: boundary reevaluation strength for BERW-Hetero (Fixed Budget, High Misranking)

Goal: test whether BERW’s fixed-budget gains depend on extra reevaluations (i.e., “hidden resampling”).

We vary the **per-point reevaluation count** used to build the residual pool / heteroscedastic scale model,
while keeping the **total evaluation budget fixed**.

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
- BERW variants (same algorithm, different `reeval_extra_per_point`):
  - `BERW-Hetero(reeval=0)` (no reevaluation; ablation lower bound)
  - `BERW-Hetero` (default, `reeval_extra_per_point=1`)
  - `BERW-Hetero(reeval=3)` (heavier reevaluation)

## Key artifacts (noise-free metric)

- Summary metrics (avg rank / win counts): `evidence/berw_reeval_ablation_fixed_budget/summary_metrics.csv`
- Paired sign-tests (exact, two-sided): `evidence/berw_reeval_ablation_fixed_budget/pairwise_sign_test.csv`
- Plot (avg rank): `evidence/berw_reeval_ablation_fixed_budget/avg_rank.png`

## High-level takeaway

On this fixed-budget slice:
- All three BERW variants are **significantly better** than `CMA-ES-sep` (paired sign-test).
- Increasing reevaluations from `reeval=1` to `reeval=3` does **not** yield a statistically significant change vs the default.

This supports the “fixed-budget sample efficiency” argument: BERW’s advantage is **not** explained by spending many extra evaluations on reevaluation.

## Reproduce

Run:

```bash
python3 tools/run_coco_bbob_noisy_parallel.py \
  --results-dir Results/_repro_berw_reeval_ablation_i1-15 \
  --dims 40 --budgets 100 \
  --functions 8,10,11,13,14,16,17,19,20,22,23,25,26,28,29 \
  --instances 1-15 \
  --algorithms "CMA-ES-sep,BERW-Hetero(reeval=0),BERW-Hetero,BERW-Hetero(reeval=3)" \
  --tag berw_reeval_ablation \
  --workers 4
```

Noise-free summary + stats:

```bash
python3 tools/summarize_coco_noisefree_from_exdata.py \
  --exdata-list Results/_repro_berw_reeval_ablation_i1-15/exdata_dirs.txt \
  --output-dir Results/_repro_berw_reeval_ablation_i1-15/noisefree

python3 tools/plot_bbob_results.py --results-dir Results/_repro_berw_reeval_ablation_i1-15/noisefree
python3 tools/pairwise_sign_test.py --results-dir Results/_repro_berw_reeval_ablation_i1-15/noisefree
```
