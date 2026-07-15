# Ablation: heteroscedastic modeling choices for BERW (Fixed Budget, High Misranking)

Goal: test robustness of the heteroscedastic correction to the modeling choice used for per-point uncertainty
(e.g. the scale model `|noise|≈s0+s1|f|`).

We compare several **drop-in** BERW variants that differ only in how they synthesize per-point uncertainty
under mixed/heteroscedastic noise, under the same **fixed evaluation budget**.

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
  - `BERW-Hetero` (scale model `s0+s1|f|`)
  - `BERW-HeteroVar` (variance model `v0+v1|f|^2`)
  - `BERW-HeteroTMatch` (parametric t-matched bootstrap draws)
  - `BERW-HeteroRobust` (winsorized z-pool + trimmed aggregation; heavy-tail oriented)

## Key artifacts (noise-free metric)

- Summary metrics (avg rank / win counts): `evidence/berw_hetero_model_ablation_fixed_budget/summary_metrics.csv`
- Paired sign-tests (exact, two-sided): `evidence/berw_hetero_model_ablation_fixed_budget/pairwise_sign_test.csv`
- Plot (avg rank): `evidence/berw_hetero_model_ablation_fixed_budget/avg_rank.png`

## High-level takeaway

On this fixed-budget slice:
- All BERW variants are **significantly better** than `CMA-ES-sep` (paired sign-test).
- Differences between heteroscedastic modeling choices are **not statistically significant** here,
  suggesting the fixed-budget gain is not tied to a single fragile modeling assumption.

## Reproduce

Run:

```bash
python3 tools/run_coco_bbob_noisy_parallel.py \
  --results-dir Results/_repro_berw_hetero_model_ablation_i1-15 \
  --dims 40 --budgets 100 \
  --functions 8,10,11,13,14,16,17,19,20,22,23,25,26,28,29 \
  --instances 1-15 \
  --algorithms "CMA-ES-sep,BERW-Hetero,BERW-HeteroRobust,BERW-HeteroVar,BERW-HeteroTMatch" \
  --tag berw_hetero_model_ablation \
  --workers 4
```

Noise-free summary + stats:

```bash
python3 tools/summarize_coco_noisefree_from_exdata.py \
  --exdata-list Results/_repro_berw_hetero_model_ablation_i1-15/exdata_dirs.txt \
  --output-dir Results/_repro_berw_hetero_model_ablation_i1-15/noisefree

python3 tools/plot_bbob_results.py --results-dir Results/_repro_berw_hetero_model_ablation_i1-15/noisefree
python3 tools/pairwise_sign_test.py --results-dir Results/_repro_berw_hetero_model_ablation_i1-15/noisefree
```
