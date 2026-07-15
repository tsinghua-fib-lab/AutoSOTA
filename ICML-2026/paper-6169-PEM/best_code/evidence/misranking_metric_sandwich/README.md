# Misranking Metric Sanity Check — Evidence Package

Goal: support the theory claim that our probe statistic
`rank_disagreement` ($M_{RD}$) is **not ad-hoc** by connecting it to:

- Kendall discordant-pair fraction (`kendall_pairwise_disagreement`, $q_{pair}$)
- top-μ elite-set flips (`topmu_disagreement`, $M_{top\mu}$)

This plot checks the expected constant-factor relationships between these metrics on COCO samples.

## What’s inside

- `misranking_metrics_bbob_noisy_d40_es.csv`: measured metrics on COCO `bbob-noisy` with ES-sampled candidate sets.
- `misranking_metric_sandwich.png`: scatter plots with the constant-factor bounds.

## Setup used in this evidence

- Suite: `bbob-noisy`
- Dimension: `D=40`
- Instances: `1`
- Functions: `1–30`
- Candidate sampling: `--sampling es` (CMA-style candidate sets)
- Candidate set size: `λ=15` (matches the default ProbeSwitch probe at `D=40`)
- Top-μ: `μ=7` (≈ λ/2)

## Reproduce

```bash
python3 tools/measure_misranking_severity.py \
  --suite bbob-noisy --dims 40 --functions 1-30 --instances 1 \
  --sampling es --lambda 15 --mu 7 --num-sets 25 --seed 123 \
  --output-csv evidence/misranking_metric_sandwich/misranking_metrics_bbob_noisy_d40_es.csv

python3 tools/plot_misranking_metric_sandwich.py \
  --csv evidence/misranking_metric_sandwich/misranking_metrics_bbob_noisy_d40_es.csv \
  --out evidence/misranking_metric_sandwich/misranking_metric_sandwich.png \
  --title "bbob-noisy D=40: $M_{RD}$ vs Kendall/top-μ (ES candidate sets)"
```
