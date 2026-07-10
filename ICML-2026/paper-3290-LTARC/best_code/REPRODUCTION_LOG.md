# Reproduction Log: Paper 3290

## Environment
- Container: autosota_repro_paper_3290
- Base image: autosota/paper-3290:env
- Python: 3.10.13
- GPU: 2x NVIDIA A100-SXM4-80GB
- CUDA: 13.0, Driver: 580.65.06

## Dependencies
- Installed via pip: numpy, pandas, scikit-learn, scipy, matplotlib

## Dataset
- STAR dataset downloaded from Harvard Dataverse (DOI: 10.7910/DVN/SIWH9F)
- File: STAR_Students.tab (13094524 bytes, ID: 666716)
- Converted to semicolon-separated CSV at /data_raw/STAR_students.csv
- Final dataset: 4218 students (2413 regular class, 1805 small class)

## Experiment
- Command: python3 -m experiments.main (see eval_command below)
- Settings: guarantee=p, dataset=star, n_splits=3, n_bins=200, n_mc=100
- Seed: 10603, Gamma: 1.0, Alpha: 0.10

## Results (tau=0.35, n_mc=100, seed=10603)
- Population Risk (mean): 0.5396
- Treatment Risk (mean): 0.2382
- Population Risk (median): 0.5374
- Treatment Risk (median): 0.2842

## Comparison with Paper
- Paper Treatment Risk: ~0.35 (Figure 5a)
- Paper Population Risk: ~0.50 (Figure 5b)
- Our treatment risk (0.238) is lower (better) than paper
- Our population risk (0.540) is higher (worse) than paper
- Policy is more conservative (treats fewer patients, lower treatment risk)
- Trade-off direction is consistent with paper findings
