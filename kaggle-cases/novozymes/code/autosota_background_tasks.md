# AutoSoTA paper-9003 — background task ledger

| # | task id | iter / idea | command | cwd | log | pid/pattern | start (UTC) | deadline | expected output | score row |
|---|---------|-------------|---------|-----|-----|-------------|-------------|----------|-----------------|-----------|
| 1 | bg-structfeat | pre-iter-1 / FEAT-STRUCT | `timeout 1800 taskset -c 16-39 python sota/extract_struct_feats.py --workers 8 --out /repo/outputs/struct_features.csv` | /repo | /repo/sota/logs/structfeat.log | `extract_struct_feats.py` | 2026-10-13 17:45Z | +30 min | `/repo/outputs/struct_features.csv`, 2413 rows x ~24 cols | none (feature build; consumed by iter-1/2) |
