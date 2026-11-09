# Reproduction notes — GeoHab 2026 MLWG (paper 9001)

Container: `autosota_repro_paper_9001` (repo at `/repo`, data at `/paper_data:ro`).
Base image: `docker.1ms.run/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
(Ubuntu 20.04, Python 3.10.13, PyTorch 2.1.0+cu121, 4× A100-80GB visible).

## Result

Reproduced the reference (lock-candidate) blend recipe with the official protocol.

| Metric (fold-mean weighted F1, pre-Bayes OOF, frozen 150 m BlockKFold) | Reproduced | Registered |
|---|---|---|
| **Blend `0.40·LGB + 0.60·CNN`** (primary) | **0.7688 ± SE 0.0133** | 0.7688 ± 0.0133 |
| LightGBM-only (87 features) — `Baseline_Value` | 0.7519 ± SE 0.0238 | 0.7519 ± 0.0238 |
| CNN ensemble alone (soft vote) | 0.7254 ± SE 0.0404 | ~0.725 |

Pooled overall F1 of the blend: 0.7651. Per-fold: `[0.7841, 0.7642, 0.8078, 0.7606, 0.7274]`.
Both runs are exactly reproducible (identical values on repeated runs).

The reproduced 0.7688 sits inside the rubric CI `[0.7555, 0.7821]` and equals
`Paper_Method_Value` to 4 decimals. Secondary evidence: submission class counts match the
registered lock candidate exactly — `{ALG: 25, FMAT: 18, NVB: 40, SGAM: 9, SGZ: 6}` — and
`acr_sapa_w21` is rank 1/87 by gain with gain ≈ 23646 (registry: rank 1, ~23646).

## Commands

```bash
cd /repo
python pipeline/run_pipeline.py --out-dir outputs                  # ~40 s, CPU only
python /paper_data/eval/score_oof.py outputs/oof_predictions.csv   # official scorer
python /paper_data/eval/validate_submission.py outputs/submission.csv
```

`score_oof.py` prints the primary metric on its last line: `F1=0.7688`.

## What runs live vs. what is a vendored reference artifact

* **Live in-container (trained on every run):** the whole tabular component — LightGBM
  5-fold OOF plus the full-train refit — the blend, the E7 Bayes adjustment, OOF file
  generation, submission generation, and all scored metrics.
* **Vendored reference artifact:** the CNN ensemble probability matrix
  (`/paper_data/features/ensemble_outputs.npz`, keys `oof` (6256×5), `test_raw` (98×5)).
  This is **Path B of the reference notebook itself** (`pipeline/solution_notebook.ipynb`
  §6 cell 23-24, "the executed path"). It is not a shortcut invented here.

Why the CNN is not retrained. Path A (cells 18-22) needs inputs that are **not present in
this workspace and cannot be derived from what is**: `train_patches.npz` / `test_patches.npz`
(Kaggle dataset `mycarta/geohab-cnn-patches`) are absent, and the derived rasters the patches
were cut from (`data/derived/backscatter_destriped_masked_median2x.tif`,
`..._destriped.tif`) are absent too. The vendored repo references those derived rasters but
does not ship the FFT-destriping implementation (the notebook gives only pseudocode for a
filter explorer), so the exact CNN input transform is not reconstructible offline.
`/paper_data/MBES/{bathymetry,backscatter}.tif` (raw, striped) are present but are *not* the
CNN inputs. The notebook's own justification for Path B is that CNN inference is not
bit-deterministic across hardware, so the released ensemble outputs are the canonical
reference. Consensus CNN OOF CV here (0.7254) matches the write-up's "~0.725" for the same
matrix, which cross-checks that the artifact is the right one.

## Configuration (frozen to the reference, `pipeline/run_pipeline.py` CONFIG)

* LightGBM: `n_estimators=1000, learning_rate=0.1, num_leaves=31, min_child_samples=20`,
  `class_weight={ALG: 2.0}` (others 1.0), `random_state=42`, all other knobs at the
  notebook's explicit values. Verified against lightgbm 4.6.0 defaults (matches).
* Features: 86 from `train_features_v07.csv` + `acr_sapa_w21` from `train_features_v09.csv`
  = 87 columns; NaN handled natively by LightGBM (10 NaNs in `acr_sapa_w21`).
* Folds: loaded from `/paper_data/features/fold_indices.csv` (frozen). Sizes verified
  `[894, 1529, 1121, 1235, 1477]`. `verde` is therefore not needed and not installed.
* Blend: `0.40 * P_LGB + 0.60 * P_CNN`, argmax. **Pre-Bayes for OOF** (never adjusted).
* E7 prior adjustment (test only): `P_adj[c] = P[c] * TEST_PRIOR[c]/TRAIN_PRIOR[c]`,
  ratios ALG 2.7156, FMAT 0.7010, NVB 0.7562, SGAM 2.2426, SGZ 0.7745.

### One deliberate deviation: LightGBM `n_jobs`

The reference notebook sets `n_jobs=-1`. On this 128-core shared host OpenMP
oversubscription makes a single fit take >10 min (measured: `n_jobs=-1` >600 s vs
`n_jobs=8` 8.7 s, `n_jobs=16` 12.5 s, `n_jobs=32` 24.0 s for the same fit). The pipeline
defaults to `n_jobs=8`; override with `--lgb-jobs` or `GEOHAB_LGB_JOBS`. Thread count does
not change the fitted model — with `n_jobs=8` the tabular OOF CV is 0.7519 ± 0.0238 and the
blend 0.7688 ± 0.0133, exact matches to the registered values.

## Environment additions (installed into the container; not part of the image)

`pip install pandas==2.1.4 scikit-learn==1.3.2 lightgbm==4.6.0` from
`https://pypi.tuna.tsinghua.edu.cn/simple` (`PIP_CACHE_DIR=/autosota_cache/pip`), plus
`git` via `apt` from `http://mirrors.aliyun.com/ubuntu/` (apt sources rewritten to Aliyun;
backup at `/etc/apt/sources.list.bak`). `py-spy` was installed for diagnostics and is not
needed to run anything.

`/paper_data`, `/autosota_cache`, `/datasets`, `/models`, `/autosota_artifacts` are
container mounts; everything else lives inside the container.

## Artifacts written

`outputs/`: `oof_predictions.csv` (6256 rows `x,y,pred`), `submission.csv` (98 rows
`ID,class`), `tab_oof.npy`, `tab_test_proba.npy`, `cnn_oof.npy`, `cnn_test_proba.npy`,
`blend_oof.npy`, `blend_test_proba.npy`, `blend_test_proba_adj.npy`, `lgb_full_model.txt`,
`metrics.json`. Full run log: `/autosota_artifacts/paper-9001/reproduction/logs/repro_run.log`.

## Levers and dead ends for the optimization stage

Genuine headroom, from the reference write-up and registry:

* `acr_sapa_w21` is worth +0.016 CV over the 86-feature baseline; feature work on the
  FMAT/NVB/ALG/SGAM boundaries (registry reports `acr_sapa_w21` rank 1 by gain on the
  FMAT-NVB, FMAT-SGZ and NVB-SGZ pairs) is where the tabular component can still move.
* The CNN component (0.7254) is the diversity source: blend gain is +0.0173 over tabular
  alone. Non-uniform backbone reweighting is a documented dead end (0/27 triples passed
  its gate); uniform soft vote is optimal among the three backbones.
* Blend weight is worth sweeping; 0.40/0.60 was selected on CV and is not claimed optimal.
* Bayes prior adjustment affects only the 98-row test submission, never OOF, so it cannot
  move the scored metric.

Documented dead ends — do not burn budget re-litigating: KNN in any form (features,
standalone, average or stacked ensemble); object-based/higra zonal-stat segment features
(CV +0.011 but LB −0.034; three regressions); northing/`y`-aliased features
(`dist_to_creek_mouth` r=0.990 with `y`) — spatial-interpolator collapse; Optuna
hyperparameter search (banned: overfits at n=6256 under 150 m spatial CV); stacking with a
logistic-regression meta (three architectures, all 0.08-0.18 below the single-LGB gate);
isotonic calibration, residual stacking, and three-model tabular ensembles (zero
prediction diversity, 95/98 test predictions identical).

Hard red lines from the task card §6: never modify `eval/`, `fold_indices.csv`, `train.csv`,
`test.csv`, or the scoring protocol; keep OOF pre-Bayes and complete (all 6256 points);
test labels do not exist, so never fabricate or obtain them.
