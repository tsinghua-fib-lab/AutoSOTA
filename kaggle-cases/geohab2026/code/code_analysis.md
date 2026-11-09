# Paper 9001 — SOTA code analysis (GeoHab 2026 MLWG, Refuge Cove)

## Evaluation path

| Item | Value |
| --- | --- |
| Container | `autosota_repro_paper_9001` (reused; `container_reusable=true`) |
| Repo | `/repo` (git, branch `master`; tags `_baseline`, `_best`, `_pre_iter1`) |
| Eval command (byte-identical to the manifest, unmodified) | `python pipeline/run_pipeline.py --out-dir outputs && python /paper_data/eval/score_oof.py outputs/oof_predictions.csv` |
| Metric parser | last stdout line `F1=<fold_mean>`; summary line `F1_fold_mean=<m> F1_SE=<se> F1_overall=<o>`; also `outputs/metrics.json -> F1_fold_mean` |
| Scorer | `/paper_data/eval/score_oof.py` (read-only; never modified) |
| Primary metric | `F1` fold-mean weighted F1 over the 5 frozen 150 m BlockKFold folds, pre-Bayes OOF; higher is better |
| Baseline | `F1=0.7688`, `F1_overall=0.7651`, `F1_SE=0.0133`, `LGB_tabular_F1=0.7519`, `CNN_ensemble_F1=0.7254`, runtime 37.6 s |

No manifest repair was needed: the manifest `eval_command` is already an in-container
command and runs as written from `/repo`. Nothing about the protocol was changed.

## Train / inference path (`/repo/pipeline/run_pipeline.py`)

1. **§2 data** — `/paper_data/train.csv` (6256 pts), `test.csv` (98), `sample_submission.csv`.
2. **§3 tabular features** — 86 columns of `features/train_features_v07.csv` + `acr_sapa_w21`
   from `features/train_features_v09.csv` = `X_train (6256, 87)`, `X_test (98, 87)`.
   Row order is asserted against `train.csv`; test is aligned by `ID`.
3. **§4 folds** — loaded frozen from `features/fold_indices.csv` (`point_index`, `fold`),
   sizes asserted `[894, 1529, 1121, 1235, 1477]`.
4. **§5 tabular model** — LightGBM 1000 trees / lr 0.1 / 31 leaves / `class_weight {ALG: 2.0}` /
   seed 42, 5-fold OOF + full-train refit; writes `tab_oof.npy`, `tab_test_proba.npy`,
   `lgb_full_model.txt`.
5. **§6 CNN component** — read from the vendored reference artifact
   `features/ensemble_outputs.npz` ("Path B"). Keys: `oof (6256,5)`, `test_raw (98,5)`,
   `test_adj`, `train_labels`, `train_x`, `train_y`, `test_ids`. Never retrained.
6. **§7 fusion + E7** — `blend = 0.40*tab + 0.60*cnn`, argmax -> `oof_predictions.csv`
   (pre-Bayes). The E7 prior adjustment is applied to the **test** probabilities only.
7. **outputs** — `oof_predictions.csv` (6256 rows), `submission.csv` (98 rows),
   the `.npy` matrices, `metrics.json`.

## Config path

All knobs live in `CONFIG` at the top of `pipeline/run_pipeline.py`; `LGB_PARAMS` is built from
it. CLI overrides: `--paper-data`, `--out-dir`, `--cnn-ensemble-npz`, `--lgb-jobs`,
`--min-child-samples`. The eval command passes none of them, so the defaults are what is scored.

## Reusable `/paper_data` resources (read-only, all inspected)

* `train.csv`, `test.csv`, `sample_submission.csv`
* `features/train_features_v07.csv` (6256×89), `test_features_v07.csv` (98×89)
* `features/train_features_v09.csv` (6256×129), `test_features_v09.csv` (98×129) —
  v07 + 1 FMM + 14 ACF + 3 ACR + 22 OB
* `features/fold_indices.csv` — frozen folds
* `features/ensemble_outputs.npz` — the vendored CNN probability matrix (Path B)
* `features/README.md` — provenance of every feature block
* `MBES/bathymetry.tif` (raw), `MBES/backscatter.tif` (raw, striped) — **not** the
  destriped rasters v07 was built from
* `eval/score_oof.py`, `eval/validate_submission.py`
* `reference/mycarta_solution_writeup.ipynb`

Genuinely missing (Path A only): `train_patches.npz` / `test_patches.npz`, the
FFT-destriped rasters, and the destriping implementation.

## Risky files — do not modify

`/paper_data/**` (read-only mount), `features/fold_indices.csv`, `train.csv`, `test.csv`,
`eval/score_oof.py`, `eval/validate_submission.py`, `outputs/oof_predictions.csv` schema.

## Safe modification targets

`pipeline/run_pipeline.py` — CONFIG, feature assembly (§3), the LightGBM fit (§5), and the
fusion rule (§7). These are the paper's own modelling choices and leave room for legitimate
work; everything downstream of the fold assignment stays untouched.

## Measurement discipline used in this SOTA phase

* Exploration harnesses live in `/repo/experiments/` and only read the probability matrices.
* Any *selection* (blend weight, per-class bias, gating, smoothing) was re-checked with a
  leave-one-fold-out protocol, because the OOF being optimised is also the OOF being reported.
* Recorded iterations always run the unmodified official eval command over a freshly trained
  model; no score is ever transcribed from a probe.

## Findings so far

| Probe | Result |
| --- | --- |
| `fusion_probe.py` | The published 0.40/0.60 weight is exactly the optimum of the linear sweep; geometric pooling, per-class bias, confidence gating and LOO weight selection are all neutral-to-worse. The fusion layer is exhausted. |
| `probe3.py` | `geomorphons`/`pennock` as categorical, and the `pennock == -128` sentinel, are all within ±0.001. |
| `probe4.py` | Labels are 99.3 % nearest-neighbour consistent, but transductive spatial smoothing *lowers* the fold-mean. |
| `probe5.py` | **The published fold models reach in-sample F1 = 1.0000 while scoring 0.6911 on the held-out fold** — the recipe memorises. Training longer adds nothing at `min_child_samples=20`. |
| `probe6/7.py` | Regularising the leaf floor fixes it: every one of 16 settings improves the blend, best `min_child_samples=100` at the published `num_leaves=31` -> blend 0.7785. |

## Pre-existing gaps (unchanged, fail-safe)

`_best` at `_baseline` until iteration 1; `eval/validate_submission.py` exists but is not part
of the scored command (it validates the 98-row submission format only).
