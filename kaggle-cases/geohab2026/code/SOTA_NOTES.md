# SOTA_NOTES — round-1 AutoSOTA optimization record (paper 9001 → this repo is its best state)

This repository is the **round-1 AutoSOTA best export** (`paper-9001-FUSE-01`, tag `_best`,
container commit `1e61d16d`) of the GeoHab 2026 MLWG solution. It is shipped as the
**round-2 baseline**: reproduce it exactly (fold-mean weighted F1 = **0.7831**, pre-Bayes
OOF on the frozen 150 m BlockKFold), then push beyond it.

Original sources: `autosota-v2.6/artifacts/paper-9001/` (sota/scores.jsonl,
sota/optimization_curve.txt, evaluation/evaluation_verdict.json) and
`autosota-v2.6/best_code_exports/paper-9001-FUSE-01/final_report.md`.

## Adopted changes (both live in `pipeline/run_pipeline.py` CONFIG)

1. **CAP-01 — LightGBM capacity correction** (`LGB_MIN_CHILD_SAMPLES` 20 → 100; iter-1, F1 0.7688 → 0.7785).
   Diagnostic, not a hyperparameter search: at the published `min_child_samples=20` the fold
   models reach in-sample weighted F1 = 1.0000 while scoring 0.6911 on the held-out fold —
   the sample is dominated by near-duplicate points (99.3% nearest-neighbour label agreement
   at 0.75 m median spacing) and the trees isolate clusters of them. All 16 regularised
   settings probed (mcs 60..260; num_leaves 4..48) improve the blend fold-mean over 0.7688;
   the response is a broad plateau, not a spike. At num_leaves=31 this single change gives
   tabular 0.7519 → 0.7623 and blend 0.7688 → 0.7785.
2. **FUSE-01 — tabular fusion temperature** (`TAB_SHARPEN = 6.0`; iter-2, F1 0.7785 → 0.7831).
   The tabular posterior enters the linear pool as `p_tab ** 6` (renormalised); the CNN
   component (`CNN_SHARPEN = 1.0`, vendored matrix) and the published 0.40/0.60 weight are
   untouched. Rationale: the tabular posterior is produced under `class_weight={ALG: 2.0}`,
   which inflates one class by construction; a temperature keeps its argmax decision while
   stopping the distortion from diluting the confident CNN calls. Two assertions in the code
   prove the operator cannot move either component's own argmax (guardrails unaffected).
   Evidence: response is a broad plateau (a = 3..12 all give 0.7826–0.7831; the a → ∞
   hard-vote limit gives 0.7827); leave-one-fold-out selection over the a grid gives 0.7822
   with a = 6 chosen in 4/5 folds — the honest estimate of the *procedure*, not of a=6.

## Rejected / negative results (do NOT blindly re-run; reasons matter)

- **iter-3 `pennock` nodata remap (−0.0008)** — remapping the -128 sentinel (4 train rows,
  0 test rows) to NaN: F1 0.7823 vs 0.7831, whole delta on fold 2, folds 0/1/3/4 bit-identical.
  Reverted. (An exploration harness had first reported +0.0022 for this idea — that was a
  harness bug, corrected before the attempt was recorded. Trust only scored pipeline runs.)
- **iter-4 `LGB_N_ESTIMATORS` 1000 → 500 (exact tie)** — 0.7831, identical to iter-2, only a
  different per-fold shape. Reverted (no gain).
- **iter-5 `boost_from_average=False` (−0.0025)** — removes the log-prior start; folds 3/4
  lose 0.0073/0.0087 (their class mixes deviate sharply from the training mix). Reverted.
- **Leakage trap refused (+0.0060 apparent)** — an idea that would have applied the E7/TEST_PRIOR
  correction to the OOF was detected and refused. The pre-Bayes OOF convention is load-bearing:
  test-prior information must never enter the scored OOF.
- Inherited from the vendored write-ups (see `writeups/negative_results.md`,
  `results_registry.md`): KNN in any form; object-based/higra zonal-stat features (CV +0.011 /
  LB −0.034); northing or y-aliased features (spatial-interpolator collapse); Optuna-style
  tuning (overfits at n=6256 under 150 m spatial CV); logistic-regression stacking (3 variants,
  −0.08 to −0.18); isotonic calibration; residual stacking; three-model tabular ensembles
  (95/98 identical test predictions).

## Honesty / integrity notes

- **Escape hatches** recover earlier epochs bit-for-bit:
  `--min-child-samples 20 --tab-sharpen 1` → 0.7688 (published recipe);
  `--tab-sharpen 1` → 0.7785 (iter-1); default → 0.7831 (iter-2, this baseline).
- **Final verification**: best state re-trained from scratch after all later iterations were
  rolled back → bit-identical F1 = 0.7831, per-fold [0.7838, 0.7833, 0.8204, 0.7727, 0.7554],
  F1_SE 0.0106, F1_overall 0.7778. Repeated runs are bit-identical.
- **CNN integrity**: the CNN component is the vendored `features/ensemble_outputs.npz`
  (sha256 36291c39f2c3040e778c513e497f59d1d3e0d847cea73d624bdca641d34b7db3). `metrics.json`
  records the npz hash and the CNN OOF hash on every run; the CNN_ensemble_F1 guardrail
  (0.7254) is checked so a gain cannot come from a swapped/corrupted matrix.
- **Evaluation verdict** (round-1 evaluation agent): success, confidence medium — gain
  +0.0143 is ~1.1× the baseline fold-SE and concentrated in folds 3–4; the leak-free LOO
  estimate (0.7822) still exceeds baseline. Guardrails: LGB_tabular_F1 0.7623 (improved),
  CNN_ensemble_F1 0.7254 (unchanged).
- **Submission** (FUSE-01, E7 prior applied test-only): 98 rows, class counts
  {ALG 26, FMAT 18, NVB 39, SGAM 9, SGZ 6}. Round-1 Kaggle outcome: 7th public / 2nd private
  on the late-submission board.
- **Runtime** ~45 s CPU; `--lgb-jobs` (env `GEOHAB_LGB_JOBS`) defaults to 8 because
  n_jobs=-1 oversubscribes a 128-core shared host; thread count does not change the fitted model.
