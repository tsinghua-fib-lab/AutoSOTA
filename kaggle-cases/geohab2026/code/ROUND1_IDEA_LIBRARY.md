# ROUND-1 IDEA LIBRARY (inherited context for round 2)

This is the idea library produced during round 1 (AutoSOTA paper 9001, base CV 0.7688).
Of these, round 1 attempted: I11 (adopted during reproduction), I4's pennock-remap half
(negative, reverted), the LightGBM capacity probes (CAP-01, adopted), and the fusion
temperature (FUSE-01, adopted). The remaining ideas below are UNTESTED and are the
highest-known-value starting points for round 2 — but note their rollback anchors
reference round-1's 0.7688 baseline; the round-2 anchor is 0.7831 (see SOTA_NOTES.md).

---

## Idea Library

### I11 — Determinism and guardrail harness (run this first)

- **Type:** CODE · **Priority:** P0
- **Hypothesis:** Every F1 delta is only interpretable if the baseline is bit-reproducible and the two guardrail quantities are re-checked on every run. Without this, a "gain" from a silently swapped or re-normalised probability matrix is indistinguishable from a modelling improvement — precisely the failure the rubric's `LGB_tabular_F1` and `CNN_ensemble_F1` guardrails exist to catch.
- **Target:** `pipeline/run_pipeline.py` §5–§7 (add assertions next to the existing `tab_oof`/`cnn_oof` shape and row-sum checks). No new files required.
- **Concrete steps:** (a) assert `tab_cv_mean == 0.7519026…` and `cnn_cv_mean == 0.7254360…` against recomputed values, not just shapes; (b) assert `blend_cv_mean == 0.7688146…` before any experiment so the starting point is pinned; (c) record SHA-256 of `cnn_oof.npy`/`cnn_test_proba.npy` and of `ensemble_outputs.npz` in `metrics.json`; (d) emit per-fold F1 for both components, not only the blend, so per-fold regressions are visible.
- **Expected metric effect:** none (0.0000). Its value is that it makes every other idea's delta trustworthy and blocks false-positive acceptances.
- **Runtime/GPU cost:** <1 s, CPU only.
- **Failure risk / rollback:** near zero. If the pinned constants fail to reproduce, stop and re-reproduce before any optimisation.
- **Red-line check:** read-only instrumentation; does not alter predictions, folds, labels, or the scorer.
- **Source:** manifest `baseline_metrics` and `optimization_objective` (two named guardrails); manifest note "Repeated runs are bit-identical"; reproduction `metrics.json`.

### I4 — Categorical declaration + `pennock` nodata-sentinel repair

- **Type:** CODE · **Priority:** P0
- **Hypothesis:** `geomorphons` (8 observed levels in train, 7 in test) and `pennock` (5 levels) are ordinals in the CSV but nominal landform classes; passing them as ordered numerics forces LightGBM into threshold splits along a meaningless order. Separately, `pennock == -128.0` occurs in 4 train rows and **0 test rows**, so any split on that value is untransferable to the 98-row submission and only adds tree noise to the OOF.
- **Target:** `pipeline/run_pipeline.py` §3 (column sanitation, applied to both `v07_train` and `v07_test` before `column_stack`) and §5 (add `categorical_feature=[feat_names.index("geomorphons"), feat_names.index("pennock")]` to `LGB_PARAMS`).
- **Expected metric effect:** +0.000 to +0.006 fold-mean F1. Small but essentially free, and it is a genuine correctness fix rather than a tuned change.
- **Runtime/GPU cost:** negligible; LightGBM fit time may shift by a few percent (<2 s total).
- **Failure risk / rollback:** categorical splits can overfit, but `pennock` levels are well populated (1.0: 2509, 4.0: 2489, 2.0: 658, 3.0: 596) and `min_child_samples=20` bounds the leaves. Rollback if fold-mean F1 drops or if any fold falls >0.005 below its I11-pinned value.
- **Red-line check:** no change to eval, folds, labels, splits, or the pre-Bayes convention; a standard, well-specified LightGBM feature-type declaration. Not an Optuna/hyperparameter search.
- **Source:** direct CSV inspection of `train_features_v07.csv`/`test_features_v07.csv`; `run_pipeline.py` §3/§5; `features/README.md` (v07 does not flag a sentinel).

### I1 — Multi-scale ACR family (`acr_sapa_w3`, `acr_tin_w3`) spliced from the existing v09 CSV

- **Type:** ALGO · **Priority:** P0
- **Hypothesis:** `acr_sapa_w21` is rank 1/87 by gain (~23646) and rank 1 on the FMAT-NVB, FMAT-SGZ and NVB-SGZ per-pair diagnostics — the exact boundaries where the blend loses most (FMAT→NVB 307 errors, 19.9% of that class). A single 5.25 m window cannot express a roughness *spectrum*, which is the physical discriminator between a fine sandy mud mat (FMAT) and coarse sand/gravel (NVB). Adding the 0.75 m scale gives the model the fine/coarse contrast directly.
- **Target:** `pipeline/run_pipeline.py` §3 — extend the `usecols` of `train_features_v09.csv`/`test_features_v09.csv` to `["x","y","acr_sapa_w21","acr_sapa_w3","acr_tin_w3"]` and `["ID","acr_sapa_w21","acr_sapa_w3","acr_tin_w3"]`, then `column_stack`. Update the `X_train.shape` assertion to `(6256, 89)`.
- **Why these two and not more:** `acr_sapa_w3` has 0 NaN (vs 10 for w21) and `acr_tin_w3` uses a different denominator (POBF planar area vs slope-corrected area) with ~6× the dynamic range (std 0.012 vs 0.002), so they are not redundant copies of the rank-1 feature. Both are already extracted — no raster work, no new data.
- **Expected metric effect:** +0.003 to +0.010 fold-mean F1 if the family extends; realistically at the low end. Must clear the +1 SE gate against the I11-pinned 0.7688.
- **Runtime/GPU cost:** ~1 s added I/O + ~1 s per LightGBM fit; total eval still well under 60 s.
- **Failure risk / rollback:** the Sub-40 mechanism is the live risk — a strong new feature reallocates splits and can cost ALG recall (measured −0.0189 before `class_weight` was added to compensate). Rollback if (a) fold-mean F1 ≤ 0.7688, (b) `LGB_tabular_F1` < 0.737 (2% guardrail), or (c) the ALG per-class F1 falls below its I11-pinned value while FMAT/NVB rise. Do **not** extend this splice to the 14 `acf_*` columns in the same run — test that family separately (see I2/I10).
- **Red-line check:** adds two columns that already exist in the read-only provided feature CSV; no raster derivation, no test labels, no fold or metric change, no OB/higra involvement (22 OB columns stay excluded per the three-regression dead end).
- **Source:** `features/README.md` (v09 provenance, `acr_sapa_w21` rank 1/87); `results_registry.md` per-pair diagnostic (2026-04-25) and Sub 41 gates; column statistics inspected directly from `train_features_v09.csv`.

### I2 — Cross-scale ratio/difference features from columns already loaded

- **Type:** CODE · **Priority:** P0
- **Hypothesis:** The v07 set contains `bathy_focal_{mean,std,range}_w{3,7,11,21,41}` and `bs_focal_{mean,std,range}_w{3,7,11,21,41}` — 30 raw multi-scale columns but no cross-scale combination. GBDT splits are axis-aligned, so a ratio like `std_w41/ std_w3` requires many sequential splits to approximate and is usually left unexploited. These ratios are the cheapest possible form of a roughness-spectrum feature and, unlike I1, require no new data at all.
- **Target:** `pipeline/run_pipeline.py` §3 — insert a pure-numpy block that builds the derived block from the already-loaded `v07_train`/`v07_test` DataFrames and appends it to the `column_stack`. Suggested ~14 columns: `log((bathy_focal_std_w41+eps)/(bathy_focal_std_w3+eps))`, the same for `bs_focal_std`, `bathy_focal_mean_w41 - bathy_focal_mean_w3`, `bs_focal_mean_w41 - bs_focal_mean_w3`, `bathy_focal_range_w21/(bathy_focal_range_w3+eps)`, the same for backscatter, plus the w11/w3 pair for both rasters. Use `np.where(mask, ratio, np.nan)` so NaN still propagates and LightGBM handles it natively (no imputation, matching the existing convention).
- **Expected metric effect:** +0.003 to +0.012 fold-mean F1. Risk of dilution is real given `feature_fraction=1.0` (see F4), so add the block as **one** coherent family and test it alone.
- **Runtime/GPU cost:** <1 s for the derived block; each added column costs a small amount of LightGBM fit time (expect +2–5 s total).
- **Failure risk / rollback:** dilution/churn is the main risk — FMAT/NVB can improve while ALG or SGZ regresses. Rollback if fold-mean ≤ 0.7688, if fewer than 4/5 folds are non-negative, or if `LGB_tabular_F1` leaves the 2% guardrail. Keep the block behind a single `--derived-features` flag so it can be reverted without touching I1.
- **Red-line check:** derived only from feature columns already in the provided read-only CSVs, computed identically for train and test (so no train/test skew is possible); no position, northing, or y-aliased input (see F1); no metric, fold, or label change.
- **Source:** v07 column list inspected from `train_features_v07.csv`; `features/README.md` focal-statistic definitions; `results_registry.md` per-pair confusion concentration; manifest `known_levers` lever 1 (FMAT/NVB/SGAM boundaries are where the tabular half can still move).

### I3 — Remove the `class_weight` probability distortion before blending

- **Type:** ALGO · **Priority:** P0
- **Hypothesis:** `class_weight={ALG:2.0}` was introduced to fix *split reallocation* (registry: after adding `acr_sapa_w21`, ALG per-class F1 fell −0.0189 without it and recovered to −0.0017 with it). But LightGBM implements sample weights through the gradient/hessian, so the weighted model's output approximates `P(x|c)·w_c / Σ P(x|c')·w_c'` — the ALG probability is inflated ≈2× relative to the true posterior. That inflation then propagates unchanged into the linear blend with the unweighted CNN, distorting the decision surface. Decoupling the structural effect (keep training as-is) from the probabilistic one (correct the output) should let the blend weight be re-selected on a cleaner surface.
- **Target:** `pipeline/run_pipeline.py` §5 (post-`predict_proba` correction for both the fold models and the full-train refit) and §7 (blend).
- **Concrete step:** `p = clf.predict_proba(X); p = p / w; p = p / p.sum(1, keepdims=True)` with `w = np.array([2.,1.,1.,1.,1.])`, applied identically to `tab_oof` and `tab_test_proba`. Re-run the existing full-train refit so the saved `lgb_full_model.txt` and `tab_test_proba.npy` stay consistent. This has **zero free parameters** — nothing is tuned, so the result is a clean ablation.
- **Expected metric effect:** genuinely uncertain: −0.005 to +0.015 fold-mean F1. Treat as a pre-registered ablation with the +1 SE gate, not an assumed win. Combine with I5 (re-select `w_tab` on the calibrated matrices by leave-one-fold-out) because the old 0.40/0.60 was tuned to compensate for the distortion.
- **Runtime/GPU cost:** negligible (<1 s of numpy).
- **Failure risk / rollback:** LightGBM's sample weighting is only approximately a prior shift (it also alters tree structure), so the correction may over- or under-shoot. Rollback if fold-mean ≤ 0.7688 after re-selecting `w_tab`, or if the ALG per-class F1 falls below the Sub-41 recorded value (−0.0017 relative to the 87-feature baseline). Keep the correction behind a flag.
- **Red-line check:** a principled change to the model's own probability output using only training-fold information; the fold assignments, `eval/`, labels, and the pre-Bayes OOF convention are untouched; no test information and no `TEST_PRIOR` reach the OOF.
- **Source:** `run_pipeline.py` `LGB_PARAMS` and §5; registry Sub 40/Sub 41 rows (the +0.0161 `acr_sapa_w21` gain and the −0.0189 → −0.0017 ALG recovery); `methods/bayesian_prior.py` (the prior-shift identity this correction inverts).

### I5 — Fusion-operator search over the saved probability matrices, with leave-one-fold-out selection

- **Type:** ALGO · **Priority:** P1
- **Hypothesis:** The blend is one scalar in a linear pool. The registry calls 0.40/0.60 "CV-selected, not claimed optimal" and lists the sweep as a known lever, yet the operator family (power/temperature scaling, geometric pooling) is untouched. Both `tab_oof.npy`/`cnn_oof.npy` and the test matrices are already written by the pipeline, so every candidate is a re-argmax costing milliseconds.
- **Target:** `pipeline/run_pipeline.py` §7; matrices at `outputs/{tab_oof,cnn_oof,tab_test_proba,cnn_test_proba}.npy` (already produced by the current run).
- **Concrete steps:** evaluate in increasing complexity, stopping at the first that clears the gate:
  1. **Linear weight only.** `w_tab ∈ {0.30,0.35,0.40,0.45,0.50,0.55}` with `w_cnn = 1 − w_tab`.
  2. **Power scaling.** `p ∝ p_tab^a · p_cnn^b`, `a,b ∈ {0.5,0.75,1.0,1.5,2.0}` (geometric pooling; shrinks over-confident components before the linear pool).
  3. **Per-class bias on the pool** — defer to I6 (it deserves its own run).
  Select the parameter set by **leave-one-fold-out**: for each held-out fold f, pick the parameters maximizing mean F1 over the other four folds, then report the fold-mean of the resulting held-out predictions. For step 1 also keep the fixed-0.40 number for comparison.
- **Expected metric effect:** +0.002 to +0.008 from step 1 alone (SE 0.0133 means this is inside noise unless it appears in ≥4/5 folds); step 2 is the more interesting unknown.
- **Runtime/GPU cost:** negligible — pure numpy over 6256×5 and 98×5 arrays; <5 s including the LOO loop.
- **Failure risk / rollback:** **this idea is the most likely to manufacture a spurious gain**, because the OOF being optimised is the OOF being reported. The LOO-fold protocol is not optional. Rollback if the LOO-selected fold-mean does not exceed 0.7688 or if fewer than 4/5 folds improve.
- **Red-line check:** does not touch `eval/`, folds, labels, or the pre-Bayes convention; nothing is derived from test labels or `TEST_PRIOR`; the reported OOF file still contains one prediction per training point. The `outputs/oof_predictions.csv` schema is unchanged.
- **Source:** manifest `known_levers` lever 2 ("sweeping w_tab/w_cnn is cheap ... both probability matrices are already saved"); `run_pipeline.py` §7 and its `.npy` writes; `results_registry.md` Sub 42 CV provenance.

### I6 — Per-class decision bias optimised for weighted F1, estimated leak-free

- **Type:** ALGO · **Priority:** P1
- **Hypothesis:** Weighted F1 aggregates per-class F1 weighted by support, so it is not maximised by the argmax of a probability pool — it is maximised by a decision rule that trades precision for recall on the classes where the model is recall-limited. This dataset has exactly that structure: SGAM is 2.72% of train (n=170), its tabular OOF F1 is ≈0.01 while the CNN reaches ≈0.65, and the blend's SGAM count swings from 9 to 1–2 between adjacent candidate recipes. A multiplicative per-class bias `pred = argmax_c (p_c · b_c)` is the minimal decision-rule generalisation, and it is the same family as the existing `class_weight={ALG:2.0}` — just applied where it belongs.
- **Target:** `pipeline/run_pipeline.py` §7, applied to `blend_oof` before argmax (and symmetrically to `blend_raw` for the submission so test-time semantics stay consistent with the OOF decision rule).
- **Concrete steps:** parameterise `b = [1, b_FMAT, b_NVB, b_SGAM, b_SGZ]` (scale-free, 4 free parameters) or restrict to 2 (`b_SGAM`, `b_SGZ`) if the 4-parameter version is unstable. Search a coarse grid with `b_SGAM, b_SGZ ∈ {1.0,1.1,…,2.0}` and `b_FMAT, b_NVB ∈ {0.8,…,1.2}`, **selected by leave-one-fold-out** on the frozen folds — never by maximising the full-OOF number that is reported. Report the LOO fold-mean and the per-fold gains.
- **Expected metric effect:** +0.003 to +0.015 fold-mean F1 if the recall/precision trade is real; the safest version (b_SGAM only) is worth ~+0.002–0.005.
- **Runtime/GPU cost:** negligible; a few thousand numpy argmaxes over a 6256×5 array (<10 s).
- **Failure risk / rollback:** the highest red-line-adjacent idea in this library, for two reasons: (a) it is tuned against the metric, and (b) it sits near the E7 prior adjustment. Both are addressed by construction — the bias is estimated **only from training-fold labels** and **never** uses `TEST_PRIOR` or any test information, so it is not the E7 adjustment applied to OOF (which the pre-Bayes convention forbids) but an ordinary model decision rule. Rollback if the LOO fold-mean ≤ 0.7688, if the gain is absent on ≥2 folds, or if `F1_overall` (pooled) diverges from the fold-mean by more than the 1% tolerance — that divergence would indicate a fold-specific artefact. Also check SGAM recall gains are not paid for by FMAT/NVB losses summing to more.
- **Red-line check:** does not modify the metric, the scorer, the folds, the labels, or the OOF completeness. Uses no test labels and no hidden information. It is not hard-coded per-point prediction editing — the bias is a global 5-vector applied through `argmax`.
- **Source:** `eval/score_oof.py` (weighted-F1 definition by support); registry Sub 37/Sub 39 (SGAM OOF F1 0.65 CNN vs ≈0.01 tabular; SGAM count 9 vs 1–2); manifest `optimization_objective` and `known_levers`; `run_pipeline.py` `LGB_ALG_WEIGHT`.

### I7 — Rare-class specialist for the FMAT / NVB / SGAM subproblem

- **Type:** ALGO · **Priority:** P1
- **Hypothesis:** The blend's dominant errors are pairwise and concentrated: FMAT→NVB 307 (19.9%), SGZ→ALG 235 (28.5%), FMAT→ALG 169, NVB→ALG 157, FMAT→SGAM 145. A single global 5-class LightGBM must spend its entire split budget separating well-represented classes, and SGAM (n=170) is effectively unlearnable in that setting. A dedicated model restricted to the confused subset can allocate all of its capacity to those boundaries. The registry built 10 per-pair binary models as a *diagnostic* and never submitted a specialist, so this architecture is genuinely untested.
- **Target:** `pipeline/run_pipeline.py` §5 (new function, e.g. `train_subset_specialist(...)`), consumed in §7.
- **Concrete steps:** (a) train a second LightGBM on the same frozen folds using only rows whose **training-fold** label ∈ {FMAT, NVB, SGAM} and the same 87 features (plus I1's ACR columns); (b) at decision time, override the blend's prediction **only** when the blend's top-1 and top-2 both lie in {FMAT, NVB, SGAM} **and** the specialist's margin for its own argmax exceeds a threshold τ; (c) set τ by leave-one-fold-out and cap the override rate at ~10% of points as a safety bound; (d) never train the specialist on a point whose label comes from its own validation fold.
- **Expected metric effect:** +0.005 to +0.015 fold-mean F1 if SGAM recall improves without FMAT/NVB precision collapse. Realistically smaller, and it should be judged on SGAM per-class F1 first and fold-mean second.
- **Runtime/GPU cost:** one extra LightGBM fit per fold plus a full-train refit — roughly +10 s, keeping eval under 90 s.
- **Failure risk / rollback:** with n=170 SGAM points, a specialist overfits spatially very quickly; the registry shows SGAM behaviour is flat across reweighting sweeps, which is a warning that it may not move. Rollback if fold-mean ≤ 0.7688, if SGAM recall rises but FMAT+NVB per-class F1 falls by more in aggregate, or if the override rate exceeds the cap on any fold.
- **Red-line check:** uses only provided features and frozen-fold training labels; no test labels, no external data, no change to the scorer, folds, or the pre-Bayes convention.
- **Source:** `results_registry.md` per-pair diagnostic and SGAM statistics; manifest `known_levers` lever 1 (FMAT/NVB/ALG/SGAM are where the tabular half can move); note that this is *not* the retired logistic-regression stacking dead end (no meta-learner, no base-learner gating).

### I8 — Missingness / raster-edge indicator features

- **Type:** CODE · **Priority:** P2
- **Hypothesis:** Missingness in this feature table is not random — `acr_sapa_w21` is NaN for exactly 10 train and 3 test points, and those are survey-boundary pixels (ACR needs a full window at the raster edge), the same region the registry flags for the contested test IDs 7, 14, 19, 23, 46. Encoding "this point is at the raster edge" lets the model route those points to a different regime instead of silently treating NaN as an ordinary LightGBM default.
- **Target:** `pipeline/run_pipeline.py` §3, appended as two columns: `n_missing` (count of NaN across the 87 features) and `is_edge` (`acr_sapa_w21` is NaN or any `glcm_*_p25` is NaN, cast to float).
- **Expected metric effect:** +0.000 to +0.005. With only ~13 affected points the arithmetic ceiling is roughly ±0.003, so this is a diagnostic rider rather than a headline idea — report the per-class F1 of those 13 points before and after.
- **Runtime/GPU cost:** <1 s.
- **Failure risk / rollback:** very low. Rollback if fold-mean drops at all, since the expected gain does not justify any regression.
- **Red-line check:** derived from provided columns only; no positional features; no effect on folds, metric, or labels.
- **Source:** `features/README.md` NaN tables (10 train / 3 test for `acr_sapa_w21`); registry ECO-1 note on contested IDs; direct NaN counts inspected from the CSVs.

### I10 — Drop only provably zero-gain features

- **Type:** CODE · **Priority:** P2
- **Hypothesis:** With `feature_fraction=1.0` every column competes for every split, so columns with exactly zero gain contribute nothing but tie-breaking noise. The registry already records an instance of this (Sub 16: "Cooper edges 4/6 zero importance"). The full-train refit is saved, so the gain ranking is available without any new fit.
- **Target:** `pipeline/run_pipeline.py` §3/§5 — read `outputs/lgb_full_model.txt` (written by the current run) or the in-process `clf_full.booster_.feature_importance("gain")`, and drop columns whose gain is **exactly 0.0** (equivalently, never used in any split).
- **Expected metric effect:** +0.000 to +0.003. Close to a no-op by construction; the value is removing split-tie stochasticity before the larger I1/I2 experiments.
- **Runtime/GPU cost:** negligible.
- **Failure risk / rollback:** the danger is scope creep. **Restrict the rule to exactly-zero gain** — SFFS-style selection is a documented dead end (Sub 9: repeated CV 0.808 ± 0.050 → LB 0.77696, and Sub 16's 150 m CV 0.738). Rollback if fold-mean drops at all.
- **Red-line check:** model-internal feature pruning on training folds only; no test labels or metric used for selection; folds, scorer, and pre-Bayes convention untouched.
- **Source:** `results_registry.md` Sub 16 and Sub 9 entries; manifest `known_levers` lever 4 ("full-train LightGBM refit is saved ... feature-importance ranking is available for feature selection"); `run_pipeline.py` `feature_fraction=1.0`.

### I9 — CNN Path A, gated behind a destriping-recovery check

- **Type:** ALGO · **Priority:** P2
- **Hypothesis:** The CNN carries 0.60 of the blend weight, is the only component that can see SGAM (OOF F1 ≈0.65 vs ≈0.01 tabular), yet scores only 0.7254 against the tabular 0.7519. The manifest calls retraining "genuine unexploited headroom". The blocker is data, not method: `train_patches.npz`/`test_patches.npz` and the FFT-destriped rasters are absent, and the destriping parameters are not in the repo.
- **Target:** a new precompute script (e.g. `pipeline/build_cnn_patches.py`) writing a cached `.npz`, consumed by the **existing unchanged** reader at `pipeline/run_pipeline.py` §6 via the existing `--cnn-ensemble-npz` override. The pipeline must keep running in ~42 s; the retrain itself cannot live inside `eval_command`.
- **Concrete steps — and a hard gate before spending any training time:**
  1. **Gate 1 (destriping recovery, ~1 min).** `MBES/backscatter.tif` is 4743 × 4040 float32, uncompressed, ~76 MB — trivially readable. Apply the documented "Median 5×5 ×1 pass" to it and sample the result at the 6256 train coordinates, comparing against the v07 `backscatter` column (which `features/README.md` states was derived from exactly that median destriped raster). **Proceed only if Pearson r ≥ 0.99 with no residual stripe structure.** If it fails, the median variant is not the CNN's channel — stop, because the FFT variant's parameters are unrecoverable from this workspace.
  2. **Gate 2 (channel 1 is free).** `bathymetry.tif` is used raw in v07 (no destriping), so the bathymetry channel needs no recovery at all.
  3. Only then rebuild 96×96 2-channel patches, retrain the three backbones under the notebook's frozen recipe (30 epochs, no early stopping, ImageNet-pretrained conv1 averaged to 2 channels, uniform soft vote), and write a new npz with the same keys and the same fold semantics.
- **Expected metric effect:** potentially large (the CNN is the weaker and more SGAM-capable component, and the blend is 60% weighted on it) but the probability of clearing Gate 1 is unknown and the manifest says the artefacts are absent.
- **Runtime/GPU cost:** the notebook records ~2 h for the full three-backbone retrain; 4× A100-80GB are visible and currently unused. This **exceeds the 60-minute `eval_timeout_minutes`**, so it must be a one-off cached precompute in `/autosota_cache`, with `eval_command` unchanged and the pipeline reading the cached npz.
- **Failure risk / rollback:** HIGH — this is the only idea that risks burning the whole iteration budget for no gain. Rollbacks: (a) abandon at Gate 1 if r < 0.99; (b) if the retrained ensemble's OOF fold-mean F1 < 0.711 (2% below 0.7254), discard it and revert to `ensemble_outputs.npz`; (c) if the full pipeline runtime exceeds ~10 minutes, the retrain has leaked into the eval path — revert. **Guardrail caveat:** replacing the vendored matrix converts `CNN_ensemble_F1` from an integrity check into a performance floor. Confirm with the orchestrator that the rubric's "preserve within 2%" may be read as "≥ 0.711" before starting; if not, skip this idea entirely and keep the vendored npz.
- **Red-line check:** legitimate model retraining on provided data with the provided folds — **not** fine-tuning or replacing pretrained weights in a way the rubric forbids, because the image-net backbones are the paper's own method. No test labels, no metric change, no contamination. The manifest explicitly identifies Path A as the paper's own path that the reproduction could not reconstruct offline.
- **Source:** manifest `notes` and `setup_notes` (missing `train_patches.npz`, FFT-destriped rasters, and destriping implementation; 4× A100 unused; 60-minute timeout); `pipeline/solution_notebook.ipynb` cell 3 (`CNN_*` config), cells 24/28 (Path B reader and its integrity assertions); `features/README.md` (v07 backscatter provenance: Median 5×5 ×1 destriped); `METADATA.MD` (backscatter processing chain).

---

**Suggested first three runs for the SOTA Agent:** I11 (pin the baseline), then I4 (free correctness fix), then I1 (free rank-1 feature family at two more scales). Each is under a minute of added runtime, each has an unambiguous rollback signal against the pinned 0.7688, and together they establish whether the tabular half — the only live-trained component — can still move at all before any budget is spent on I3, I6, or I9.
