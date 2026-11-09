# RESULTS REGISTRY — GeoHab 2026 MLWG Competition
# Source-of-truth for all LB submissions. Append-only.
# RULE: No row enters without Kaggle submission page or paste-after-submit.
# RULE: No handoff, no summary, no chat may override these numbers.
# RULE: Consult this file BEFORE any strategic claim about what scored what.
# Built: 2026-04-07 from Kaggle submission history screenshot (15 submissions).

| Sub | Filename | Description | LB |
|-----|----------|-------------|----|
| 1 | quick_baseline_submission.csv | RF 500 trees, 27 geomorphometric features | 0.77888 |
| 2 | full_feature_submission.csv | RF 500 trees, 82 features (geomorphometric + focal stats + rank texture + GLCM + HSI) | 0.81096 |
| 3 | lgbm_submission.csv | LightGBM 1000 trees, 82 features, default hyperparams | 0.81096 |
| 4 | optuna_lgbm_submission.csv | LightGBM 82 features, Optuna 150 trials | 0.77289 |
| 5 | submission_mstp_85feat.csv | 85 features (82 base + 4 MSTP - 3 dropped). RF 500, balanced, BlockKFold 50m CV=0.787 | 0.77928 |
| 6 | submission_raster_only_diag.csv | 67 raster-only features, no GLCM, no MSTP. RF 500, CV=0.770. Diagnostic to isolate GLCM rebuild regression | 0.77928 |
| 7 | submission_RF500_95feat_knn_20260402_2338.csv | 95 features (85 base + 10 KNN label features). RF 500, balanced. KNN k=5 features 100x more important than raster features | 0.72732 |
| 8 | submission_83feat_mstp1.csv | 82 base + mstp_band1. RF 500 balanced. CV=0.788. 1 prediction changed vs 0.811 baseline | 0.81096 |
| 9 | submission_lgbm_sffs_20feat_20260404_0701.csv | SFFS 20 features (from 85), LGB 1000 balanced default. Repeated CV 0.808±0.050 (10 seeds × 5-fold BlockKFold) | 0.77696 |
| 10 | submission_knn_k5_20260404_0812.csv | Pure KNN classifier (k=5, distance-weighted, coords only). Spatial-only ceiling test. No raster features | 0.72732 |
| 11 | submission_avg_knn_lgbm_20260404_0812.csv | Equal-weight average: 0.5×KNN(coords) + 0.5×LightGBM(85feat). No meta-learner | 0.76394 |
| 12 | submission_stacked_knn_lgbm_20260404_0812.csv | KNN(coords) + LightGBM(85feat) OOF probabilities → LogReg meta-learner. BlockKFold 50m | 0.76214 |
| 13 | submission_raster_ensemble3_20260404_0826.csv | 3-model raster-only ensemble: equal-weight probability average of LightGBM(1000 trees) + RF(500 trees) + Ridge(CalibratedCV). All 85 features, all balanced, no spatial labels. Testing model diversity without KNN | 0.73970 |
| 14 | submission_14_detrended.csv | Submission 14: LGB default, 86 features (85 original + bathy_detrended). Polynomial detrended bathymetry captures creek depositional fan and reef platform residuals. 1 prediction changed: ID 71 NVB→SGAM | 0.73970 |
| 15 | sub_XX_lgb_85feat_v01_reproduce_baseline.csv | LGB default n_est=1000, 85 features v01, Sub A baseline reproduction attempt | 0.73970 |
| 16 | submission_16_cooper_edges_20260407.csv | v06: 91 features (85 v01 + tilt_angle + tdxn + thag + canny_dist + compressor_s21 + knn_entropy_k5). LGB default n_est=1000. Cooper edges 4/6 zero importance. knn_entropy_k5 ranked 3rd (0.0318). 50m CV=0.807±0.057, 150m CV=0.738±0.151 | 0.81096 |
| 17 | submission_17_knn_entropy_only_20260407.csv | v01 + knn_entropy_k5 only (86 features). LGB default n_est=1000. Isolation test: entropy alone does not explain 0.73970→0.81096 jump. 50m CV=0.808±0.061, 150m CV=0.739±0.151 | 0.75807 |
| 18 | submission_18_canny_only_20260407.csv | v01 + bathy_canny_dist only (86 features). LGB default n_est=1000. Isolation test: canny_dist alone. 50m CV=0.802±0.054, 150m CV=0.727±0.138. bathy_canny_dist perm importance 0.0107 rank #4 | 0.81096 |
| 19 | submission_19_cnn_resnet18_64px_20260407.csv | CNN ResNet18 pretrained, 64x64 patches (bathy+bs+bathy). Spatial CV 50m 3 seeds. CV=0.770±0.070, OOF F1=0.786. No class weighting. Augmentation: rot360+flip+brightness0.1 | 0.75574 |
| 20 | submission_20_lgb_cnn_ensemble_20260407.csv | 0.5 LGB (Sub 18, 86feat) + 0.5 CNN (ResNet18 64px, Sub 19) ensemble. 3 predictions flipped from Sub 18 (IDs 2,55,76). All 3 wrong on public LB | 0.77289 |
| 21 | submission_21_ob_v2seg_felzenszwalb_20260408.csv | v08: 95 features (v07 86feat + 9 OB from Felzenszwalb v2 segments, 1209 segs). LGB default n_est=1000. mean/std/skew of bathy/bs_median/rugosity_index per segment. Single-seed BlockKFold: 50m CV=0.810±0.028, 150m CV=0.738±0.066. ob_rugindex_std ranked #3 perm importance (0.0129) — Ierodiaconou's #1 variable confirmed in-model. 4 preds flipped from Sub 18 (IDs 7,14,19,55) — all wrong on public LB. Felzenszwalb segments contaminated by backscatter stripes → noisy zonal stats. OB concept valid; Felzenszwalb segmentation is the bottleneck. | 0.77696 |
| 22 | sub_d4_configA_argmax.csv | v09: 101 features (Boruta survivors from 126), LGB n_est=1000, no early stopping. v09 = v07 destriped (86) + T_east (1) + ACF texture (14) + ACR rugosity (3) + Higra OB dyn800+dyn1200 (22). 150m CV=0.761±0.114, 50m CV=0.843±0.055. 5 preds differ from Sub 18 (IDs 1,2,7,48,76). Harmful — OB features cause regression. Do not use v09 for submissions. | 0.79615 |
| 23 | sub_23_v07_baseline_lgb1000.csv | v07 baseline isolation test. LGB n_est=1000, 86 features (v07 destriped), no early stopping. Only 1 pred differs from Sub 18 (ID 7: FMAT→SGAM — destriped input effect). Confirms v07 baseline matches 0.811. | 0.81096 |
| 24 | submission_24_cnn_resnet18_64px_classweighted_rugosity.csv | CNN ResNet18 64px, class weighted, rugosity channel. Changed 3 things from Sub 19 — can't isolate cause. | 0.74846 |
| 25 | sub_25_v07_optuna_oof_avg.csv | v07 86 feat, LGB Optuna params (lr=0.079, depth=11, leaves=77), 5-fold StratKFold test prob averaging, early stopping (best_iter 133–194). Competitor params wrong for our features. | 0.80196 |
| 26 | sub_26_v07_default_oof_avg.csv | v07 86 feat, LGB default, 5-fold StratKFold test prob averaging, early stopping (best_iter 51–64). ES killed it — fold averaging not cleanly tested yet. | 0.75900 |
| 27 | sub_27_v07_lgb_fold_avg.csv | v07 86 feat, LGB default, 5-fold StratKFold test prob averaging, NO early stopping, n_est=1000 | 0.77289 |
| 28 | sub_28_v07_xgb_fold_avg.csv | v07 86 feat, XGB default, 5-fold StratKFold test prob averaging, NO early stopping, n_est=1000 | 0.81096 |
| 29 | sub_29_v07_cat_fold_avg.csv | v07 86 feat, CatBoost default, 5-fold StratKFold test prob averaging, NO early stopping, n_est=1000 | 0.77928 |
| 30 | sub_30_v07_equal_ensemble.csv | Equal-weight avg of LGB+XGB+Cat test probs. NOT SUBMITTED — CSV identical to Sub 28 | — |
| 31 | sub_31_v07_weighted_ensemble.csv | Nelder-Mead weighted avg of LGB+XGB+Cat OOF probs. NOT SUBMITTED — CSV identical to Sub 28 | — |
| 32 | sub_32_v07_baseline_corrected.csv | v07 86 feat, LGB default n_est=1000, ID 17 ALGdescription→ALG corrected. Clean baseline | 0.81096 |
| 33 | sub_33_v07_classweight_balanced.csv | v07 86 feat, LGB default n_est=1000, class_weight='balanced' (NVB=0.412, FMAT=0.810, SGZ=1.518, ALG=1.835, SGAM=7.360). NOT SUBMITTED — byte-identical to Sub 32, 0/98 predictions flipped | — |
| 34 | exp_beta_candidate.csv | β: v07+T_east (87 feat) + 3 isolated rugosity-SD OB features (ob_rugindex_w3_mean/std/skew) from Higra dyn800 segmentation on destriped bathymetry gradient, zonal-stat over acr_rugosity_w3.tif. LGB n_est=1000 default. CV_150=0.7683 PASS. LB 0.76214. Third OB failure — OB approach closed at Refuge Cove regardless of segmentation/input/isolation. | 0.76214 |
| 35 | sub_35_seed_avg_lgb.csv | v07+T_east (87 features), seed-averaged LGB default n_est=1000. Seed averaging confirmed byte-identical null-op (LGB defaults deterministic). 5 argmax flips vs Sub 32 at contested Monarch IDs [2, 7, 11, 55, 76]: 2 FMAT→ALG, 7 SGAM→FMAT, 11 NVB→ALG, 55 ALG→FMAT, 76 NVB→ALG (3/5 ALG, 2/5 FMAT). 150m CV = 0.7257 ± SE 0.0362 (vs v07 86-feat baseline 0.7317 ± SE 0.0179 — within-SE on 1-SE gate, 4/5 folds positive, fold 2 ALG-sparse outlier). Source dir: outputs/session_32/exp_seed_average_lgb_20260420_1826/ | 0.81470 |
| 36 | sub_36_bayes_prior_adjust.csv | E7: Bayesian prior adjustment on Sub 35 predict_proba. Post-hoc, no retraining. P_adj[c] = P_raw[c] × (test_prior[c]/train_prior[c]), renormalized. Test priors from Ierodiaconou 2018 Fig 7 (ALG=0.2959, FMAT=0.1735, NVB=0.3673, SGAM=0.0612, SGZ=0.1020). 1/98 predictions changed: ID 55 FMAT→ALG. LB gain +0.03480 confirms ID 55 is a public point with true label ALG. Sub 35 was wrong at ID 55 (called FMAT). | 0.84950 |
| 37 | submission_cnn_v2_ensemble.csv | CNN-v2 ensemble (ResNet18+EfficientNet-B0+ConvNeXt-Tiny), 96×96 px patches, 2ch (destriped bathy + FFT-destriped BS), 30 epochs fixed no ES, ImageNet pretrained avg-RGB→2ch conv1, BlockKFold 150m spacing=150 5-fold seed=42. Ensemble CV 0.7851±0.0691 (GATE PASS ≥0.7619). SGAM OOF F1=0.65 vs tabular ~0.01. CNN confidently wrong at ID 55 (0.979 FMAT, truth=ALG). Bayesian prior adjustment applied post-hoc (E7 ratios). | 0.79615 |
| 38 | submission_cnn_v2_ensemble_raw.csv | Same CNN-v2 ensemble, raw argmax (no Bayesian prior adjustment). Identical public score to Sub 37 confirms Bayesian adjustment flipped 0 public points — all adjustment-flipped points fell on private IDs or margins too wide. | 0.79615 |
| 39 | submission_kappa_blend_w0.4.csv | κ blend: 0.4×tabular(Sub36 recipe) + 0.6×CNN-v2 ensemble, Bayesian prior adj on blended output (E7 ratios). Canonical verde 150m BlockKFold CV=0.7648±0.0219 (PASS gate 0.7619, +1 SE above both components). McNemar vs Sub36: chi2=40.84, p=0.0000 (blend statistically better, 507 vs 322 discordant). 4 flips vs Sub36 (IDs 7,19,49 unknown truth; ID55 ALG→FMAT known wrong). SGAM count 9 vs ~1-2 Sub36. Panel 8/8 unanimous: blend is leading candidate. | 0.84950 |
| 40 | — | EXPERIMENT (not submitted): v07+acr_sapa_w21, 87 feat, no class weighting. Pre-registered gates: T1-A PASS (+0.0096), T1-B PASS (FMAT-NVB −18), T1-C FAIL (ALG −0.0189 vs threshold −0.01). Both tiers failed gate C. 150m BKF OOF: baseline 0.7349 → treatment 0.7444. acr_sapa_w21 rank #1/87 by gain (21,411); LightGBM split-attention reallocation caused ALG side effect. No submission CSV generated. See artifacts/experiment/v07_plus_acr_sapa_w21/ | — |
| 41 | submission_v07_acr_sapa_w21_alg_weighted.csv | v07+acr_sapa_w21+ALG class_weight=2.0, 87 feat. All pre-registered gates passed: T1-A PASS (+0.0161), T1-B PASS (FMAT-NVB −23), T1-C PASS (worst SGAM −0.0061). 150m BKF OOF baseline 0.7349 → treatment 0.7510. Per-class: FMAT +0.0185, NVB +0.0040, SGZ +0.0378, ALG −0.0017 (recovered from −0.0189 in Sub 40), SGAM −0.0061. acr_sapa_w21 rank #1/87 gain (23,646). Bayes prior adj (Ierodiaconou 2018 test priors). 3 flips vs Sub 36 (IDs 7→SGAM, 11→NVB, 19→ALG). See artifacts/experiment/v07_plus_acr_sapa_w21_alg_weighted/ | 0.84714 |
| 42 | submission_kappa_blend_acr_alg_weighted_w0.4.csv | κ blend substitution: 0.4×tabular(Sub41 recipe: v07+acr_sapa_w21, ALG weight=2.0) + 0.6×CNN-v2 ensemble, Bayesian prior adj on blended test output (E7 ratios). Pre-Bayes 150m BKF OOF CV=0.7688±SE 0.0133 (PASS gate ≥0.7648; both primary and secondary gates passed). Sub 39 reproduction confirmed 0.7648±SE 0.0219. SE tightened 0.0219→0.0133 (robustness improvement: new tabular OOF more consistent across spatial folds). Class dist: ALG 25, FMAT 18, NVB 40, SGAM 9, SGZ 6. 1 flip vs Sub 39: ID 76 ALG→NVB. See artifacts/experiment/kappa_blend_with_acr_alg_weighted/ | 0.84950 |

# McNemar test — Sub 42 (κ blend, lock candidate) vs Sub 36 (tabular only):
#   Date: 2026-05-02. Reference: Raschka 2018 arXiv:1811.12808v3 §4.3.
#   Discordant pairs: 1,046. n_10 (Sub 42 right, Sub 36 wrong): 619.
#   n_01 (Sub 42 wrong, Sub 36 right): 427.
#   χ² (continuity-corrected): 34.88. p-value: <0.0001. Statistically distinguishable.
#   Comparison to Sub 39 vs Sub 36 (April 23, χ²=40.84): Δχ²=−5.96.
#   Sub 42 is LESS cleanly distinguishable from Sub 36 than Sub 39 was.
#   Mechanism: ACR+ALG tabular swap creates 217 net-new disagreements with Sub 36
#   beyond Sub 39's 829. Of those ~217 extra discordant pairs, wins barely outpace
#   losses (~112 wins / ~105 losses). More total churn, more symmetric churn — the
#   new tabular component changes blend decisions in both directions.
#   Interpretation: Lock candidate's CV gain (0.7648→0.7688, +0.0040) is real but
#   fragile. ACR+ALG weighting genuinely changed blend decisions, not just confidence.
#   Caveat: OOF-level test, pre-Bayes argmax for both submissions per project convention.
#   Source: outputs/session_33/mcnemar_blend_vs_sub36/mcnemar_test.txt

# --- FACTS DERIVED FROM THIS TABLE (and only this table) ---
#
# Best LB: 0.84950 (Sub 36 2026-04-21, Sub 39 2026-04-23, Sub 42 2026-04-25 — all tied on 29-point public split). 8th place.
#   CNN-v2 (Subs 37–38): LB 0.79615. CV 0.7851±0.0691 (beats tabular CV 0.7257 by +0.06, ~1.6 SE).
#   Public LB says tabular better; CV says CNN better. 1-SE rule favors tabular (simpler). Blend (κ) is next.
#   Previous best: 0.81470 (Sub 35 — v07+T_east seed-avg LGB, 2026-04-20).
#   Best cluster: 0.81096 (Subs 2, 3, 8, 16, 18, 23, 28, 32 — all argmax-identical
#   on 29 public LB points. NOT 8 independent confirmations. Public LB = 30%
#   of 98 ≈ 29 points. Per-flip penalty ≈ 0.035 ≈ 1/29.)
#
# ID 55 CONFIRMED PUBLIC, TRUE LABEL = ALG. Sub 35 called FMAT (wrong).
#   Sub 36 single flip FMAT→ALG at ID 55 → +0.03480 on public LB. Clean 1/29 signal.
#
# Worst LB: 0.72732 (Subs 7, 10 — both KNN-based)
# New pipeline best: 0.73970 (Subs 13, 14, 15 — all new pipeline)
# Old-to-new gap: 0.81096 - 0.73970 = 0.07126
#
# KNN RECORD:
#   Sub 7:  KNN as features inside RF       → 0.72732
#   Sub 10: KNN standalone classifier        → 0.72732
#   Sub 11: KNN + LGB average ensemble       → 0.76394
#   Sub 12: KNN + LGB stacked ensemble       → 0.76214
#   Verdict: KNN hurts in every configuration tested.
#
# CLAIMS THIS TABLE DISPROVES:
#   "KNN produced 0.779→0.811" — FALSE. 0.811 was raster-only (Subs 2,3,8).
#   "KNN features were the largest improvement" — FALSE. They were the largest regression.
#   "Re-implementing KNN will get us back to 0.811" — FALSE. 0.811 never had KNN.
#
# PANEL RULING (2026-04-07, 17/17 unanimous):
#   KNN label features are a DEAD END in all tested forms:
#   as features in trees, standalone classifier, average ensemble, stacked ensemble.
#   Mechanism: model collapse + boundary-enriched test set.
#   KNN Shannon entropy approved as single-feature rider on Cooper edges session.
#   Expect nothing. If zero importance, dead-end entropy too.
#
# OPEN QUESTION:
#   Why does the old pipeline (82 feat, RF500) score 0.81096 but the new
#   pipeline (85 feat, LGB1000) scores 0.73970? The gap is 0.071. Five test
#   points disagree (IDs 7, 14, 19, 23, 46), all ALG-related. Root cause:
#   feature VALUE differences from GLCM recomputation + rebuild. Never closed.
#
# OB zonal-stats record (Session 31, 2026-04-18 update):
#   Sub 21: Felzenszwalb, 9 OB features        → 0.77696 (−0.034)
#   Sub 22: Higra dyn800+dyn1200 bundle, 22 OB → 0.79615 (−0.015)
#   Sub 34: Higra dyn800 isolated, 3 OB        → 0.76214 (−0.049)  ← worst LB
# Three LB regressions across three segmentation approaches and three feature
# counts. OB-zonal-stats paradigm closed at Refuge Cove — see
# docs/dead_ends_pending.md "OB Zonal Stats from Higra Hierarchical Watershed"
# entry for pseudoreplication mechanism and retry bars.
#
# Day 2 sweep (Subs 27-29): fold-averaging default GBMs on v07 = non-productive.
# Sub 33: class_weight='balanced' = byte-identical to baseline. Feature-bounded ceiling confirmed.
#
# Subs 30, 31, 33 exist on disk but were NOT uploaded to Kaggle.
# Sub 35 CSV is at outputs/session_32/exp_seed_average_lgb_20260420_1826/sub_35_seed_avg_lgb.csv
#   NOT in data/submissions/ — copy there if needed for reproducibility.
# Sub numbering is append-only. Next submission = Sub 43.
# Leading final-selection candidate: Sub 42 (κ blend ACR+ALG-weighted, CV 0.7688±SE 0.0133, LB PENDING).
#   Displaces Sub 39 (CV 0.7648±SE 0.0219) on CV-only lock criterion (+0.0040 CV, SE tightened).
#   Sub 39 remains best confirmed LB (0.84950) until Sub 42 scores.
#
# Per-class-pair diagnostic (2026-04-25, no submission): 10 binary LightGBM models on
#   v07(86)+3 ACR+2 HiGRA=91 features, one per unordered class pair, 150m BlockKFold.
#   OOF confusion matrix reconstructed from kappa-blend components (tab seed=42 + CNN, w=0.4).
#   Top-5 confusion pairs: FMAT→NVB 307 (19.9%), SGZ→ALG 235 (28.5%), FMAT→ALG 169,
#   NVB→ALG 157, FMAT→SGAM 145. Headline: acr_sapa_w21 ranks #1 gain on FMAT-NVB,
#   FMAT-SGZ, NVB-SGZ while absent from global v07 model. HiGRA segment-area features
#   rank 3–10 in 7 pairs. Motivated Subs 40–41. See artifacts/diagnostic/per_pair_importance/
#
# Sub 41 LB (2026-04-25): 0.84714 vs Sub 36/39 best 0.84950, delta −0.00236.
#   Within noise floor (per-flip penalty ~0.034 on 29-point public split).
#   Public LB cannot discriminate Sub 41 from Sub 36/39 at this magnitude.
#   Tier 2-D (private LB confirmation): structurally unavailable — Matteo not registered
#   for 2026 GeoHab conference. Tier 2 assessed on CV alone: Sub 41 CV 0.7510 trails
#   Sub 39 CV 0.7648 by −0.0138. Sub 39 was lock leader at Sub 41 submission time;
#   displaced same day by Sub 42 (CV 0.7688 ± SE 0.0133, LB 0.84950 — tied on public split).
#
# Sub 42 LB (2026-04-25): 0.84950 (public). Ties Sub 36 and Sub 39 on 29-point public split.
#   1 flip vs Sub 39: ID 76 ALG→NVB. ID 76 is inferred private: public score did not change
#   despite the flip (identical 0.84950), consistent with the Sub 35 flip analysis where
#   ID 76 was among 4 non-ID-55 flips that produced no public LB movement.
#   CV: 0.7688 ± SE 0.0133 vs Sub 39 0.7648 ± SE 0.0219 (+0.0040 mean, SE tightened by 0.0086).
#   Sub 42 is the lock candidate under CV-only framework (highest 150m CV by May 4).
#
# Lock framework (2026-04-25): Matteo not registering for 2026 GeoHab conference;
#   will not receive private LB. Lock for final submission = highest 150m BlockKFold CV
#   by May 4, no public-LB tiebreaker. Public LB demoted to decoration at noise-floor
#   differences (per-flip penalty ~0.034 on 29-point public split).
#
# Post-deadline isolation tests and ecological investigations (no Kaggle submissions):
#
# y-only — Raw northing (y) as isolation feature (2026-04-27):
#   Pre-registration: docs/y_only_test/preregistration.md, commit 42ac797.
#   Baseline (87 feat) OOF 0.7510 → Treatment (88 feat, +y) OOF 0.7830, ΔF1 +0.0321.
#   P2 collapsed: y ranks #1 importance (mean ΔF1 0.2292), 9/10 top baseline features
#   lost >=50% importance. Verdict: spatial-interpolator collapse. Dead end confirmed.
#   Artifacts: outputs/y_only/. Run commits: 42ac797 → ee690b0 → 3d16770.
#
# CNN backbone reweight — Non-uniform weight sweep on CNN-v2 OOF (2026-05-02):
#   27 weight triples (w_r, w_e, w_c) from {0.1…0.6}^3 summing to 1.0.
#   Gate = uniform CNN baseline + 1 SE = 0.7254 + 0.0404 = 0.7658.
#   Best non-uniform: (0.3, 0.3, 0.4) CV=0.7293. 0/27 triples pass gate.
#   SGAM flat across entire sweep (0.214–0.231). Dead end: uniform (1/3,1/3,1/3) is optimal.
#   Artifacts: outputs/session_33/exp_cnn_backbone_reweight_20260502_0127/
#
# Post-deadline ecological investigations (no Kaggle submissions):
#
# ECO-1 — Depth-threshold at contested IDs (2026-04-27):
#   Hypothesis: contested test IDs (7, 14, 19, 23, 46) sit at NVB<->photosynthetic
#   depth boundaries exploitable by depth-threshold features. Halted at pre-flight
#   scope check — none of 6 contested points lie at class-transition depth zones.
#   Setup-fit failure. No K-nearest run. Verdict: dead end for this dataset.
#
# ECO-2 — Substrate-via-rugosity + creek-mouth proximity (2026-04-27–29):
#   Pre-registration: docs/eco/idea1_creek_mouth_preregistration.md (commit bd88417).
#   Treatment: v07 + acr_sapa_w21 + dist_to_creek_mouth (88 feat), ALG weight=2.0.
#   Baseline OOF cross-check: 0.7510 (exact match). Treatment OOF: 0.7954 (+0.044).
#   Spatial audit: dist_to_creek_mouth is a northing alias (Pearson r=0.990 vs y,
#   R2=0.980 on y alone). Global +128 SGZ recovery driven by Cluster 0 (north margin,
#   665 rows); south-margin recall (158 rows) fell 0.101→0.038 (net delta=−10).
#   3-feature K-nearest (rugosity SD, ms_roughness, bathy) at IDs 14+23 vs ALG OOF
#   candidates: partially consistent at ID 14 (ms_roughness positive all K), partially
#   consistent at ID 23 (rugosity SD positive all K); no clean dual-feature agreement.
#   kth_zdist 0.4–1.8, no manifold flag. Published as Kaggle discussion post.
#   Artifacts: outputs/eco/, scripts/eco_*.py, docs/eco/.
#   P3 (SHAP direction) deferred — shap conda install would upgrade numpy 1.26→2.4.
#
# UNREGISTERED FILE — NEEDS MATTEO VERIFICATION:
#   data/submissions/EXTERNAL_submission_lgbm_oof_preds - 0.81525.csv
#   Filename encodes LB score 0.81525 (> Sub 35 = 0.81470). NOT in this registry.
#   Rule: no row enters without Kaggle submission page or paste-after-submit.
#   Action required: Matteo to confirm whether this was submitted to Kaggle and
#   what Sub number it corresponds to. If submitted, add a registry row. If not
#   submitted, update filename to make clear it was not uploaded.
#
# === STACKING RETIREMENT — three-variant cumulative result (Session 33N, 2026-05-02) ===
# Three architectural variants of stacking with LR meta tested on this dataset:
#
#   Variant              | CV mean ± SE        | Gap to gate (0.7619)
#   ---------------------|---------------------|---------------------
#   Session 31 (5-BL)    | 0.6424 ± 0.0463     | −0.0833
#   FDS 4-BL             | 0.5893 ± 0.0443     | −0.1726
#   FDS-min 2-BL         | 0.5790 ± 0.0600     | −0.1829
#
# All three variants fail by 0.08 to 0.18 below the single-LGB baseline gate.
# Failure mode consistent across architectures: LogisticRegression meta with
# use_features_in_secondary=True cannot selectively gate weak base-learner
# inputs while preserving strong ones. Diversity was achieved in FDS variants
# (max pairwise dcor 0.41, all argmax agreement <=63%) — the failure is
# meta-learner expressivity, not base-learner setup.
#
# Session 31 variant: GBM trio (LGB/XGB/CatBoost) dcor 0.88–0.93, argmax
# agreement 90–93% — diversity was additionally not achieved in that variant.
#
# None of the three variants generated a Kaggle submission.
# See dead_ends.md "Stacking with logistic-regression meta — RETIRED across
# three architectures" for full post-mortem and retry conditions.
