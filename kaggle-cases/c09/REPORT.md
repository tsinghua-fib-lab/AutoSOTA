# Final Report: C09 — Open Problems: Single-Cell Perturbations

- Primary metric: mean row-wise root-mean-square error (MRRMSE; lower is better)
- Final submitted method: consensus-relative residual transfer with B-cell correction and Myeloid abstention (RAMRT)
- Final submission reference: `57126892`
- Final scored submission: **Public 0.521; Private 0.721**
- Official historical winner: Public 0.523; Private 0.729

## Best Result

| Submission | Public MRRMSE | Private MRRMSE | Decision |
| --- | ---: | ---: | --- |
| Official winner | 0.523 | 0.729 | Historical reference |
| **Final RAMRT B correction (`57126892`)** | **0.521** | 0.721 | Below the official winner on both splits |

The final submission improves on all three official top-ranked entries on both leaderboards. Their public scores were 0.523, 0.523, 0.526; their private scores were 0.729, 0.729, 0.732. The final RAMRT method applies the B-cell correction supported by compound-held-out evidence. Its score record is in `best_code/evidence/`; the candidate and audit are in `best_code/final_submission/`.

## Task and evaluation

The task predicts signed gene-expression significance for unseen cell-type–compound conditions. The observed data comprise 614 conditions across six cell types; the test set contains 128 B-cell and 127 Myeloid conditions, each with 18,211 gene values. Four well-observed source contexts are NK, CD4 T, CD8 T, and Treg. Only 17 compounds have target labels for each of B cells and Myeloid in the compound-held-out validation used here.

Leaderboard MRRMSE evaluates a submitted test matrix. Local cross-validation (CV) MRRMSE is the mean of per-row RMSE across 34 held-out compound–target predictions, 17 per target. A positive `delta` in the experiment records means `baseline MRRMSE − candidate MRRMSE`. Different experiment families use different local baselines and folds; their deltas must be read within their own protocol, not ranked as one common leaderboard.

## Research trajectory and decisions

### 1. Source-response modeling before the final gate

The modeling search compared direct reconstruction and residual transfer against source-cell anchors. Unless noted, the values below are local compound-held-out results, not Kaggle leaderboard scores.

| Idea / experiment | Main result | Decision | Record |
| --- | --- | --- | --- |
| Masked-cell neural transfer v2 | B delta −0.272 on 15 complete compounds; Myeloid −0.235 | Rejected: transfer harmed both targets | `masked_cell_transfer_v2_results.json` |
| Source-residual PCA plus masked Tucker completion | Combined delta −0.0152 on 17 compounds; B −0.0092, Myeloid −0.0211 | Rejected | `tucker_completion_cv_results.json` |
| Strength-normalized residual DNPP | Combined delta +0.0416, but 95% CI [−0.0826, 0.1645]; Myeloid improved on 6/17 compounds | Rejected by uncertainty / stability gate | `dnpp_residual_cv_results.json` |
| Source-PCA features plus componentwise TabPFN v2 residual prediction | Versus adaptive source anchor: B +0.1398 (14/17 wins), Myeloid +0.1992 (13/17); combined +0.1695 | Promising residual direction; carried forward | `tabpfn_component_cv_results.json` |
| Third-place architecture plus frozen TabPFN correction | Combined improvement at alpha 0.05 only +0.000088; alpha 0.1 and 0.2 worsened | Avoid broad fixed correction | `third_place_bridge_cv_results.json` |

The three TabPFN seeds were 17, 29, and 43; their combined anchor-relative CV deltas were +0.16951, +0.16065, and +0.13342, respectively. The per-seed result files are in `best_code/experiments/`. The independently evaluated bridge experiment found that a small correction could help B cells, while the target-specific behavior argued against automatically correcting Myeloid.

### 2. Reliability gates, target abstention, and failed variants

The next experiments used nested compound-held-out CV against a strong frozen neural OOF base. They examined whether source agreement, response distance, response strength, and target-level activation could select safe correction magnitudes. The full JSON records include fold assignments, candidates, per-compound errors, seed outcomes, and gate decisions.

| Variant | Nested combined delta | Key target findings | Outcome |
| --- | ---: | --- | --- |
| RAMRT joint adaptive correction | −0.00030 | B +0.02147; Myeloid −0.02206 | Rejected: Myeloid negative transfer erased B gain |
| Target-level abstention variant | +0.00576 | B +0.01153; Myeloid off; combined CI lower bound −0.00026 | Rejected by its nested acceptance gate |
| Bayesian target-activation variant | +0.00010 | B +0.01160; Myeloid −0.01140; one seed negative | Rejected by its nested acceptance gate |
| Masked-cell empirical-Bayes transport (MEBT) | −0.00522 (seed 17) | Inner folds mostly chose zero blend; no stable gain | Rejected |
| Metric-geometric cross-fit calibration (MGCC) | −0.04668 (seed 17) | B −0.10230; Myeloid +0.00895 | Rejected: geometry calibration harmed B |

The final target-level decision retained the B-cell correction and set the Myeloid coefficient to zero. The final-candidate audit recomputes B-cell OOF delta **+0.02146665**, 10/17 compounds improved, 95% compound-bootstrap CI **[0.001950, 0.046251]**. All three seed-specific B deltas were positive: +0.02010, +0.02286, +0.02125. The mean combined delta was +0.01073 with Myeloid frozen. This final-candidate audit uses the selected target-level rule on preserved OOF corrections; the stricter fully nested gate-selection variants above are reported separately.

#### MOSAIC transfer-gain route

MOSAIC tested program-wise, baseline-relative transfer-gain prediction using source-cell pseudo-target tasks. Source-context validation was positive, but both real-target evaluations failed, so the method was rejected.

| MOSAIC experiment | Result | Decision |
| --- | --- | --- |
| Pseudo-target gain feasibility | Source-only lower-quantile gate improved all four held-out contexts; mean delta +0.01757, harm rate 7.73% | Advance to real-target check |
| Normalized action selection | Four source contexts improved; mean delta +0.01753 | Advance to real-target check |
| First real-target OOF test | B delta −0.02814; Myeloid −0.43305; combined −0.23059 | Rejected: negative transfer |
| Isomorphic TabPFN validation | Four source contexts improved (mean +0.01004); real-target B delta −0.00033, Myeloid −0.00347 | Rejected: no demonstrated target benefit |

The source-only results did not establish improvement on the target cells. The MOSAIC scripts and result records are in `best_code/experiments/`.

### 3. Final method and submission

For each compound, the method builds a source-response consensus from NK, CD4 T, CD8 T, and Treg measurements. The source response is projected to 16 PCA components per source cell and augmented with four observation masks and four response-strength features, yielding 72 input features. For target-labeled compounds, the supervision is `target response − source consensus`; an eight-component target PCA represents this residual. Componentwise TabPFN v2 regressors predict the residual, averaged across seeds 17, 29, and 43.

The final correction is added to the *frozen submission*, rather than replacing it with a reconstructed full target response. For B cells, a bounded strength-based coefficient ranges from 0.05 to 0.10 (mean 0.07166 in the submitted test set). For Myeloid, the coefficient is zero because validation exposed negative transfer. The final candidate changed all 128 B-cell test rows and left other predictions unchanged up to CSV round-trip precision (maximum absolute non-B difference 3.39e-21). Its audit verifies 255 unique IDs, finite values, a valid ZIP, and the submitted artifact SHA-256 `a2f9273f208a1ce04f4ee15acd2680a9f70db8136a0874bba60edfcd70fba1ae`.

`best_code/experiments/ramrt_nested_cv.py` contains the core representation, correction, CV and gate code. `best_code/experiments/build_ramrt_b_abstention_candidate.py` applies the accepted B-only rule and writes the submission. Supporting experiment scripts and their results are in `best_code/experiments/` and `best_code/evidence/`.

## Controlled ablations after method selection

A matched five-fold compound-grouped experiment held the source features, fold assignments, TabPFN estimator count, eight-component target space, and direct correction strength fixed; only the predicted output representation changed. Combined MRRMSE is the arithmetic mean of the B and Myeloid row-wise errors over 34 held-out predictions.

| Output representation | Combined MRRMSE | B | Myeloid |
| --- | ---: | ---: | ---: |
| Direct full-target response | 2.661 | 2.389 | 2.934 |
| Residual to source mean | 2.986 | 2.122 | 3.849 |
| Residual to source median | 2.971 | 2.322 | 3.620 |
| **Residual to adaptive consensus** | **2.481** | **1.879** | 3.083 |

Adaptive consensus lowered combined MRRMSE relative to mean-residual by 0.504 (95% compound-bootstrap CI [0.101, 1.003]) and relative to median-residual by 0.490 ([0.036, 1.108]). The gain over direct full-target prediction was 0.180, with a CI spanning zero. `best_code/experiments/tabpfn_representation_ablation.py` and `best_code/experiments/tabpfn_representation_ablation_results.json` contain the matched protocol and results.

A separate reanalysis of fully nested OOF predictions found that a fixed 5% B correction already improved B MRRMSE by 0.01215 (10/17 wins), while applying it to both targets gave a near-zero combined gain because Myeloid worsened. The complete bounded adaptive decision also failed its fully nested combined gate. This supports the consensus-relative representation and target-selective application as the key findings; the adaptive strength coefficient is an operational choice, not a separately established source of improvement. See `best_code/experiments/targeted_ablation_results.json`.

## Additional ideas

The method-development notes considered the following twelve families. Their status distinguishes tested candidates from ideas that remained proposals.

| Proposed idea | Status in this case |
| --- | --- |
| Cross-fitted empirical-Bayes shrinkage over TabPFN residuals | Tested through MEBT, RAMRT, and Bayesian abstention variants; no fully nested acceptance |
| Target-aware shared B/Myeloid shrinkage | Tested; Myeloid negative transfer led to target-level abstention |
| Nonnegative ensemble of TabPFN, DNPP, and latent-additive corrections | Proposed; not tested |
| Cross-fitted prediction-interval uncertainty gate | Proposed; not tested |
| OOF residual learner on a reproduced winner / third-place proxy | Third-place bridge tested; gains negligible |
| Cell-type graph prior | Proposed; not tested |
| Drug-response gene modules / pathway residual PCA | Tested through MOSAIC program-level routing; target gate failed |
| Nearest-neighbor retrieval in source-response geometry | DNPP residual tested; stability gate failed |
| SMILES/RDKit-conditioned TabPFN or local metric learning | Proposed; not tested |
| Low-rank-plus-sparse response decomposition | Tucker/low-rank route tested and rejected; sparse variant not tested |
| Multi-task B/Myeloid residual transfer | Tested through joint RAMRT, MEBT, MGCC; target-wise outcomes disagreed |
| Conformal correction acceptance / abstention | Proposed; not tested |


## Code and result map

- `best_code/experiments/`: final RAMRT build/CV code, modeling and ablation scripts, and their result files.
- `best_code/evidence/`: final submission score record and non-submission method audits.
- `best_code/final_submission/`: the frozen base, final submitted candidate ZIP, and final candidate audit.
- `best_code/data/id_map.csv`: public test-row metadata used to apply B-only corrections.

The competition training parquet and third-party TabPFN checkpoint are external inputs; their expected locations and runtime dependencies are described in `best_code/README.md`.
## GitHub archive scope

This archive contains code and this English report only. The submission ZIPs,
OOF prediction matrices, test-row mapping CSV, and raw experiment/evidence JSON
files named above are not included. Their paths refer to the supplied handoff.
No new Kaggle submission or model evaluation was performed during integration.
