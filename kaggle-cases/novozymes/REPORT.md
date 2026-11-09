# Novozymes Enzyme Stability Prediction — AutoSOTA research report

**Archive date: November 9, 2026.** This English report summarizes the supplied
Chinese report and the paper-9003 iteration record. The scores are historical.

## Task and final results

The competition predicts enzyme melting temperature from amino-acid sequences
and experimental context. The metric is Spearman rank correlation, so relative
ordering matters more than temperature calibration. The supplied report describes
31,390 original training rows and 2,413 test variants of the same enzyme.

| Evaluation | Reference / reproduced baseline | AutoSOTA final |
|---|---:|---:|
| Released full-test Spearman, historical container | 0.55785 | **0.59599** |
| Released full-test Spearman, historical local environment | 0.55569 | **0.59420** |
| Public leaderboard | 0.58917 | **0.60035** |
| Private leaderboard | 0.53258 | **0.58370** |

The final submission reference is **57143817**, submitted on October 13, 2026.
The historical winner's private score in the supplied report is 0.54541. The
AutoSOTA late-submission score is higher, but it does not change the frozen
official standings or award a new competition rank. Local, full-test, and
private-subset scores must not be interchanged.

## Starting point and evaluation exposure

The technical baseline is Laura Romar's public seventh-place solution. Its
original rank ensemble combines SASA, LightGBM, ThermoNet, B-factor, and Rosetta
components with weights 0.10 / 0.30 / 0.20 / 0.10 / 0.30.

The train/test distributions differ substantially: training spans many proteins
and data sources, while the test set contains single-mutant variants of one
enzyme. The LightGBM component uses Jin ddG data and wild-type-grouped folds.
Structural and frozen upstream component predictions provide complementary
information.

The project evaluated candidates against the **released full-test labels** through
the scorer. Even if those labels did not enter base-model gradient training,
their aggregate feedback informed the iterative search and ensemble selection.
These are benchmark-adaptive results, not an untouched independent test.

## Search history

The project first deferred the competition over label and late-submission
availability concerns. After GeoHab demonstrated a late-submission route and
released test labels were located, one technical AutoSOTA chain, paper-9003,
was completed. It is not a second optimization phase built on a separate
Novozymes AutoSOTA run.

| Iteration | Main change | Full-test Spearman | Decision |
|---|---|---:|---|
| 0 | Reproduced five-component rank ensemble | 0.557850 | Baseline |
| 1 | Refit rank-average weights | 0.561880 | Retained |
| 2 | Add four label-free AlphaFold2 structural components | 0.580100 | Retained |
| 3 | Add neighborhood-density information; refit 11 components | 0.587550 | Retained |
| 4 | Replace raw rank pooling with van der Waerden normal scores | 0.593260 | Retained |
| 5 | Add a LightGBM twin without the distribution-shifted `group_size` feature | **0.595990** | Final method |
| 6 | Refit using previously rejected components | 0.595990 retained | Apparent gain rejected as search overfitting |
| 7 | Sequence-conservation components | 0.595990 retained | Rejected |
| 8 | Target transforms and exposure-weighted features | 0.595990 retained | Rejected |
| 9 | Commensurate neighborhood-density feature in LightGBM | 0.595990 retained | No effective change |

The supplied final verification reran the unchanged scoring command and reproduced
0.595990. The repeated values in rejected rows denote the retained final state,
not independent successful candidates.

## Selected method

The final method pools 11 complementary components after mapping each component's
ranks to normal scores, `Phi^-1((rank − 0.5) / n)`. Added AlphaFold2 packing,
mutant/WT accessibility, and neighborhood components complement the public
sequence/structure ensemble. The corrected LightGBM twin removes `group_size`,
whose meaning transfers poorly to the single-enzyme test distribution.

The reference and AutoSOTA hyperparameters are retained in the source code and
the small selected ensemble configuration. Derived fitted-weight dumps,
structural datasets, predictions, PDB files, and test labels are excluded.

## Reproduction and qualifications

Use [`code/`](code/) and its reproduction notes. The historical entry point is
`bash scripts/eval.sh`, which runs `pipeline/run_pipeline.py` with external
`/paper_data` inputs before invoking the scorer. External prerequisites include
the competition data, released labels for scoring, AlphaFold2 structures, Jin ddG
inputs, and upstream component predictions.

The recorded container/local difference is partly associated with numerical and
thread-count differences in LightGBM; report the environment with the score.
The 0.55785 → 0.59599 gain is about 6.84%, but repeated full-test feedback means
it cannot establish independent generalization or statistical significance.

The public baseline and external component models retain their original credit.
No new model training, scoring, or Kaggle submission was performed during this
repository integration.
