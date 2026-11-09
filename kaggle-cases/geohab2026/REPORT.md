# GeoHab 2026 MLWG — AutoSOTA research report

**Archive date: November 9, 2026.** This English report summarizes the supplied
Chinese research report and the two recorded AutoSOTA phases. Scores describe
historical experiments, not new evaluations performed during integration.

## Task and results

The task is five-class benthic habitat classification from multibeam sonar at
Refuge Cove, Australia. Inputs include bathymetry, backscatter, and derived terrain
features. The supplied report describes 6,256 labeled training points and 98 test
points, with weighted F1 as the competition metric.

| Evaluation | Public reference | AutoSOTA final |
|---|---:|---:|
| Five-fold spatial CV, mean weighted F1 | 0.7688 | **0.7957** |
| Public leaderboard | 0.84950 | **0.88296** |
| Private leaderboard | 0.85811 | **0.88691** |

The final submission reference is **57130319**, submitted on October 13, 2026.
It is a **late submission**. Its private score exceeds the historical leading
score in the supplied comparison, but it does not establish a new official first
place or alter the frozen standings. The reported public position would be a tie
for second if inserted by score; it is not an official awarded rank.

## Baseline and evaluation protocol

The starting point is Matteo Niccoli's public mycarta solution: a LightGBM model
using 87 features, a frozen CNN ensemble, and a test-only ecological prior.
The CNN ensemble combines ResNet18, EfficientNet-B0, and ConvNeXt-Tiny. The
reference blend assigns 0.40 to LightGBM and 0.60 to the CNN probabilities.

Validation uses five frozen `verde.BlockKFold` folds with 150 m spatial blocks
and seed 42. The main optimization metric is the mean of the five weighted-F1
scores, with pooled F1 reported separately. Random folds would overstate
performance because nearby observations share spatial context and labels.

The E7 ecological prior is applied only to test predictions, never to the OOF
validation metric. The supplied report states that test labels were unavailable.
The tiny public/private test subsets make small score differences unstable.

## Two-stage search

Phase 2 starts from the selected Phase 1 checkpoint, not from the original public
reference. Only the final Phase 2 source snapshot is included in this archive.

| Phase | Change | CV weighted F1 | Decision |
|---|---|---:|---|
| 1 | Reproduce the public reference | 0.7688 | Baseline |
| 1 | Increase LightGBM `min_child_samples` from 20 to 100 | 0.7785 | Retained |
| 1 | Sharpen tabular probabilities with exponent 6 before fusion | 0.7831 | Retained; Phase 1 endpoint |
| 1 | Sentinel-value change, fewer trees, and `boost_from_average=False` | At or below 0.7831 | Rejected or reverted |
| 2 | Reproduce the Phase 1 endpoint | 0.783125 | Phase 2 baseline |
| 2 | Class-specific reliability-weighted fusion | 0.7877 | Retained |
| 2 | Spatial smoothing restricted to the appropriate fold | 0.7951 | Retained |
| 2 | Increase the tabular sharpening exponent from 6 to 12 | **0.7957** | Final selected state |
| 2 | Further fusion/smoothing candidates | Higher apparent scores but failed robustness rules | Rejected |

The capacity change was motivated by severe in-sample overfitting and near-duplicate
spatial neighborhoods. Phase 2 then addressed class-dependent reliability and
spatial consistency rather than adding a new learned backbone.

## Final method

The selected method combines the LightGBM and frozen CNN probability branches,
sharpens the tabular branch with exponent 12, applies class-specific reliability
weights, and smooths predictions spatially using the fold-aware rule. Test
predictions use nearby training points for spatial context, then receive the
test-only E7 prior correction before `argmax`.

The original pipeline separates live LightGBM fitting from externally supplied
CNN probability artifacts. Those artifacts are **not included** here. Reproducing
the recorded scores requires preparing the original raster/feature inputs, frozen
fold assignments, and compatible CNN outputs, or regenerating them through the
supplied model code.

## Reproduction and limitations

The implementation is under [`code/`](code/). Its reproduction notes describe the
historical container and `/paper_data` layout. Historical commands include
`python pipeline/run_pipeline.py --out-dir outputs` and the separate official OOF
scorer. These commands require external inputs; this GitHub archive contains no
data, stored probabilities, model weights, or predictions.

The original report records a CV gain of approximately 0.0269 over the reference.
Repeated development on the same folds and the very small final test set limit
claims of independent generalization. Historical artifact paths in the retained
code notes are references to external experiment files.

The public baseline and its original training remain attributed to their authors.
AutoSOTA's contribution is the recorded capacity adjustment, fusion, spatial
smoothing, candidate testing, and final selection. No training or competition
submission was run while preparing this archive.
