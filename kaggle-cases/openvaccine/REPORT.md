# OpenVaccine — AutoSOTA research report

**Archive date: November 9, 2026.** This English report summarizes the supplied
OpenVaccine report and transcribes its experiment and blend scores below. The
original Chinese report and raw result files remain outside this repository.

## Task and final result

OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction asks for position-wise
RNA reactivity and degradation predictions. Inputs include sequence, secondary
structure, loop types, and derived base-pair probabilities. Five targets are
predicted, but the competition averages RMSE over three scored targets:
`reactivity`, `deg_Mg_pH10`, and `deg_Mg_50C`. Lower MCRMSE is better.

| Method | Public MCRMSE | Private MCRMSE |
|---|---:|---:|
| Historical winning solution | 0.22815 | 0.34198 |
| Reproduced baseline, s2 | 0.23321 | 0.34382 |
| First cross-family pseudo-label round, R1 | 0.23005 | 0.34038 |
| Second cross-family round, R2 | 0.22922 | 0.33945 |
| Final s5 + t10, local evaluation | 0.22860 | 0.33918 |
| **Final evaluated Kaggle submission** | **0.22859** | **0.33917** |

Submission **57170018** was evaluated on October 14, 2026, after the competition
deadline. The private error is 0.00281 lower than the historical winner's, while
the public error is 0.00044 higher. It does not change the historical official
standings. The private improvement is approximately 0.82% against the historical
winner and 1.35% against the reproduced baseline.

## Data and reconstruction

The supplied report describes 2,400 training sequences of length 107, with labels
for the first 68 positions; 1,589 pass the high-quality `SN_filter`. The public
set contains 629 length-107 sequences. The private scoring set contains 1,172
quality-filtered length-130 sequences, scored over the first 91 positions.

Data were reconstructed from public DasLab resources. The project checked the
reconstructed scoring cohort against the historical winning submission and
recovered its private score of 0.34198. It also corrected corrupted loop-type
annotations, marked duplicated low-quality Mg-target measurements as missing,
and fixed the mismatch between model-output and Kaggle-column order.

Training/public secondary structures and private structures came from different
folding pipelines. Directly switching all inputs to EternaFold degraded the
public score, so that candidate was not retained. Final models were trained from
scratch; organizer-released winner checkpoints were used only to diagnose the
feature reconstruction and were not part of the final ensemble.

## Baseline and selection protocol

The starting model follows Jiayang Gao / Nullrecurrent's first-place solution:
55-dimensional node features, eight adjacency channels, denoising-autoencoder
pretraining, a graph/attention/convolution trunk, and GRU, LSTM, feed-forward, or
WaveNet heads. Training uses noise-aware target masking, cluster weighting,
target perturbation, sequence reversal, and alternating supervised/pseudo-label
segments with rollback.

Five-fold validation groups similar sequences by edit-distance clusters. The
source report explicitly notes that pseudo-label teachers trained on all folds
can introduce validation leakage, making pseudo-label CV optimistic. Subsequent
pseudo-label rounds and blends were selected using **public-set** performance;
the source states that private-set values were reserved for reporting. This is
benchmark-adaptive development, not evaluation on a newly untouched benchmark.

The same-family second pseudo-label round improved CV but worsened the private
score, motivating a more diverse teacher rather than indefinite self-training.

## Retained method

The added structure-aware Transformer uses pair biases derived from adjacency
channels, dynamic relative-position bias, optional graph message passing, and
convolutional sublayers. Individual Transformer models were weaker than the
Keras baseline, but their errors were complementary. Small-model and dropout-only
changes did not solve that gap; learning-rate changes and graph variants were
more useful.

The first cross-family teacher combines s1 Keras and t3 Transformer predictions
with weights 0.7 and 0.3. That teacher trains s4, t4, and t5. Their blend forms
R1: s4 0.5 + t4 0.25 + t5 0.25. R1 then teaches s5 and t10. Transformer t10 is
initialized using t9 masked-node pretraining.

The final prediction is **0.5 × s5 + 0.5 × t10**, comprising 20 Keras models
(four heads × five folds) and 25 Transformer models (five variants × five folds).
The final per-target private RMSEs are 0.25147, 0.45268, and 0.31339, compared
with baseline s2's 0.25752, 0.45650, and 0.31743.

The report retains negative findings: same-family repeated pseudo-labeling,
unified EternaFold structures, and some alternative teacher choices failed the
selection protocol. Public and private preferences did not always agree.

## Reproduction

The supplied implementation is under [`code/`](code/), including model definitions,
feature reconstruction, training, scoring, blending, and all 18 YAML experiment
configurations. Recreate the external dataset with `code/scripts/build_data.py`
and prepare EternaFold before running the training pipeline. The original code
expects an appropriate external TensorFlow environment and a separate PyTorch
interpreter; adjust the historical interpreter paths to the local installation.

From `code/`, the selected dependency chain is:

```bash
python scripts/train.py --config configs/s1_nopl.yaml
python scripts/train.py --config configs/t3_sweep.yaml
python scripts/train.py --config configs/t9_pre.yaml
python scripts/train.py --config configs/s4_pl_div.yaml
python scripts/train.py --config configs/t4_tfm_pl.yaml
python scripts/train.py --config configs/t5_tfm_pl_seed.yaml
python scripts/train.py --config configs/s5_pl2.yaml
python scripts/train.py --config configs/t10_tfm_pl2_pre.yaml
python scripts/blend.py s5_pl2:0.5 t10_tfm_pl2_pre:0.5 --out runs/final/test_pred.json
```

The source report records 18 experiment groups, 372 training jobs, 31 blend
experiments, and one late submission. Original hardware was eight A800 80 GB GPUs;
total task runtime was approximately 207.4 hours. These are historical resource
records, not integration-session measurements.

## Source-only archive

Training/test JSON, released test labels, random pseudo-label sequences, model
weights, predicted values, submission files, result JSON/CSV, and figures are
excluded. Tables below preserve the supplied score summaries as Markdown. No
model training, scoring, or Kaggle submission was performed during integration.

## Appendix A: all 18 recorded experiment groups

| Run | Family | CV | Public | Private |
|---|---|---:|---:|---:|
| s1_nopl | Keras | 0.21964 | 0.23636 | 0.34767 |
| s2_pl | Keras | 0.20111 | 0.23321 | 0.34382 |
| s3_pl | Keras | 0.19467 | 0.23332 | 0.34493 |
| e1_ef_nopl | Keras | 0.22218 | 0.23702 | 0.34663 |
| t1_tfm | Transformer | 0.23896 | 0.24881 | 0.35417 |
| t2_sweep | Transformer | 0.23059 | 0.24628 | 0.35122 |
| t3_sweep | Transformer | 0.22559 | 0.24128 | 0.34725 |
| t4_tfm_pl | Transformer | 0.20886 | 0.23555 | 0.34275 |
| t5_tfm_pl_seed | Transformer | 0.20698 | 0.23566 | 0.34267 |
| s4_pl_div | Keras | 0.20184 | 0.23126 | 0.34231 |
| t6_tfm_pl2 | Transformer | 0.20085 | 0.23505 | 0.34192 |
| t7_tfm_pl_ef | Transformer | 0.20859 | 0.23579 | 0.3425 |
| s5_pl2 | Keras | 0.19793 | 0.22959 | 0.3407 |
| t8_tfm_pl3 | Transformer | 0.1981 | 0.23532 | 0.34221 |
| s6_pl3 | Keras | 0.19574 | 0.22938 | 0.34024 |
| t9_pre | Transformer | 0.22475 | 0.23967 | 0.34561 |
| t10_tfm_pl2_pre | Transformer | 0.20057 | 0.23369 | 0.34145 |
| s5b_pl2_seed | Keras | 0.19659 | 0.22998 | 0.34107 |

## Appendix B: all 31 recorded blends

| Index | Blend | CV | Public | Private |
|---:|---|---:|---:|---:|
| 1 | s2 + s3（1:1） |  | 0.23291 | 0.34408 |
| 2 | s1 + s2 + s3 (equal weights) |  | 0.23263 | 0.34418 |
| 3 | s1 0.7 + t3 0.3 (first-round teacher) | 0.21658 | 0.23395 | 0.34486 |
| 4 | s2 0.5 + t4 0.5 |  | 0.23032 | 0.34059 |
| 5 | s2 0.6 + t4 0.4 |  | 0.23024 | 0.3408 |
| 6 | s2 0.4 + t4 0.6 |  | 0.23072 | 0.34059 |
| 7 | s2 0.4 + s3 0.2 + t4 0.4 |  | 0.22992 | 0.34074 |
| 8 | t4 0.5 + t5 0.5 |  | 0.23545 | 0.3426 |
| 9 | s2 0.5 + t4 0.25 + t5 0.25 |  | 0.23031 | 0.34054 |
| 10 | s2 0.4 + s3 0.2 + t4 0.2 + t5 0.2 |  | 0.22992 | 0.3407 |
| 11 | s4 0.5 + t4 0.5 |  | 0.23006 | 0.34043 |
| 12 | R1：s4 0.5 + t4 0.25 + t5 0.25 |  | 0.23005 | 0.34038 |
| 13 | s4 0.6 + t4 0.2 + t5 0.2 |  | 0.22976 | 0.34044 |
| 14 | s4 0.5 + s2 0.2 + t4 0.15 + t5 0.15 |  | 0.22987 | 0.34075 |
| 15 | s4 0.35 + s2 0.25 + t4 0.2 + t5 0.2 |  | 0.22989 | 0.34051 |
| 16 | s4 0.3 + s2 0.2 + t4 0.25 + t5 0.25 |  | 0.2301 | 0.3404 |
| 17 | s4 0.3 + s2 0.15 + s3 0.15 + t4 0.2 + t5 0.2 |  | 0.22968 | 0.34049 |
| 18 | R2：s5 0.5 + t6 0.5 | 0.19491 | 0.22922 | 0.33945 |
| 19 | s5 0.6 + t6 0.4 | 0.19479 | 0.2288 | 0.3394 |
| 20 | s5 0.5 + t6 0.25 + t4 0.125 + t5 0.125 |  | 0.22916 | 0.33946 |
| 21 | s5 0.3 + s4 0.2 + t6 0.3 + t4 0.1 + t5 0.1 |  | 0.22929 | 0.33961 |
| 22 | s5 0.25 + s4 0.25 + t6 0.25 + t4 0.125 + t5 0.125 |  | 0.22936 | 0.3397 |
| 23 | R3：s6 0.5 + t8 0.5 | 0.1925 | 0.22941 | 0.33949 |
| 24 | s6 0.5 + t6 0.5 |  | 0.22924 | 0.33923 |
| 25 | s5 0.25 + s6 0.25 + t6 0.25 + t8 0.25 |  | 0.22917 | 0.33934 |
| 26 | s5 0.25 + s6 0.25 + t6 0.5 |  | 0.22919 | 0.3393 |
| 27 | Final: s5 0.5 + t10 0.5 |  | 0.2286 | 0.33918 |
| 28 | s5 0.5 + t6 0.25 + t10 0.25 |  | 0.22881 | 0.33924 |
| 29 | s5 0.25 + s5b 0.25 + t10 0.5 |  | 0.22866 | 0.33924 |
| 30 | s5 0.25 + s5b 0.25 + t10 0.25 + t6 0.25 |  | 0.22888 | 0.3393 |
| 31 | s5 0.3 + s5b 0.3 + t10 0.4 |  | 0.22838 | 0.33925 |
