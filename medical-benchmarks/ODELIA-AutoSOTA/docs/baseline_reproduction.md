# Released baseline reproduction and evidence

## What is included

third_party/meisenmeister is an unmodified source snapshot at commit
a9b8ded20d59a7f6df8b03f24a6ec28d9d8ede0e. Its architecture, preprocessing,
official inference, documentation, tests and license are retained. The portable
baseline command additionally reproduces the identity adaptation / seven-probability
arithmetic average used for the final controlled comparison.

No champion training run is claimed. The exact MAMA-MIA ResEncUNetL 2000-epoch
fold_all initialization was not in the audited public segmentation weights
(those were single-channel PlainConvUNet five-fold checkpoints). Strict end-to-end
champion retraining therefore cannot be reconstructed from those weights.

## Historical execution

The public Fold 0 test run completed on 2026-08-03 using all seven best checkpoints,
TTA and torch.compile defaults, on A100 80GB. Three-channel arrays were converted by
the released int16 NumPy -> TorchIO -> NIfTI route. Seven disjoint batches were merged
into 150 unique examinations / 300 breast ROIs.

Upstream BreastDivider plans requested unavailable SimpleITKIOWithReorient. A runtime
copy replaced that string with NibabelIOWithReorient. This integration workaround
did not change source data, model tensors or metric definitions. The segmenter SHA-256 was
18441fe5536ce9209d216f8a9ba6e6331c69f54a0acb3fa2fca6428214b92d18.

The final locked micro-ROC baseline is Challenge score 0.876448148148, Micro-AUC
0.951011111111, specificity@90sens 0.841666666667, sensitivity@90spec 0.836666666667.
The historical full-cohort macro OVR AUC was 0.931091299686, balanced accuracy
0.675724876321 and macro F1 0.710471167918. A separate older report's operating-point
values 0.7710/0.7357 were computed with a different macro/binary summary; they are
not the flattened micro-ROC components used by the fixed final comparison.

## Historical input integrity

results/historical_metrics.json retains the SHA-256 values for the seven training
cache manifests, released test predictions, test labels, descriptor archive and
original protected metrics module. provenance.json records original source hashes
and Hugging Face revisions. Compare these against local artifacts if attempting
the historical route. A hash proves artifact identity, not unbiased sampling or
independence from upstream training. Exact subject-membership files and clinical
artifacts are external to this source release.

## Limits to fairness

The released converter pooled 937 examinations and created leave-one-source-out
folds. It did not necessarily preserve public Fold 0 test membership as unseen
for all ensemble members. Each public case is likely present in the training
set of several released models; full per-checkpoint membership is not established.
Consequently these scores should be used for same-released-weights relative
comparison, not as independent test performance or unseen-center generalization.

The local head training/validation split excludes public test cases, but that
does not retroactively remove potential encoder pretraining overlap. Public-test
aggregate feedback also selected the final head design. These are distinct limitations.
