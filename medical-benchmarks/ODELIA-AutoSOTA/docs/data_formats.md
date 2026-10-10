# Feature and label formats

All NPZ files are loaded with allow_pickle=False. Store IDs as Unicode strings,
labels as integer 0/1/2, and features/logits as float32. Keep the seven fold axes in order 0..6.
Feature columns: stage3 mean/max/std, stage4 mean/max/std, stage5 mean/max/std.
Population std (unbiased=False) is required.

## Training archive

For prefix train and val:
- PREFIX_sample_ids: (N,), unique breast-ROI identifiers
- PREFIX_case_ids: (N,), examination identifiers; both breasts share one case ID
- PREFIX_labels: (N,), integers 0/1/2
- PREFIX_features: (N,7,2688)
- PREFIX_released_logits: (N,7,3)

Train and validation sample IDs AND case IDs must be disjoint. Validation must contain
all three classes for macro AUC. CLI pack derives this archive from the legacy cache.

## Seven-fold legacy cache

For fold 0..6, fold_F/manifest.json contains fold=F and records with
sample_id, case_id, label, partition (train or val). Record order and labels must
agree across folds. fold_F/samples/SAMPLE_ID.npz stores released_logits (3,) and
stage3_mean/max/std (256,), stage4_mean/max/std (320,), stage5_mean/max/std (320,).
Do not include test cases in training manifests. Keep this directory outside Git.

## Modern inference archive

sample_ids (N,), features (N,7,2688), released_logits (N,7,3).
No labels key is allowed. Use a separate evaluation file with sample_ids and labels (N,).
Evaluation aligns by IDs and rejects missing/extra/duplicate IDs.

## Historical inference route

released_predictions.json has cases[CASE].rois[ROI].per_fold[F].probabilities (3,).
block_features.npz has sample_ids (N,) and block_features (N,7,2688).
The toolkit aligns them by sample ID and uses log-clipped probabilities as base logits.
No test labels are passed to infer or baseline. Labels are supplied only to evaluate.

For compatibility with the original identity baseline, predictions.npz contains
sample_ids, per_fold_probabilities (N,7,3), ensemble_probabilities (N,3).
Final-head runs save one state dictionary per fold; only toolkit-generated checkpoints
with matching method and feature/fold contracts are accepted, via weights_only=True.
