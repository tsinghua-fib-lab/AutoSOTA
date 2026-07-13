# Paper 5009 SOTA Preparation Repair — Code Analysis

## Original Preparation Failure

**Root cause**: Shell quoting bug in `eval.sh` line 54:
```bash
python3 compute/build_features_helper.py "$PP_PATH" "$CPD_PATH" "$FEATURES_CSV" "$WINDOWS"
```
The variable `$WINDOWS="1 5 10 15 20"` was double-quoted, causing it to be passed as a single string argument `"1 5 10 15 20"` to `build_features_helper.py`. The Python script expects each window size as a separate integer argument:
```python
windows = [int(w) for w in sys.argv[4:]]  # fails on "1 5 10 15 20"
```

**Fix**: Removed double-quotes around `$WINDOWS` to allow shell word splitting:
```bash
python3 compute/build_features_helper.py "$PP_PATH" "$CPD_PATH" "$FEATURES_CSV" $WINDOWS
```

## Corrected In-Container Evaluation Command

```bash
cd /repo && bash eval.sh
```

No host-side commands or Docker mounts needed beyond the container itself.

## Baseline Metrics (Verified)

| Metric | Reproduced | Manifest | Status |
|--------|-----------|----------|--------|
| CPD_F1 | 0.77647 | 0.776 | Match |
| CPD_AUROC | 0.84027 | 0.840 | Match |
| Best_WPP_F1 | 0.73592 | 0.736 | Match |
| Best_WPP_AUROC | 0.80497 | 0.805 | Match |
| PP_F1 | 0.6490 | 0.649 | Match |

All metrics are within normal numerical noise of the reproduction manifest.

## Optimization Baseline Commit

The commit `83697b9` ("optimization baseline") was created from the original reproduction commit `1a6c055` with these changes:
- Added `eval.sh` (75 lines) — the evaluation script
- Added `compute/build_features_helper.py` (39 lines) — merges PP and CPD CSVs
- Modified `compute/compute_token_stats.py` (minor fix)
- Modified `config/models.yaml` (simplified to only llama-7b config)

## Reusable Resources

- Model: `/models/Llama-2-7b-chat-hf/` (from ModelScope, already downloaded)
- Token stats: `stats/llama-7b_benign_mix_ppgap1_800_token_stats.csv` (pre-computed, reused on re-run)
- Dataset: 1514 rows (724 adversarial, 790 benign at alpha=1 matched-PP)
- Output CSVs: `results/changepoints/` and `results/llama-7b_benign_mix_ppgap1_800_k_0/`

## Safe Optimization Targets

The pipeline has clear separation of concerns:

### 1. `compute/compute_token_stats.py` — Feature extraction (entropy from logits)
- Safe to modify: entropy formula, temperature scaling (I-01), top-K entropy (I-05), numerical precision (I-08), intermediate-layer entropy (I-02)
- Do NOT change: tokenizer behavior, model forward pass structure, dataset loading

### 2. `CPD/cpd_online.py` — CUSUM detector implementation
- Safe to modify: k adaptation (I-03, I-07), detector algorithm (I-04 BOCD), W_minus tracking (I-07)
- Do NOT change: the OnlineCUSUMConfig interface without backward compatibility

### 3. `CPD/run_cpd_batch.py` — Batch CPD evaluation
- Safe to modify: feature extraction from CPD state (I-09 Kendall tau), additional output columns

### 4. `compute/build_features_helper.py` — Feature CSV assembly
- Safe to modify: add feature columns (I-09), merge logic

### 5. `compute/pick_best_threshold.py` — Threshold selection and CV evaluation
- Safe to modify: threshold criterion (I-12 Youden J), multi-feature classifier (I-06), per-family calibration (I-10), cross-fold ensembling (I-11)
- Do NOT change: CV fold construction, stratification, metric computation

### 6. `eval.sh` — Evaluation orchestration
- Safe to modify: pass additional flags to sub-scripts
- Do NOT change: data loading, output paths, evaluation protocol

## Red Lines

- Never modify evaluation data, labels, dataset splits, or benchmark outputs
- Never hard-code predictions or metrics
- Always use `/tools/record_score.sh` for score recording
- Always commit successful implementations in git
