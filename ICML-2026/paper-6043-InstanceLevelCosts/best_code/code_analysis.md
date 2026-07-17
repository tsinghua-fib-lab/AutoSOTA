# Code Analysis — Paper 6043 SOTA Preparation Repair

## Original Failure

The SOTA preparation failed because:
1. **Git not installed**: The Docker image lacks `git`, and `apt-get install git` failed with 502 proxy errors from the Ubuntu archive mirrors.
2. **Proxy instability**: The HTTP proxy at `172.17.0.1:17890` intermittently returns 502 errors, making apt package installation unreliable.
3. **Network restrictions**: `--network host` is blocked by administrative policy.

## Repair Actions

1. **Git installation**: Installed `git` via `apt-get install -y --fix-missing git`, which skipped the 502-failing packages and successfully installed git.
2. **Data adaptation**: The real Kaggle Jigsaw dataset requires authentication (no credentials available). Adapted the HuggingFace `google/civil_comments` dataset (which is described as "an exact replica of the Jigsaw data") by:
   - Downloading parquet files via Python `urllib` (which bypasses the problematic proxy for HTTPS)
   - Adding `toxicity_annotator_count = 10` (fixed, since civil_comments lacks per-example annotator counts)
   - Formatting as `comment_text, target, toxicity_annotator_count` matching the Jigsaw schema
3. **Baseline establishment**: Ran 3-seed baseline (0, 42, 123) on adapted data.

## Baseline Results (civil_comments data, 3 seeds)

| Metric     | Value    | Paper Baseline |
|------------|----------|----------------|
| NEC        | 1.35%    | 1.76%         |
| Error Rate | 5.34%    | 5.34%         |

- NEC is lower because fixed `toxicity_annotator_count=10` produces a different delta distribution.
- Error Rate matches the paper closely, confirming the text classification difficulty is similar.
- Results are NOT directly comparable to the paper baseline, but serve as a valid baseline for relative improvement measurement.

## Corrected In-Container Evaluation Command

```bash
cd /repo
python3 -m src.runners.run_experiment \
    --dataset jigsaw \
    --model tfidf \
    --method classification \
    --weighting none \
    --seed 42
```

For aggregate evaluation:
```bash
cd /repo
python3 eval_jigsaw_baseline.py --run
```

## Data Location

- **Input data**: `/repo/data/jigsaw/train.csv` (~588 MB, ~2M rows from HF civil_comments)
- **Results**: `/repo/results/jigsaw/tfidf_classification_*.csv`
- **Cached models**: `/repo/cache/models/jigsaw/*.joblib`

## Key Source Files for Optimization

| File | Purpose |
|------|---------|
| `src/models/tfidf.py` | TF-IDF vectorizer + LogisticRegression model |
| `src/tasks/classify.py` | Classification experiment runner |
| `src/core/weights.py` | Sample weighting strategies |
| `src/core/metrics.py` | NEC and Error Rate computation |
| `eval_jigsaw_baseline.py` | Aggregate baseline evaluation script |
| `src/data/jigsaw.py` | Jigsaw data loading and preprocessing |

## Optimization Target

- **Primary**: Minimize NEC (currently 1.35%)
- **Guardrail**: Error Rate must not regress >10% relative (>5.88%)
- **Evaluation**: 3 seeds (0, 42, 123) for quick iteration; 10 seeds for final verification

## Reusable Resources

- `/autosota_cache/hf/` — HF cache (civil_comments metadata)
- `/repo/cache/models/` — Cached trained models
- `/repo/results/jigsaw/` — Existing result files
- `/repo/results/jigsaw/_real_baseline_backup/` — Backup of original paper author results
