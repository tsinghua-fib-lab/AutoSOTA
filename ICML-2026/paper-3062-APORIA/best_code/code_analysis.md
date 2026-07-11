# Code Analysis — Paper 3062 SOTA Preparation Repair

## Preparation Failure

The original `eval_command` from the manifest (`python3 /repo/run_eval.py`) failed because:
- The default Python environment (`/opt/conda/bin/python3`, Python 3.10.13) lacks `pandas`.
- The required packages (`pandas`, `numpy`, `aporia`, `scipy`, `scikit-learn`) are installed in the `py311` conda environment at `/opt/conda/envs/py311`.

## Corrected Evaluation Command

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate py311 && cd /repo && python3 /repo/run_eval.py
```

## Baseline Verification

The corrected command reproduces the manifest baseline exactly:

| Model | Accuracy | F1 |
|-------|----------|-----|
| Mistral-7B | 86.8% (6.6) | 92.1% (4.4) |
| Ensemble Avg | 87.2% (7.4) | 90.7% (7.1) |

All 10 model results match the paper Table 2 exactly.

## Key Code Architecture

### Entry Point: `/repo/run_eval.py`
- Loads SOCRATES-300K parquet dataset
- Runs structural analysis (cached)
- Runs `run_full_label_propagation_study()` with FisherProjection + WassersteinLabelPropagator
- Aggregates and prints per-model Accuracy/F1 table

### Core Modules (in `/repo/aporia/`, installed editable)

| File | Key Classes/Functions | Role |
|------|----------------------|------|
| `projections.py` | `FisherProjection`, `fisher_direction()` | Supervised 1D Fisher projection |
| `label_propagation.py` | `WassersteinLabelPropagator`, `CentroidPropagator`, `SKLearnPropagator` | Propagator classifiers |
| `data.py` | `load_dataframe()`, `subsample_training_set()`, `generate_fixed_test_sets()` | Data loading and splitting |
| `evaluation.py` | `LabelPropagationEvaluator` | Metric computation |
| `config.py` | `Config`, `DatasetConfig` | TOML config parsing |

### Optimization Levers

1. **Lambda regularization** (`config/socrates.toml` → `best_lambda=1.2`): Direct parameter change
2. **Projection:** `FisherProjection` params (lambda, normalise, normalise_by_trace); alternative projections (CentroidFeatures, Identity, PCA)
3. **Propagator:** Wasserstein (default), Centroid(cosine/euclidean), SKLearnPropagator(SVM/LR/kNN)
4. **Embedding preprocessing:** L2 normalization before Fisher, embedding column change
5. **Distance metric:** Wasserstein uses euclidean cdist → cosine is natural for SBERT embeddings

### Safe Optimization Targets
- Modifying `aporia/label_propagation.py` (propagator code, new propagator classes)
- Modifying `aporia/projections.py` (Fisher preprocessing, new projection classes)
- Modifying `aporia/data.py` (RNG seeding fix)
- Modifying `run_eval.py` (parameter sweeps, propagator selection)
- NOT: dataset, labels, test splits, metric definitions

### Cache Strategy
- Cache at `cache/socrates/LP-fisher/` uses per-pair parquet files keyed by (model, prompt, projector, propagator)
- Use `overwrite_cache=True` when changing code that affects results
- Full 10-model eval takes ~13 seconds with cache warm
