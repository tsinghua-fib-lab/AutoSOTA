# Experiment Scripts

CLI entry points for running ProEval experiments. All scripts should be run from the **project root** as Python modules.

> [!IMPORTANT]
> Labels use **1 = failure, 0 = correct** throughout. Estimates represent error rates, and the results table reports both accuracy and failure rate.

## Framework Overview

ProEval experiments are organized along two axes:

### GP Feature Setups

| Feature Setup | Abbreviation | Kernel | Encoder Required | Description |
|---------------|:---:|:---:|:---:|-------------|
| **Score Features** | **SF** | Linear | No | Other models' predictions (GMM-selected) |
| **Raw Prompt Features** | **RPF** | Matérn 2.5 | No | Raw text embeddings, neutral 0.5 prior |
| **Tuned Prompt Features** | **TPF** | Matérn 2.5 | Yes | Neural encoder embeddings, trained on GMM-selected models |

### Problem Settings

| Setting | Description |
|---------|-------------|
| **`new_pair`** | Exclude only the target model–benchmark pair from training |
| **`new_benchmark`** | Exclude the entire target benchmark from training |
| **`new_model`** | Exclude the target model's predictions from all benchmarks |

### Method Table (14 methods per run)

| Category | Method | Kernel | Description |
|----------|--------|--------|-------------|
| **Baselines** | `Random` | — | Random sampling, simple mean |
| | `RF+IS` | — | Random Forest + Importance Sampling |
| | `LR+IS` | — | Logistic Regression + IS |
| | `RF+LURE` | — | Random Forest + LURE |
| | `LR+LURE` | — | Logistic Regression + LURE |
| **Random+BQ** | `BQ-SF Rand` | Linear | Random selection + SF BQ posterior |
| | `BQ-RPF Rand` | Matérn | Random selection + RPF BQ posterior |
| | `BQ-TPF Rand` | Matérn | Random selection + TPF BQ posterior |
| **Active+BQ** | `BQ-SF` | Linear | Active BQ with score features |
| | `BQ-RPF` | Matérn | Active BQ with raw embeddings |
| | `BQ-TPF` | Matérn | Active BQ with encoder embeddings |
| | `BQ-SF Rounded` | Linear | BQ-SF with rounded posterior |
| | `BQ-RPF Rounded` | Matérn | BQ-RPF with rounded posterior |
| | `BQ-TPF Rounded` | Matérn | BQ-TPF with rounded posterior |

---

## Scripts

### 1. Performance Estimation (Unified)

Runs **all 14 methods** in a single invocation: 5 baselines + 9 BQ variants (3 feature setups × 3 selection strategies).

```bash
python -m experiment.exp_performance_estimation \
    --dataset svamp \
    --target-model gemini25_flash \
    --setting new_pair \
    --encoder-path data/checkpoints/encoder_svamp_new_pair.pth \
    --budget 50 \
    --n-runs 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `svamp` | Dataset name (`svamp`, `gsm8k`, `strategyqa`, etc.) |
| `--target-model` | `gemini25_flash` | Model to evaluate |
| `--setting` | `new_pair` | `new_pair`, `new_benchmark`, or `new_model` |
| `--data-selection` | `gmm` | `all` models or `gmm` filtered sets for SF |
| `--budget` | `50` | Number of samples to acquire |
| `--n-runs` | `1` | Independent runs |
| `--noise-variance` | `0.3` | GP noise |
| `--encoder-path` | `None` | Path to trained encoder (for TPF methods; skipped if absent) |
| `--seed` | `42` | Random seed |

---

### 2. Failure Discovery

Online GP experiment for failure discovery. Supports seed-pool sampling (pre-computed labels, no API key) and LLM-based generation across any dataset.

```bash
# Seed-pool methods on GSM8K (fast, no API key)
python -m experiment.exp_failure_discovery \
    --dataset gsm8k --model gemma3_27b --iterations 50

# All 6 methods on StrategyQA
python -m experiment.exp_failure_discovery \
    --dataset strategyqa --model gemini25_flash --runall --iterations 30

# Failure mode: stop after 20 failures
python -m experiment.exp_failure_discovery \
    --dataset gsm8k --model gemma3_27b --failure 20

# Multi-run for variance estimation
python -m experiment.exp_failure_discovery \
    --dataset gsm8k --model gemma3_4b --runs 3 --iterations 50
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `gsm8k` | Dataset name (`gsm8k`, `strategyqa`, `svamp`, etc.) |
| `--model` | `gemma3_27b` | Target model (CSV column name) |
| `--iterations` | `100` | Iterations per method |
| `--runs` | `1` | Independent runs for variance |
| `--runall` | off | Run all 6 methods |
| `--methods` | `SS Rand` | Specific methods to run |
| `--failure` | `None` | Failure mode: stop at N failures |
| `--ss-beta` | `1.96` | SS acquisition β |
| `--ss-threshold` | `0.0` | SS acquisition threshold λ |
| `--top-k` | `5` | Hard anchors per generation step |
| `--n-topics` | `11` | BERTopic clusters |

#### Methods

| Method | Type | Description |
|--------|------|-------------|
| **SS** | Sampling | Superlevel Set acquisition from seed pool |
| **Rand** | Sampling | Random selection from seed pool (baseline) |
| **SS-Gen** | Generation | SS anchors + LLM generation (no topic) |
| **TSS** | Generation | UCB topic + SS anchors + generation |
| **Rand-T-Gen** | Generation | Random topic + generation (no SS) |
| **Rand-Gen** | Generation | Pure random generation (baseline) |

> [!NOTE]
> Sampling methods (SS, Rand) use pre-computed labels from the CSV — no API key required. Generation methods (SS-Gen, TSS, Rand-T-Gen, Rand-Gen) need `OPENROUTER_API_KEY`.

---

### 3. Train Neural Encoder

Train a cross-benchmark neural encoder for TPF (Tuned Prompt Features).

```bash
python -m experiment.exp_train_encoder \
    --train-benchmarks gsm8k strategyqa \
    --target-benchmark svamp \
    --target-model gemini25_flash \
    --hidden-dim 16 \
    --kernel-type matern \
    --init-lengthscale 1.0 \
    --epochs 1000 \
    --include-models model_a model_b model_c
```

| Flag | Default | Description |
|------|---------|-------------|
| `--train-benchmarks` | _required_ | Benchmarks for training (space-separated) |
| `--target-benchmark` | `svamp` | Target benchmark for evaluation |
| `--target-model` | `gemini25_flash` | Model to evaluate |
| `--hidden-dim` | `16` | Encoder hidden dimension |
| `--kernel-type` | `matern` | `linear`, `matern`, or `rbf` |
| `--init-lengthscale` | `1.0` | Initial lengthscale (matern/rbf) |
| `--epochs` | `200` | Training epochs |
| `--setting` | `new_benchmark` | `new_benchmark`, `new_pair`, or `new_model` |
| `--include-models` | `None` | Train only on these models' labels (GMM-selected) |
| `--checkpoint-path` | auto | Exact save path for encoder |

---

### 4. Aggregate Results

Combine `summary_*.csv` files from multiple experiments into one master table.

```bash
python -m experiment.aggregate_results --results-dir results/
```

---

### 5. Unified Shell Script

End-to-end pipeline: GMM model selection → encoder training → unified experiment → aggregation.

> [!NOTE]
> The `script/` directory and its `*.sh` wrappers are **not included** in this
> repository. Run the `python -m experiment.*` commands above directly (section
> 6 covers the train + sample pipeline). The variables below document the
> intended wrapper configuration if you recreate it.

```bash
# Run all datasets with defaults
bash script/run_performance_estimation_all.sh

# Override configuration via environment
N_RUNS=10 BUDGET=100 bash script/run_performance_estimation_all.sh
```

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `N_RUNS` | `5` | Runs per experiment |
| `BUDGET` | `50` | Sampling budget |
| `TARGET_MODEL` | `gemini25_flash` | Target model |
| `DATASETS` | `svamp gsm8k strategyqa` | Datasets to run |
| `SETTINGS` | `new_pair new_benchmark` | Problem settings |

---

### 6. Unified Train + Sample (All Groups)

End-to-end pipeline: trains an encoder and runs BQ sampling for each dataset, grouped by domain.

```bash
# Run all datasets with new_pair setting
python -m experiment.exp_run_all_groups \
    --setting new_pair \
    --target-model gemini25_flash \
    --n-runs 10 \
    --epochs 500

# Run specific datasets only
python -m experiment.exp_run_all_groups \
    --datasets svamp gsm8k \
    --setting new_benchmark \
    --n-runs 10

# Reuse previously trained encoders
python -m experiment.exp_run_all_groups --skip-training --n-runs 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `--groups` | all | Domain groups to run (`safety`, `reasoning`, `math`) |
| `--datasets` | all | Specific datasets (overrides `--groups`) |
| `--target-model` | `gemini25_flash` | Model to evaluate |
| `--setting` | `new_benchmark` | `new_benchmark` or `new_pair` |
| `--n-runs` | `10` | Independent runs |
| `--epochs` | `1000` | Encoder training epochs |
| `--hidden-dim` | `8` | Encoder hidden dimension |
| `--kernel-type` | `linear` | `linear`, `matern`, or `rbf` |
| `--train-mode` | `group` | `group` (same-group benchmarks) or `all` (all benchmarks) |
| `--skip-training` | off | Reuse existing encoder checkpoints |

> [!NOTE]
> A `run_all_case_2.sh` wrapper for this command is referenced in places but is
> not included in the repository — invoke `python -m experiment.exp_run_all_groups`
> directly.

---

## Typical Workflow

```bash
# 1. Unified experiment (all 14 methods in one run)
python -m experiment.exp_performance_estimation \
    --dataset svamp --setting new_pair \
    --encoder-path data/checkpoints/encoder_svamp_new_pair.pth \
    --n-runs 5

# 2. Or run the full train + sample pipeline (GMM → encoder training → sampling)
python -m experiment.exp_run_all_groups --setting new_pair --n-runs 5

# 3. Failure discovery
python -m experiment.exp_failure_discovery \
    --dataset gsm8k --model gemma3_27b --runall --iterations 50
```

## Data Requirements

All scripts expect data files in `data/`:
- `<dataset>_predictions.csv` — model predictions (`label_<model>` columns: 0=correct, 1=failure)
- `<dataset>_embeddings_text_embedding_3_large.npy` — question embeddings
