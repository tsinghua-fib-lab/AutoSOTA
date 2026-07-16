# p-IRT Estimation for Metabench

## Overview

This directory contains scripts for p-IRT accuracy estimation using **Metabench** selected items. Metabench provides curated subsets of benchmark datasets for efficient model evaluation.

## Metabench Structure

Metabench provides **two versions** for each dataset:
- **Primary**: Main curated subset
- **Secondary**: Alternative curated subset

Both versions are designed to be representative of the full dataset while requiring fewer items.

## Supported Datasets

- ARC
- GSM8K
- HellaSwag
- TruthfulQA
- Winogrande

## Scripts

### 1. `generate_all_metabench_ids.py`

**Purpose:** Downloads metabench data and generates item ID files for all datasets.

**Usage:**
```bash
cd baseline_compare
python generate_all_metabench_ids.py
```

**Outputs:**
- `metabench_{dataset}_item_ids_primary.csv`
- `metabench_{dataset}_item_ids_secondary.csv`

**Note:** Requires `datasets` package: `pip install datasets`

### 2. `pirt_metabench.r`

**Purpose:** Computes p-IRT accuracy estimates using metabench selected items.

**Usage:**
```bash
Rscript baseline_compare/pirt_metabench.r \
  --benchmark=truthfulqa \
  --version=secondary
```

**Parameters:**
- `--benchmark`: Dataset name (lowercase)
- `--version`: Either `primary` or `secondary`

**Outputs:**
- `baseline_compare/pirt_metabench_{benchmark}_{version}.csv`

**Columns:**
- `model`: Model name
- `pirt_accuracy`: p-IRT accuracy estimate
- `theta`: Estimated ability parameter
- `n_subset_items`: Number of items in metabench subset
- `n_all_items`: Total number of items in full dataset
- `avg_observed`: Average accuracy on observed items
- `avg_predicted`: Average predicted probability on unobserved items

### 3. `compare_pirt_metabench.r`

**Purpose:** Compares metabench p-IRT estimates with actual accuracy.

**Usage:**
```bash
Rscript baseline_compare/compare_pirt_metabench.r \
  --benchmark=truthfulqa \
  --version=secondary
```

**Outputs:**
- `baseline_compare/pirt_metabench_vs_actual_{benchmark}_{version}.csv`
- `baseline_compare/pirt_metabench_vs_actual_{benchmark}_{version}.png`

**Metrics:**
- Mean Error (ME)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Correlation (r)
- R-squared

### 4. `run_pirt_metabench.job`

**Purpose:** Grid Engine job script for cluster execution.

**Usage:**
```bash
export BENCHMARK=truthfulqa
export VERSION=secondary
qsub baseline_compare/run_pirt_metabench.job
```

### 5. `run_pirt_metabench_all.sh`

**Purpose:** Batch script to process all metabench datasets and versions.

**Usage:**
```bash
bash baseline_compare/run_pirt_metabench_all.sh
```

Processes all combinations:
- arc (primary + secondary)
- gsm8k (primary + secondary)
- hellaswag (primary + secondary)
- truthfulqa (primary + secondary)
- winogrande (primary + secondary)

**Total:** 10 configurations (5 datasets × 2 versions)

## Workflow

### Complete Workflow (All Datasets)

```bash
# Step 1: Generate metabench item IDs
cd /store01/nchawla/pli9/llmbenchmark/baseline_compare
python generate_all_metabench_ids.py

# Step 2: Run p-IRT estimation for all
bash run_pirt_metabench_all.sh
```

### Single Dataset Workflow

```bash
# Generate item IDs (if not already done)
python generate_all_metabench_ids.py

# Run p-IRT for specific dataset and version
Rscript pirt_metabench.r --benchmark=truthfulqa --version=secondary
Rscript compare_pirt_metabench.r --benchmark=truthfulqa --version=secondary
```

### Cluster Submission

```bash
# Submit single job
export BENCHMARK=truthfulqa
export VERSION=primary
qsub baseline_compare/run_pirt_metabench.job

# Submit multiple jobs
for benchmark in arc gsm8k hellaswag truthfulqa winogrande; do
  for version in primary secondary; do
    export BENCHMARK=$benchmark
    export VERSION=$version
    qsub baseline_compare/run_pirt_metabench.job
  done
done
```

## Data Files

### Required Input Files

For each dataset, you need:

1. **Metabench item IDs:**
   - `baseline_compare/metabench_{dataset}_item_ids_primary.csv`
   - `baseline_compare/metabench_{dataset}_item_ids_secondary.csv`

2. **Item parameters:**
   - `{dataset}/irt_item_parameters_combined.csv` (a1, d, g, u format)

3. **Response matrix:**
   - `select_model/gaussian_sampled_{dataset}_response_matrix_test.csv`

### File Format

**Metabench Item IDs** (e.g., `metabench_truthfulqa_item_ids_secondary.csv`):
```csv
item_id
725
421
202
...
```

**Output Format** (e.g., `pirt_metabench_truthfulqa_secondary.csv`):
```csv
model,pirt_accuracy,theta,n_subset_items,n_all_items,avg_observed,avg_predicted
model1,0.7123,0.423,100,817,0.71,0.7145
model2,0.6892,0.189,100,817,0.69,0.6901
...
```

## Comparison: Metabench vs Random 100

| Aspect | Metabench | Random 100 |
|--------|-----------|------------|
| Selection method | Curated by experts | Random sampling |
| Versions | 2 per dataset | 1 per dataset |
| Item quality | High-quality items | Mixed quality |
| Representativeness | Optimized | Statistical average |
| Subset size | Varies (~50-100) | Fixed at 100 |
| Expected performance | Better RMSE | Baseline |

## Expected Results

Based on metabench design:
- **RMSE:** Should be lower than random 100
- **Correlation:** r > 0.95 for well-calibrated datasets
- **Subset size:** Typically 50-100 items (varies by dataset)
- **Reduction ratio:** 0.05-0.15 (5-15% of full dataset)

## Troubleshooting

### Item IDs file not found
```
ERROR: Selected items file not found
```
**Solution:** Run `python generate_all_metabench_ids.py` first.

### Dataset not available in Metabench
```
Error loading dataset
```
**Solution:** Check that the dataset name is correct and available in HCAI/metabench.

### No matching items found
```
Warning: No matching items found for model
```
**Solution:** Check that item IDs in metabench file match the response matrix column names.

## Validation

To validate metabench p-IRT estimates:

```r
# Read results
metabench_primary <- read.csv("pirt_metabench_truthfulqa_primary.csv")
metabench_secondary <- read.csv("pirt_metabench_truthfulqa_secondary.csv")
random100 <- read.csv("pirt_random_100_truthfulqa.csv")
actual <- read.csv("../accuracy/truthfulqa/actual_accuracy.csv")

# Compare RMSE
rmse_primary <- sqrt(mean((metabench_primary$pirt_accuracy - actual$actual_accuracy)^2))
rmse_secondary <- sqrt(mean((metabench_secondary$pirt_accuracy - actual$actual_accuracy)^2))
rmse_random <- sqrt(mean((random100$pirt_accuracy - actual$actual_accuracy)^2))

cat("RMSE Comparison:\n")
cat("  Metabench Primary:", round(rmse_primary, 4), "\n")
cat("  Metabench Secondary:", round(rmse_secondary, 4), "\n")
cat("  Random 100:", round(rmse_random, 4), "\n")
```

## Integration with Main Pipeline

The metabench scripts follow the same structure as other p-IRT scripts:
- Use same item parameter conversion (a1, d, g, u) → (a, b, c)
- Use same theta estimation method
- Generate compatible output formats
- Support same visualization and comparison tools

## References

- Metabench paper: [HCAI/metabench](https://huggingface.co/datasets/HCAI/metabench)
- p-IRT method: See `../PIRT_ESTIMATION_README.md`
- Baseline comparison: See `../PIRT_SCRIPTS_SUMMARY.md`
