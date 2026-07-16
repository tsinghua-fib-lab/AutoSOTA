# p-IRT Estimation for TinyBenchmarks

## Overview

This directory contains scripts for p-IRT accuracy estimation using **TinyBenchmarks** selected items. TinyBenchmarks provides curated, extremely small subsets of benchmark datasets designed for rapid model evaluation with minimal compute requirements.

## What is TinyBenchmarks?

TinyBenchmarks (from Hugging Face `tinyBenchmarks/*`) are carefully selected subsets of popular benchmarks that maintain strong correlation with full dataset performance while using only a tiny fraction of the data (typically 5-10% of items).

## Supported Datasets

- **tinyAI2_arc** (ARC-Challenge)
- **tinyGSM8K**
- **tinyHellaswag**
- **tinyTruthfulQA**
- **tinyWinogrande**

## Key Characteristics

- **Very small subsets**: Typically 50-100 items
- **Curated selection**: Items chosen for representativeness
- **High correlation**: Designed to maintain r > 0.95 with full dataset
- **Fast evaluation**: Minimal computational cost

## Scripts

### 1. `generate_all_tinybenchmark_ids.py`

**Purpose:** Downloads TinyBenchmarks data and extracts item IDs for all datasets.

**Usage:**
```bash
cd baseline_compare
python generate_all_tinybenchmark_ids.py
```

**Outputs:**
- `tinyarc_item_ids.csv`
- `tinygsm8k_item_ids.csv`
- `tinyhellaswag_item_ids.csv`
- `tinytruthfulqa_item_ids.csv`
- `tinywinogrande_item_ids.csv`

**Requirements:**
- `datasets` package: `pip install datasets`
- Optional: Prompts files in `metabench_data/benchmark-data/` for ID matching

**Note:** Some datasets may use indices as item IDs if direct ID fields are not available.

### 2. `pirt_tinybenchmark.r`

**Purpose:** Computes p-IRT accuracy estimates using TinyBenchmarks selected items.

**Usage:**
```bash
Rscript baseline_compare/pirt_tinybenchmark.r --benchmark=arc
```

**Parameters:**
- `--benchmark`: Dataset name (lowercase: arc, gsm8k, hellaswag, truthfulqa, winogrande)

**Outputs:**
- `baseline_compare/pirt_tiny{benchmark}.csv`

**Columns:**
- `model`: Model name
- `pirt_accuracy`: p-IRT accuracy estimate
- `theta`: Estimated ability parameter
- `n_subset_items`: Number of items in TinyBenchmark
- `n_all_items`: Total number of items in full dataset
- `avg_observed`: Average accuracy on observed items
- `avg_predicted`: Average predicted probability on unobserved items

### 3. `compare_pirt_tinybenchmark.r`

**Purpose:** Compares TinyBenchmarks p-IRT estimates with actual accuracy.

**Usage:**
```bash
Rscript baseline_compare/compare_pirt_tinybenchmark.r --benchmark=arc
```

**Outputs:**
- `baseline_compare/pirt_tiny{benchmark}_vs_actual.csv`
- `baseline_compare/pirt_tiny{benchmark}_vs_actual.png`

**Metrics:**
- Mean Error (ME)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Correlation (r)
- R-squared

### 4. `run_pirt_tinybenchmark.job`

**Purpose:** Grid Engine job script for cluster execution.

**Usage:**
```bash
export BENCHMARK=arc
qsub baseline_compare/run_pirt_tinybenchmark.job
```

### 5. `run_pirt_tinybenchmark_all.sh`

**Purpose:** Batch script to process all TinyBenchmarks datasets.

**Usage:**
```bash
bash baseline_compare/run_pirt_tinybenchmark_all.sh
```

Processes all 5 datasets: arc, gsm8k, hellaswag, truthfulqa, winogrande

## Workflow

### Complete Workflow (All Datasets)

```bash
# Step 1: Generate TinyBenchmarks item IDs
cd /store01/nchawla/pli9/llmbenchmark/baseline_compare
python generate_all_tinybenchmark_ids.py

# Step 2: Run p-IRT estimation for all
bash run_pirt_tinybenchmark_all.sh
```

### Single Dataset Workflow

```bash
# Generate item IDs (if not already done)
python generate_all_tinybenchmark_ids.py

# Run p-IRT for specific dataset
Rscript pirt_tinybenchmark.r --benchmark=arc
Rscript compare_pirt_tinybenchmark.r --benchmark=arc
```

### Cluster Submission

```bash
# Submit single job
export BENCHMARK=arc
qsub baseline_compare/run_pirt_tinybenchmark.job

# Submit all datasets
for benchmark in arc gsm8k hellaswag truthfulqa winogrande; do
  export BENCHMARK=$benchmark
  qsub baseline_compare/run_pirt_tinybenchmark.job
done
```

## Data Files

### Required Input Files

For each dataset, you need:

1. **TinyBenchmarks item IDs:**
   - `baseline_compare/tiny{dataset}_item_ids.csv`

2. **Item parameters:**
   - `{dataset}/irt_item_parameters_combined.csv` (a1, d, g, u format)

3. **Response matrix:**
   - `select_model/gaussian_sampled_{dataset}_response_matrix_test.csv`

### File Format

**TinyBenchmarks Item IDs** (e.g., `tinyarc_item_ids.csv`):
```csv
item_id
ARCCH_123
ARCCH_456
ARCCH_789
...
```

**Output Format** (e.g., `pirt_tinyarc.csv`):
```csv
model,pirt_accuracy,theta,n_subset_items,n_all_items,avg_observed,avg_predicted
model1,0.7234,0.512,80,1200,0.72,0.7245
model2,0.6891,0.123,80,1200,0.69,0.6895
...
```

## Comparison: TinyBenchmarks vs Others

| Aspect | TinyBenchmarks | Metabench | Random 100 | ATLAS |
|--------|---------------|-----------|------------|-------|
| Selection method | Expert curated | Expert curated | Random | Adaptive |
| Subset size | 50-100 | 50-100 | 100 | Variable |
| Design goal | Speed | Efficiency | Baseline | Precision |
| Versions | 1 per dataset | 2 per dataset | 1 per dataset | Per model |
| Expected RMSE | Very low | Very low | Medium | Low |

## Expected Results

Based on TinyBenchmarks design goals:
- **RMSE:** Should match or beat Metabench
- **Correlation:** r > 0.95 (design target)
- **Subset size:** Smallest of all methods (~5-10% of full dataset)
- **Speed:** Fastest evaluation time

## Troubleshooting

### Item IDs file not found
```
ERROR: Selected items file not found
```
**Solution:** Run `python generate_all_tinybenchmark_ids.py` first.

### Cannot download TinyBenchmarks
```
Error loading dataset tinyBenchmarks/...
```
**Solution:** 
- Check internet connection
- Verify `datasets` package is installed
- Check dataset name is correct

### No matching items found
```
Warning: No matching items found for model
```
**Solution:** 
- Check item ID format matches response matrix
- Some datasets may use indices instead of original IDs
- Verify prompts file exists if using prompt matching

### Low correlation with actual
```
Correlation: 0.75 (expected > 0.95)
```
**Possible causes:**
- Item IDs may not match correctly
- IRT parameters may have issues
- Dataset-specific calibration problems

## Validation

To validate TinyBenchmarks p-IRT estimates:

```r
# Read results
tiny <- read.csv("pirt_tinyarc.csv")
metabench <- read.csv("pirt_metabench_arc_secondary.csv")
random100 <- read.csv("pirt_random_100_arc.csv")
actual <- read.csv("../accuracy/arc/actual_accuracy.csv")

# Compare RMSE
rmse_tiny <- sqrt(mean((tiny$pirt_accuracy - actual$actual_accuracy)^2))
rmse_meta <- sqrt(mean((metabench$pirt_accuracy - actual$actual_accuracy)^2))
rmse_rand <- sqrt(mean((random100$pirt_accuracy - actual$actual_accuracy)^2))

cat("RMSE Comparison:\n")
cat("  TinyBenchmarks:", round(rmse_tiny, 4), "\n")
cat("  Metabench:", round(rmse_meta, 4), "\n")
cat("  Random 100:", round(rmse_rand, 4), "\n")

# Compare efficiency (subset size)
cat("\nEfficiency Comparison:\n")
cat("  TinyBenchmarks:", tiny$n_subset_items[1], "items\n")
cat("  Metabench:", metabench$n_subset_items[1], "items\n")
cat("  Random 100:", random100$n_subset_items[1], "items\n")
```

## Item ID Extraction Methods

Different datasets use different methods to extract item IDs:

1. **Direct ID field** (ARC, HellaSwag):
   - Uses `id` or `ind` field from TinyBenchmarks dataset

2. **Prompt matching** (TruthfulQA):
   - Matches questions to prompts file to get item IDs
   - Requires prompts CSV in `metabench_data/benchmark-data/`

3. **Index-based** (GSM8K, Winogrande):
   - Uses sequential indices as item IDs
   - Fallback method when no ID field exists

## Integration with Main Pipeline

The TinyBenchmarks scripts follow the same structure as other p-IRT scripts:
- Use same item parameter conversion (a1, d, g, u) → (a, b, c)
- Use same theta estimation method
- Generate compatible output formats
- Support same visualization and comparison tools

## Performance Tips

For fastest results:
1. Use TinyBenchmarks for initial screening
2. Use Random 100 for baseline comparison
3. Use Metabench for validation
4. Use ATLAS for final precision estimates

## References

- TinyBenchmarks: [Hugging Face tinyBenchmarks](https://huggingface.co/tinyBenchmarks)
- p-IRT method: See `../PIRT_ESTIMATION_README.md`
- Baseline comparisons: See `../PIRT_SCRIPTS_SUMMARY.md`
