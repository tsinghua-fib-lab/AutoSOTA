# Metabench p-IRT Quick Start Guide

## What is Metabench?

Metabench provides curated subsets of benchmark datasets designed for efficient model evaluation. Each dataset has **two versions**:
- **Primary**: Main curated subset
- **Secondary**: Alternative curated subset

## Quick Start (TruthfulQA Example)

```bash
# Navigate to baseline_compare directory
cd /store01/nchawla/pli9/llmbenchmark/baseline_compare

# Step 1: Generate metabench item IDs (one-time setup)
python generate_all_metabench_ids.py

# Step 2: Run p-IRT for TruthfulQA secondary version
Rscript pirt_metabench.r --benchmark=truthfulqa --version=secondary

# Step 3: Compare with actual accuracy
Rscript compare_pirt_metabench.r --benchmark=truthfulqa --version=secondary
```

## Results

After running, you'll find:
- `pirt_metabench_truthfulqa_secondary.csv` - p-IRT accuracy estimates
- `pirt_metabench_vs_actual_truthfulqa_secondary.csv` - Comparison with ground truth
- `pirt_metabench_vs_actual_truthfulqa_secondary.png` - Visualization

## Run All Datasets

```bash
cd /store01/nchawla/pli9/llmbenchmark/baseline_compare

# Generate all metabench item IDs
python generate_all_metabench_ids.py

# Run p-IRT for all datasets and both versions
bash run_pirt_metabench_all.sh
```

This processes:
- arc (primary + secondary)
- gsm8k (primary + secondary)
- hellaswag (primary + secondary)
- truthfulqa (primary + secondary)
- winogrande (primary + secondary)

**Total: 10 configurations**

## Cluster Submission

```bash
# Single dataset
export BENCHMARK=truthfulqa
export VERSION=secondary
qsub /store01/nchawla/pli9/llmbenchmark/baseline_compare/run_pirt_metabench.job

# All datasets (submit 10 jobs)
cd /store01/nchawla/pli9/llmbenchmark/baseline_compare
for benchmark in arc gsm8k hellaswag truthfulqa winogrande; do
  for version in primary secondary; do
    export BENCHMARK=$benchmark
    export VERSION=$version
    qsub run_pirt_metabench.job
  done
done
```

## Viewing Results

```r
# Load results
results <- read.csv("baseline_compare/pirt_metabench_vs_actual_truthfulqa_secondary.csv")

# Check performance
cat("RMSE:", sqrt(mean(results$squared_error)), "\n")
cat("Correlation:", cor(results$pirt_accuracy, results$actual_accuracy), "\n")

# View plot
# Open: baseline_compare/pirt_metabench_vs_actual_truthfulqa_secondary.png
```

## Comparison with Other Methods

```bash
# Compare Metabench vs Random 100 vs ATLAS
cd /store01/nchawla/pli9/llmbenchmark

# Random 100
Rscript baseline_compare/pirt_random_100.r --benchmark=truthfulqa

# Metabench (already done above)
# ...

# ATLAS (if available)
Rscript pirt_accuracy_estimation.r --benchmark=truthfulqa --se_theta_stop=0.1
```

Then compare RMSE values across all three methods.

## Troubleshooting

### Item IDs not found
```
ERROR: Selected items file not found
```
**Solution:** Run `python generate_all_metabench_ids.py` first

### No metabench_data package
```
ModuleNotFoundError: No module named 'datasets'
```
**Solution:** `pip install datasets` or use conda/module

### Response matrix not found
```
ERROR: Response matrix not found
```
**Solution:** Check that test response matrix exists for the benchmark

## Expected Performance

Metabench should outperform Random 100:
- **RMSE:** Lower error
- **Correlation:** Higher r (typically > 0.95)
- **Efficiency:** Similar or better subset sizes

## Files Created

After running for truthfulqa:
```
baseline_compare/
├── metabench_truthfulqa_item_ids_primary.csv
├── metabench_truthfulqa_item_ids_secondary.csv
├── pirt_metabench_truthfulqa_primary.csv
├── pirt_metabench_truthfulqa_secondary.csv
├── pirt_metabench_vs_actual_truthfulqa_primary.csv
├── pirt_metabench_vs_actual_truthfulqa_primary.png
├── pirt_metabench_vs_actual_truthfulqa_secondary.csv
└── pirt_metabench_vs_actual_truthfulqa_secondary.png
```

## See Also

- Full documentation: `METABENCH_PIRT_README.md`
- Main p-IRT guide: `../PIRT_ESTIMATION_README.md`
- All scripts summary: `../PIRT_SCRIPTS_SUMMARY.md`
