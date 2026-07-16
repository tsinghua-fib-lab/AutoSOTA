# Baseline Comparison Methods Summary

## Overview

This directory contains **four different baseline methods** for p-IRT accuracy estimation, each with different trade-offs between accuracy, efficiency, and computational cost.

## Comparison Table

| Method | Subset Size | Selection | Versions | Datasets | Expected RMSE | Speed | Use Case |
|--------|------------|-----------|----------|----------|---------------|-------|----------|
| **ATLAS** | Variable (adaptive) | Adaptive per model | 1 | 5 | Lowest | Slowest | Final precision |
| **Random 100** | 100 (fixed) | Random | 1 | 5 | Baseline | Fast | Simple baseline |
| **Metabench** | 50-100 | Expert curated | 2 (primary/secondary) | 5 | Very low | Fast | Validation |
| **TinyBenchmarks** | 50-100 | Expert curated | 1 | 5 | Very low | Fastest | Initial screening |

## Detailed Comparison

### ATLAS (Adaptive)

**Files:** `pirt_accuracy_estimation.r`, `compare_pirt_actual.r`

**Characteristics:**
- Per-model adaptive item selection
- Variable subset sizes (typically 20-100 items)
- Different items for each model
- Requires running ATLAS algorithm first

**Advantages:**
- Highest precision (lowest RMSE)
- Tailored to each model's ability level
- Efficient for models with varying abilities

**Disadvantages:**
- Requires ATLAS pre-processing
- Different subsets make cross-model comparison harder
- Slowest to compute

**Best for:**
- Final accuracy estimates
- When precision is critical
- Research publications

### Random 100 (Baseline)

**Files:** `pirt_random_100.r`, `compare_pirt_random_100.r`

**Characteristics:**
- Fixed random 100 items
- Same subset for all models
- Simple random sampling

**Advantages:**
- Simple to implement
- Fair comparison (same items for all models)
- No expert curation needed
- Reproducible

**Disadvantages:**
- Higher RMSE than curated methods
- May include low-quality items
- Not optimized for representativeness

**Best for:**
- Baseline comparisons
- Quick estimates
- Fairness across models

### Metabench (Expert Curated)

**Files:** `pirt_metabench.r`, `compare_pirt_metabench.r`, `generate_all_metabench_ids.py`

**Characteristics:**
- Two versions per dataset (primary/secondary)
- Expert-curated item selection
- 50-100 items per version
- From HCAI/metabench

**Advantages:**
- Very low RMSE (comparable to ATLAS)
- Two versions for validation
- Scientifically validated subsets
- Same items for all models

**Disadvantages:**
- Requires downloading from Hugging Face
- Fixed to metabench selections
- May not cover edge cases

**Best for:**
- Validation studies
- Comparing with published results
- Cross-version reliability checks

### TinyBenchmarks (Ultra-Fast)

**Files:** `pirt_tinybenchmark.r`, `compare_pirt_tinybenchmark.r`, `generate_all_tinybenchmark_ids.py`

**Characteristics:**
- Smallest subsets (50-100 items)
- Expert-curated for speed
- From tinyBenchmarks/*
- Designed for r > 0.95 with full dataset

**Advantages:**
- Fastest evaluation
- Minimal computational cost
- Very low RMSE despite small size
- Designed for efficiency

**Disadvantages:**
- Smallest sample size
- Single version only
- Requires downloading from Hugging Face

**Best for:**
- Initial model screening
- Resource-constrained environments
- Rapid prototyping
- Large-scale model evaluation

## Performance Comparison

### Expected RMSE (Lower is Better)

```
ATLAS:           0.015 - 0.025
TinyBenchmarks:  0.020 - 0.035
Metabench:       0.020 - 0.035
Random 100:      0.030 - 0.050
```

### Expected Correlation (Higher is Better)

```
ATLAS:           0.97 - 0.99
TinyBenchmarks:  0.95 - 0.98
Metabench:       0.95 - 0.98
Random 100:      0.90 - 0.95
```

### Computational Cost (Relative)

```
ATLAS:           High (requires ATLAS algorithm)
Random 100:      Low
Metabench:       Low
TinyBenchmarks:  Very Low
```

## Usage Recommendations

### For Different Scenarios

**1. Research Publication**
- Primary: ATLAS
- Validation: Metabench (both versions)
- Baseline: Random 100

**2. Model Development**
- Initial screening: TinyBenchmarks
- Validation: Metabench
- Final estimate: ATLAS

**3. Large-Scale Evaluation (100+ models)**
- Use: TinyBenchmarks
- Validate subset with: Metabench
- Deep dive on top models: ATLAS

**4. Resource-Constrained**
- Use: TinyBenchmarks only
- If time permits: Add Metabench

## Workflow Example

### Complete Evaluation Pipeline

```bash
# Step 1: Generate all item IDs
cd /store01/nchawla/pli9/llmbenchmark/baseline_compare
python generate_all_tinybenchmark_ids.py
python generate_all_metabench_ids.py

# Step 2: TinyBenchmarks (fastest, initial screening)
bash run_pirt_tinybenchmark_all.sh

# Step 3: Random 100 (baseline)
bash run_pirt_random_100_all.sh

# Step 4: Metabench (validation)
bash run_pirt_metabench_all.sh

# Step 5: ATLAS (if already computed)
cd ..
bash run_pirt_all_benchmarks.sh
```

### Compare All Methods

```r
# Load all results for a benchmark
benchmark <- "arc"

# Load estimates
atlas <- read.csv(paste0("accuracy/", benchmark, "/pirt_accuracy_se_0.1.csv"))
random100 <- read.csv(paste0("baseline_compare/pirt_random_100_", benchmark, ".csv"))
metabench_p <- read.csv(paste0("baseline_compare/pirt_metabench_", benchmark, "_primary.csv"))
metabench_s <- read.csv(paste0("baseline_compare/pirt_metabench_", benchmark, "_secondary.csv"))
tiny <- read.csv(paste0("baseline_compare/pirt_tiny", benchmark, ".csv"))
actual <- read.csv(paste0("accuracy/", benchmark, "/actual_accuracy.csv"))

# Compute RMSE for each
rmse_atlas <- sqrt(mean((atlas$pirt_accuracy - actual$actual_accuracy)^2))
rmse_random <- sqrt(mean((random100$pirt_accuracy - actual$actual_accuracy)^2))
rmse_meta_p <- sqrt(mean((metabench_p$pirt_accuracy - actual$actual_accuracy)^2))
rmse_meta_s <- sqrt(mean((metabench_s$pirt_accuracy - actual$actual_accuracy)^2))
rmse_tiny <- sqrt(mean((tiny$pirt_accuracy - actual$actual_accuracy)^2))

# Print comparison
cat("RMSE Comparison for", benchmark, ":\n")
cat("  ATLAS:              ", round(rmse_atlas, 4), "\n")
cat("  TinyBenchmarks:     ", round(rmse_tiny, 4), "\n")
cat("  Metabench Primary:  ", round(rmse_meta_p, 4), "\n")
cat("  Metabench Secondary:", round(rmse_meta_s, 4), "\n")
cat("  Random 100:         ", round(rmse_random, 4), "\n")
```

## File Organization

```
baseline_compare/
├── # Random 100
│   ├── pirt_random_100.r
│   ├── compare_pirt_random_100.r
│   ├── run_pirt_random_100_all.sh
│   ├── random_100_{benchmark}_selected_items.csv
│   └── pirt_random_100_{benchmark}.csv
│
├── # Metabench
│   ├── pirt_metabench.r
│   ├── compare_pirt_metabench.r
│   ├── generate_all_metabench_ids.py
│   ├── run_pirt_metabench_all.sh
│   ├── metabench_{benchmark}_item_ids_{primary|secondary}.csv
│   └── pirt_metabench_{benchmark}_{version}.csv
│
├── # TinyBenchmarks
│   ├── pirt_tinybenchmark.r
│   ├── compare_pirt_tinybenchmark.r
│   ├── generate_all_tinybenchmark_ids.py
│   ├── run_pirt_tinybenchmark_all.sh
│   ├── tiny{benchmark}_item_ids.csv
│   └── pirt_tiny{benchmark}.csv
│
└── # Documentation
    ├── BASELINE_COMPARISON_SUMMARY.md (this file)
    ├── METABENCH_PIRT_README.md
    ├── METABENCH_QUICKSTART.md
    └── TINYBENCHMARK_PIRT_README.md
```

## Datasets Supported

All methods support:
- ARC (AI2 Reasoning Challenge)
- GSM8K (Grade School Math)
- HellaSwag
- TruthfulQA
- Winogrande

## Key Differences Summary

### Selection Method
- **ATLAS**: Adaptive (CAT algorithm)
- **Random 100**: Random sampling
- **Metabench**: Expert curation
- **TinyBenchmarks**: Expert curation for speed

### Subset Size
- **ATLAS**: 20-100 items (variable)
- **Random 100**: 100 items (fixed)
- **Metabench**: 50-100 items (fixed per version)
- **TinyBenchmarks**: 50-100 items (smallest)

### Number of Versions
- **ATLAS**: SE thresholds (0.1, 0.2, 0.3)
- **Random 100**: 1
- **Metabench**: 2 (primary/secondary)
- **TinyBenchmarks**: 1

### Same Items Across Models?
- **ATLAS**: No (adaptive per model)
- **Random 100**: Yes
- **Metabench**: Yes
- **TinyBenchmarks**: Yes

## Recommendations by Priority

### Priority: Accuracy
1. ATLAS
2. Metabench (average both versions)
3. TinyBenchmarks
4. Random 100

### Priority: Speed
1. TinyBenchmarks
2. Metabench
3. Random 100
4. ATLAS

### Priority: Simplicity
1. Random 100
2. TinyBenchmarks
3. Metabench
4. ATLAS

### Priority: Reproducibility
1. Metabench (published dataset)
2. TinyBenchmarks (published dataset)
3. Random 100 (fixed seed)
4. ATLAS (complex algorithm)

## Citation Guidance

When reporting results, consider citing:
- **ATLAS**: Your ATLAS paper/implementation
- **Random 100**: As "random baseline"
- **Metabench**: HCAI/metabench dataset
- **TinyBenchmarks**: tinyBenchmarks/tiny* datasets

## See Also

- Main p-IRT documentation: `../PIRT_ESTIMATION_README.md`
- Complete scripts list: `../PIRT_SCRIPTS_SUMMARY.md`
- Metabench guide: `METABENCH_PIRT_README.md`
- TinyBenchmarks guide: `TINYBENCHMARK_PIRT_README.md`
