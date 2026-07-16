# Theta Calculation Methodology

## Overview
The theta comparison scripts now use the same methodology as `metabench_theta.r` for calculating theta values from item subsets.

## Key Updates

### 1. Parameter Format
**Changed from**: Converting item parameters to IRT standard format (a, b, c)
```r
# Old approach
a = a1
b = -d / a1  # Convert from mirt's d to IRT's b
c = g
```

**Changed to**: Using mirt's native parameter format (a1, d, g) directly
```r
# New approach - matches metabench_theta.r
a1 = a1  # discrimination
d = d    # difficulty (mirt parameterization)
g = g    # guessing
```

### 2. Why This Matters
- **Consistency**: Matches the exact approach used in `metabench_theta.r`
- **Accuracy**: Avoids potential conversion errors when d/a1 produces extreme values
- **Direct mirt usage**: Uses mirt's native parameterization without intermediate conversions

### 3. Updated Files

#### `/store01/nchawla/pli9/llmbenchmark/baseline_compare/theta_estimation_utils.r`
- `estimate_theta_mirt()`: Now expects `a1`, `d`, `g` columns in item_params
- Added `estimate_theta_mirt_abc()`: Legacy function for backward compatibility

#### `/store01/nchawla/pli9/llmbenchmark/baseline_compare/compare_theta_whole_vs_reduced.r`
- Updated to prepare item parameters in `a1`, `d`, `g` format
- Passes parameters directly to mirt without conversion

## Theta Estimation Process

### Step-by-Step
1. **Load item parameters** (a1, d, g) from combined parameter files
2. **Load response matrix** for all models
3. **For each model and method** (TinyBenchmark, Random100, Metabench):
   - Extract responses for subset items
   - Match item parameters for those items
   - Build mirt 3PL model with fixed parameters
   - Estimate theta using EAP (Expected A Posteriori) method
   - Calculate MAE = |theta_reduced - theta_whole|

### mirt EAP Estimation
```r
# Build model with fixed parameters
mod_fixed <- mirt(response_df, 1, itemtype = "3PL", pars = pars,
                  technical = list(NCYCLES = 1000),
                  verbose = FALSE)

# Estimate theta using EAP
theta_scores <- fscores(mod_fixed, method = "EAP",
                       full.scores.SE = TRUE,
                       response.pattern = NULL,
                       quadpts = 61,
                       verbose = FALSE)
```

## Running the Analysis

### Single Benchmark
```bash
module load R
cd /store01/nchawla/pli9/llmbenchmark/baseline_compare

# Run for ARC
Rscript compare_theta_whole_vs_reduced.r --benchmark=arc

# Run for GSM8K
Rscript compare_theta_whole_vs_reduced.r --benchmark=gsm8k
```

### All Benchmarks
```bash
cd /store01/nchawla/pli9/llmbenchmark/baseline_compare
./run_theta_comparison_all.sh
```

## Output Interpretation

### Per-Model Results
Each CSV file contains:
- `theta_whole`: Theta estimated from all items
- `theta_tinybenchmark`: Theta estimated from ~100 TinyBenchmark items
- `theta_random100`: Theta estimated from 100 random items
- `theta_metabench_primary`: Theta estimated from Metabench primary subset
- `theta_metabench_secondary`: Theta estimated from Metabench secondary subset
- `mae_*`: Mean Absolute Error for each comparison

### Summary Statistics
- **MAE (mean)**: Average absolute difference across all models
- **MAE (SD)**: Standard deviation of absolute differences
- **Items used**: Number of items in each subset
- **Models processed**: Number of models successfully estimated

## Example Results (ARC)

```
SUMMARY STATISTICS - ARC
======================================================================

Total items in full benchmark: 839 

TinyBenchmark:
  Items used: 99 
  MAE (mean): 0.419642 
  MAE (SD): 0.331892 
  Models processed: 3747 

Random100:
  Items used: 100 
  MAE (mean): 0.510967 
  MAE (SD): 0.350682 
  Models processed: 3747 

Metabench Primary:
  Items used: 145 
  MAE (mean): 0.57605 
  MAE (SD): 0.331938 
  Models processed: 3747 

Metabench Secondary:
  Items used: 100 
  MAE (mean): 0.755044 
  MAE (SD): 0.371478 
  Models processed: 3747 
```

## Technical Notes

### Parameter Handling
- **a1**: Discrimination parameter (slope of item characteristic curve)
- **d**: Difficulty parameter in mirt's parameterization
- **g**: Guessing parameter (lower asymptote of 3PL model)
- **u**: Upper asymptote (typically fixed at 1)

### Relationship to IRT Standard
In standard IRT notation:
- `a` (discrimination) = `a1`
- `b` (difficulty) = `-d / a1`
- `c` (guessing) = `g`

The script uses mirt's native format to avoid the conversion step.

### Fallback Mechanism
If mirt estimation fails, the script uses a logit transformation:
```r
p_obs <- mean(responses, na.rm = TRUE)
avg_g <- mean(item_params$g, na.rm = TRUE)
p_adj <- max(0.01, min(0.99, (p_obs - avg_g) / (1 - avg_g)))
theta <- log(p_adj / (1 - p_adj))
```

## Validation

To verify results match `metabench_theta.r`:
```bash
# Run metabench_theta.r for a benchmark
Rscript metabench_theta.r  # (edit DATASET_NAME and METABENCH_SET first)

# Compare with comparison results
# The theta values should match for the same item subsets
```

## References
- **mirt package**: Chalmers, R. P. (2012). mirt: A Multidimensional Item Response Theory Package for the R Environment.
- **EAP estimation**: Expected A Posteriori - Bayesian estimation method
- **3PL model**: Three-Parameter Logistic model for item response theory
