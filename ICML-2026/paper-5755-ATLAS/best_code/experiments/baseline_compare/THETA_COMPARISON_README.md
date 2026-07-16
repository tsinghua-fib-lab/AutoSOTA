# Theta Comparison: Theta_Whole vs Theta_Reduced

This directory contains scripts to compare `theta_whole` (ability estimates from all items) with `theta_reduced` (ability estimates from item subsets) across different sampling methods.

## Overview

The comparison evaluates three item selection methods:
1. **TinyBenchmark**: ~100 items selected using the TinyBenchmark methodology
2. **Random100**: 100 randomly selected items
3. **Metabench Primary/Secondary**: Items selected using Metabench methodology

## Files

### R Scripts
- **`compare_theta_whole_vs_reduced.r`**: Main R script (parameterized by benchmark)
  - Calculates theta_reduced for each method
  - Compares with theta_whole using MAE (Mean Absolute Error)
  - Generates detailed CSV output

### Shell Scripts
- **`run_theta_comparison_all.sh`**: Runs comparison for all benchmarks
  - Processes: arc, gsm8k, hellaswag, truthfulqa, winogrande
  - Creates individual CSV files for each benchmark
  - Generates a summary table across all benchmarks

## Usage

### Run for a single benchmark:
```bash
module load R
cd /store01/nchawla/pli9/llmbenchmark/baseline_compare
Rscript compare_theta_whole_vs_reduced.r --benchmark=arc
```

### Run for all benchmarks:
```bash
cd /store01/nchawla/pli9/llmbenchmark/baseline_compare
./run_theta_comparison_all.sh
```

## Output Files

### Per-Benchmark Results
- `theta_comparison_arc.csv`
- `theta_comparison_gsm8k.csv`
- `theta_comparison_hellaswag.csv`
- `theta_comparison_truthfulqa.csv`
- `theta_comparison_winogrande.csv`

Each file contains:
- `model`: Model name
- `theta_whole`: Theta from all items
- `theta_tinybenchmark`: Theta from TinyBenchmark subset
- `theta_random100`: Theta from Random100 subset
- `theta_metabench_primary`: Theta from Metabench primary subset
- `theta_metabench_secondary`: Theta from Metabench secondary subset
- `mae_*`: Mean Absolute Error for each method
- `n_items_*`: Number of items used in each subset

### Summary Table
- `theta_comparison_summary.csv`: Aggregated statistics across all benchmarks

## Metrics

### MAE (Mean Absolute Error)
- **Definition**: `|theta_reduced - theta_whole|`
- **Interpretation**: Lower MAE indicates better ability estimation from subset
- **Reported**:
  - Mean MAE across all models
  - SD (standard deviation) of MAE

### Items Used
Number of items in each subset for theta estimation

## Requirements

### Input Files Required
For each benchmark `<benchmark>`:

1. **Item Parameters**: 
   - Path: `/store01/nchawla/pli9/llmbenchmark/<benchmark>/irt_item_parameters_combined.csv`
   - Contains: a1, d, g parameters for 3PL IRT model

2. **Theta Whole**:
   - Path: `/store01/nchawla/pli9/llmbenchmark/<benchmark>/irt_person_scores_WLE_SE.csv`
   - Contains: Model_Name, Theta_WLE, SE

3. **Response Matrix**:
   - Path: `/store01/nchawla/pli9/llmbenchmark/select_model/gaussian_sampled_<benchmark>_response_matrix_train.csv`
   - Contains: Model responses to all items

4. **Item Selection Files**:
   - TinyBenchmark: `tiny<benchmark>_numeric_indices.csv`
   - Random100: `random_100_<benchmark>_selected_items.csv`
   - Metabench: `metabench_<benchmark>_item_ids_primary.csv` and `metabench_<benchmark>_item_ids_secondary.csv`

## Example Output

```
======================================================================
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

## Interpretation

- **Lower MAE** = Better approximation of theta_whole
- **Lower MAE SD** = More consistent estimates across models
- Compare different methods to identify the most efficient item subset for theta estimation

## Notes

- Uses `mirt` package with EAP (Expected A Posteriori) estimation
- Theta estimation uses 3PL IRT model
- Missing data handled gracefully (returns NA for failed estimations)
