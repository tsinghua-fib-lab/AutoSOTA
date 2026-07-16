# TinyBenchmark IRT Calibration

This directory contains scripts to perform 3PL IRT calibration on TinyBenchmark items using MetaBench data.

## Overview

The goal is to estimate item parameters for TinyBenchmark items by:
1. Loading TinyBenchmark item IDs for each dataset
2. Extracting model performance (0/1) from MetaBench data for these items
3. Filtering to models present in the training datasets
4. Performing 3PL IRT calibration to obtain item parameters

## Files

### Main Scripts
- `irt_tinybenchmark_items.r` - Main R script that performs IRT calibration
- `run_irt_tinybenchmark.job` - SLURM job script for single benchmark
- `run_irt_tinybenchmark_all.sh` - Shell script to submit jobs for all benchmarks

### Input Files
- `tiny{benchmark}_item_ids.csv` - TinyBenchmark item IDs for each dataset
  - Example: `tinytruthfulqa_item_ids.csv`, `tinywinogrande_item_ids.csv`
- `/store01/nchawla/pli9/llmbenchmark/metabench_data/benchmark-data/{benchmark}.csv` - MetaBench performance data
  - Contains: `source` (model name), `item` (item ID), `correct` (True/False)
- `/store01/nchawla/pli9/llmbenchmark/select_model/gaussian_sampled_{benchmark}_response_matrix_train.csv` - Training model list
  - Used to filter which models to include in calibration

### Output Files
All outputs are saved to `tinybenchmark_calibration/`:
- `irt_item_parameters_tiny{benchmark}.csv` - Item parameters (a1, d, g, u)
- `irt_person_scores_tiny{benchmark}.csv` - Model theta scores (ability estimates)
- `response_matrix_tiny{benchmark}.csv` - Response matrix used for calibration

## Usage

### Run for a single benchmark
```bash
# Interactive
Rscript irt_tinybenchmark_items.r --benchmark=truthfulqa

# Via job submission
qsub -v BENCHMARK=truthfulqa run_irt_tinybenchmark.job
```

### Run for all benchmarks
```bash
bash run_irt_tinybenchmark_all.sh
```

This will submit jobs for: truthfulqa, winogrande, hellaswag, gsm8k, and arc.

## Supported Benchmarks
- `truthfulqa`
- `winogrande`
- `hellaswag`
- `gsm8k`
- `arc`

## Output Format

### Item Parameters CSV
Contains IRT parameters for each item:
- Row names: Item IDs
- Columns: `a1` (discrimination), `d` (difficulty), `g` (guessing), `u` (upper asymptote)

Note: These can be converted to standard 3PL format:
- a = a1 (discrimination)
- b = -d/a1 (difficulty)
- c = g (guessing parameter)

### Response Matrix CSV
- Rows: Model names
- Columns: Item IDs
- Values: 0 (incorrect) or 1 (correct)

### Theta Scores CSV
- Rows: Models (same order as response matrix)
- Columns: `F1` (theta estimate), `SE_F1` (standard error)
- Method: EAP (Expected A Posteriori) with 61 quadrature points

## Data Cleaning

The script automatically:
1. Removes rows/columns with all NAs
2. Removes constant columns (all 0s or all 1s)
3. Removes constant rows (models that got everything right or wrong)
4. Ensures sufficient data for calibration (≥10 models, ≥5 items)

## Model Fitting

- Method: 3PL IRT with EM algorithm
- Maximum cycles: 100,000
- Package: `mirt` in R

## Verification

After running, check:
```bash
# List output files
ls -lh tinybenchmark_calibration/

# View item parameters
head tinybenchmark_calibration/irt_item_parameters_tinytruthfulqa.csv

# View theta scores
head tinybenchmark_calibration/irt_person_scores_tinytruthfulqa.csv

# Check response matrix dimensions
head -5 tinybenchmark_calibration/response_matrix_tinytruthfulqa.csv | cut -d',' -f1-10
```

## Troubleshooting

### No matching data found
- Check that TinyBenchmark item IDs exist in MetaBench data
- Verify item ID format matches (e.g., "74" vs 74)

### Not enough models/items after cleaning
- Check for data quality issues in MetaBench data
- Verify training models file is accessible
- Review which models overlap between TinyBenchmark and training data

### Model convergence issues
- Check for items with extreme difficulty or discrimination
- Review response patterns in response_matrix file
- Ensure sufficient response variance across models and items

## Next Steps

After obtaining item parameters and theta scores, you can:
1. Compare item difficulties between TinyBenchmark and full dataset
2. Validate theta estimates against actual model performance
3. Use parameters for p-IRT accuracy estimation on new models
4. Analyze model abilities across different benchmarks

## Contact
For questions or issues, contact: pli9@nd.edu
