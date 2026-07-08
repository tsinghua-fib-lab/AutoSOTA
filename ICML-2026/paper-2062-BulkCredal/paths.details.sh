#!/usr/bin/env bash

set -euo pipefail

# ---------------------------------------------------------------------
# Newsvendor Experiment Paths
# ---------------------------------------------------------------------
# Where the aggreated info experiment.json + real_world_experiment.slurm + plots/csv are written
export NEWSVENDOR_AGG_INFO_DIR="path/to/your/experiment_aggregated_info_directory"

# Where per-UUID CSVs and results.csv will be written. These files can be large (a few GB for each experiment).
export NEWSVENDOR_RUN_DATA_DIR="path/to/your/experiment_run_data_directory"

# ---------------------------------------------------------------------
# California Housing Experiment Paths
# ---------------------------------------------------------------------
# Where the aggreated info experiment.json + real_world_experiment.slurm + plots/csv are written
export CALIFORNIA_HOUSING_AGG_INFO_DIR="path/to/your/experiment_aggregated_info_directory"

# Where per-UUID CSVs and results.csv will be written. These files can be large (a few GB for each experiment).
export CALIFORNIA_HOUSING_RUN_DATA_DIR="path/to/your/experiment_run_data_directory"

# REQUIRED: Where the California Housing dataset is stored
export CALIFORNIA_HOUSING_DATASET_DIR="path/to/your/california_housing_dataset_directory"

# ---------------------------------------------------------------------
# Civilcomments Experiment Paths
# ---------------------------------------------------------------------
# Where the aggreated info experiment.json + real_world_experiment.slurm + plots/csv are written
export CIVILCOMMENTS_AGG_INFO_DIR="path/to/your/experiment_aggregated_info_directory"

# Where per-UUID CSVs and results.csv will be written. These files can be large (a few GB).
export CIVILCOMMENTS_RUN_DATA_DIR="path/to/your/experiment_run_data_directory"

# REQUIRED: the Civilcomments dataset is stored in "$CIVILCOMMENTS_DATASET_DIR/wilds/"
export CIVILCOMMENTS_DATASET_DIR="path/to/your/civilcomments_dataset_directory"

# REQUIRED: derived cache root. The cache can be large (~10 GB).
export CIVILCOMMENTS_CACHE_DIR="path/to/your/civilcomments_cache_directory"