#!/usr/bin/env bash
# BRIDGE analysis pipeline — runs all steps from raw data to final IRT parameters.

# Step 1: Parse raw Cybench evaluation logs into normalized results and human time estimates
uv run python parse_cybench_logs.py --verbose

# Step 2: Prepare SWE-bench IRT input from verified evaluation runs and model mapping
uv run python prepare_irt.py --verbose

# Step 3: Merge all benchmarks (SWE-bench, GDPVal, MLEBench, Cybench) into a single sparse py-IRT dataset
uv run python prepare_sparse_pyirt.py \
  --model-mapping data/model_run_mapping.json \
  --pyirt-input data/swe_a_pyirt.jsonl \
  --runs-input data/all_runs.jsonl \
  --gdpval-input data/gdpval_normalized_results.jsonl \
  --mlebench-input data/mlebench_normalized_results.jsonl \
  --cybench-input data/cybench_normalized_results.jsonl \
  --output data/all_a_pyirt.jsonl \
  --print-subject-counts \
  --keep-unmapped-pyirt-subjects \
  --verbose

# Step 4: Fit a two-parameter logistic (2PL) IRT model on the combined dataset
uv run python fit_irt.py --input_path data/all_a_pyirt.jsonl

# Step 5: Compute logit success-rate baseline estimates for comparison with IRT
uv run python compute_baseline.py --input_path data/all_a_pyirt.jsonl

# Step 6: Combine SWE-bench and Cybench human time annotations into a single file
cat data/human_minutes_by_task.jsonl data/cybench_human_minutes_by_task.jsonl > data/combined_human_minutes.jsonl

# Step 7: Attach human_minutes values to IRT item parameters by matching on task_id
uv run python merge_human_minutes.py --csv params/all_a_pyirt.csv --jsonl data/combined_human_minutes.jsonl

# Step 8: Annotate subject abilities with model release dates
uv run python add_release_time.py
