# infrastructure/

System-level utilities: file handling, pipeline orchestration, SLURM job management.

## Key files

- **orchestration.py** — High-level functions that wire together the full pipeline. Each function coordinates a pipeline stage: `generate_ga_dataset()`, `create_propen_sft_dataset()`, `create_propen_preference_dataset()`, `train_initial_sft()`, `train_sft()`, `train_dpo()`, `train_marge()`, `run_iterative_generation()`, `run_compute_liks_all_models_and_cal_data()`. These are called from `main.py` and return file paths / metadata for downstream steps.
- **file_handler.py** — `LocalOrS3Client`: unified interface for local and S3 file operations (`exists()`, `ls()`, `get()`, `put()`, `open()`). Abstracts storage backend so the same code works locally and on cloud.
- **slurm_utils.py** — `submit_cmd_to_slurm()` and `wait_for_slurm_jobs_to_complete()` for distributed execution on HPC clusters. `submit_cmd_direct()` and `wait_for_direct_jobs_to_complete()` for local/Modal execution without a job scheduler.
- **process_utils.py** — Lower-level utilities for running shell commands (`get_output_or_exit()`), creating SLURM/LSF job submission scripts (`create_job_submission()`), and running parallel commands (`run_all_commands_and_wait_until_all_completed()`).
