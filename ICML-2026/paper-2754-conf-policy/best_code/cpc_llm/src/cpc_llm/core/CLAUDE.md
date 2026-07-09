# core/

Model inference and likelihood computation. Wraps HuggingFace transformers for sequence generation and log-likelihood evaluation.

## Key files

- **model_client.py** — `ModelClient` class: loads a HuggingFace causal LM, handles CUDA initialization with exponential backoff, computes log-likelihoods via `compute_likelihoods_avg()`, and generates sequences via `generate()`.
- **likelihoods.py** — Consolidated likelihood computation module. `compute_likelihoods_inmemory()`: core in-memory function that accepts DataFrames and returns a DataFrame with `lik_r{i}` columns appended. Used directly by `rejection_sampling.py`. `compute_likelihoods_all_models_one_target()` and `compute_likelihoods_one_model_all_data()`: file I/O wrappers for subprocess execution.
- **compute_liks_all_models_one_target.py** — Backward-compatible shim, re-exports from `likelihoods.py`. Preserves `python -m` entry point for subprocess invocation.
- **compute_likelihoods_one_model_all_data.py** — Backward-compatible shim, re-exports from `likelihoods.py`. Preserves `python -m` entry point for subprocess invocation.

## Data structures

Likelihood matrices are (n_particles, n_models) NumPy arrays stored as JSONL DataFrames with columns `lik_r0, lik_r1, ...` (one per model checkpoint).
