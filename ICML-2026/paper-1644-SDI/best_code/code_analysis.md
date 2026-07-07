# Code Analysis — Paper 1644 (Step-Resolved Data Attribution for Looped Transformers)

## Evaluation Path

**Entry point**: `evaluate.py`
**Command**: `python3 evaluate.py --device cuda:0 --n-trials 10 --output outputs/metrics.json`

### Evaluation flow:
1. Creates `RandomTokenDataset` with synthetic random tokens (N_TRAIN=4, N_QUERY=4, SEQ_LEN=128, VOCAB=50304)
2. Instantiates `LoopedGPT2` (134.6M params, tau=32)
3. For each trial (default 10):
   a. Generates data with `trial_seed = args.seed + trial * 1000`
   b. Runs `FullGradientTracInSDI` (exact baseline) → exact SDI, exact TracIn
   c. Runs `ProjectedTracInSDI` (sketched, M=2048) → sketch SDI, sketch TracIn
   d. Computes `relative_error(sketch, exact)` for both SDI and TracIn
   e. Records runtime overhead
4. Aggregates across trials: mean and std of errors + runtime
5. Saves to `outputs/metrics.json`

## Metric Parsing

The JSON output file `outputs/metrics.json` contains:
- `relative_sdi_error_mean`: primary optimization metric (lower is better)
- `relative_sdi_error_std`: trial-level standard deviation
- `relative_tracin_error_mean`: secondary guardrail metric
- `relative_tracin_error_std`: trial-level standard deviation
- `runtime_overhead_mean_s`: resource guardrail metric
- `per_trial`: list of per-trial results

## Key Source Files

| File | Role | Safe to Modify |
|------|------|----------------|
| `evaluate.py` | Evaluation entry point, trial loop | Yes — add flags, seed plumbing |
| `src/sdi/estimators.py` | Core estimators (ProjectedTracInSDI, FullGradientTracInSDI) | Yes — feature storage, dot-product computation |
| `src/sdi/sketch.py` | CountSketch/TensorSketch primitives | Yes — hash functions, FFT caching |
| `src/sdi/runner.py` | Checkpoint loading, output dataclasses | No (plumbing, not algorithmic) |
| `looped_gpt2.py` | Model definition | No (model architecture fixed) |

## Modification Targets

### Safe:
- `evaluate.py:M=2048` → add `--m` CLI flag for dimension sweep
- `evaluate.py:trial_seed` → decompose into data_seed + sketch_seed
- `evaluate.py:trial loop` → antithetic seed pairing
- `src/sdi/estimators.py:_handle_module_backward` → FP16 cast before append
- `src/sdi/estimators.py:compute_query_sdi` → FP32 cast before dot product
- `src/sdi/estimators.py:compute_tracin` → FP32 cast before matmul
- `src/sdi/sketch.py:make_count_sketch` → CW pairwise-independent hash
- `src/sdi/sketch.py:make_tensor_sketch` → CW pairwise-independent hash

### Risky:
- `src/sdi/estimators.py:_finalize_features` — step counting logic
- `src/sdi/runner.py` — checkpoint loading protocol

## Constraints
- Model architecture (LoopedGPT2) unchanged
- Evaluation protocol (10 trials, self-influence, relative error metric) unchanged
- Test data unchanged (synthetic random tokens generated on-the-fly)
- Metric computation unchanged
