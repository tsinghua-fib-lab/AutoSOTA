# Code Analysis for Paper 492: I-PERI SOTA Optimization

## Evaluation Path
- `eval.py` at repo root. Runs 10 seeds x 4 n_clients x (3 n_samples + 1 horizontal split variant) = 160 configurations
- Each config: generate synthetic data -> run PC on each client -> run I-PERI (CPDAG phase 10 iters + orientation phase 1 iter)
- Metrics: SHD (Structural Hamming Distance) and F1 (orientation F1) computed via `utils.shd()` and `utils.f1_orientation()`
- Output format: "SHD: X.XX +/- Y.YY" and "F1:  X.XX +/- Y.YY" lines at end of stdout

## Config Path
- No external config file. All parameters are inlined in eval.py and source defaults.

## Metric Parser
- Parse stdout for lines matching: `SHD: <mean> +/- <std>` and `F1:  <mean> +/- <std>`
- Use mean values for score recording.

## Reusable Resources
- No pre-downloaded paper data. All data is synthetically generated via `reproducibility/icml2026/dataset.py`.
- Container has pyciphod installed with reproducibility extras.

## Safe Modification Targets
1. **`eval.py`**: Can change `max_iters` parameter (hyperparameter), add new CLI args for tuning
2. **`iperi.py`**: CPDAG phase max_iters (line 79), orientation phase (line 92), patience, tol
3. **`score.py`**: `_compute_local_score` (line 106-136): regret aggregation, sparsity penalty
4. **`client.py`**: PEN_COEFF (line 9), scoring class parameters, cd_function parameters
5. **`utils.py`**: `pc_wrapper` alpha (line 64), `get_scoring_class` BIC coefficient (line 31), new helper functions

## Risky Files (avoid modifying)
- `dataset.py`: Data generation -- changing would alter evaluation protocol
- `ges/`: Core GES algorithm -- complex, changing could break correctness
- `utils.py`: `shd()`, `f1_orientation()`, `f1_skeleton()`, `shd_skeleton()` metric functions

## Key Bottlenecks Identified
1. **score.py:134**: `min(regrets)` -- single worst client dominates, no robustness
2. **eval.py**: `max_iters=1` for orientation -- likely insufficient for convergence
3. **utils.py:31**: BIC lambda=0.5*log(n) -- asymptotic, not optimal for small n
4. **utils.py:64**: PC alpha=0.1 hardcoded -- not tuned for p=3 ER graphs
5. **client.py:9**: PEN_COEFF=1e4 constant -- may prematurely lock orientations
6. **iperi.py:79**: CPDAG max_iters=10 hardcoded (likely sufficient at p=3)

## Evaluation Timing Note
- 160 configs x O(1s) each ~= 2-5 minutes expected. 5-min timeout is tight.
- Progress bar uses tqdm; exceptions are silently caught.
