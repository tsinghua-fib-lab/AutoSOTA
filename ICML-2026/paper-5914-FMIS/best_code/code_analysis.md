# Code Analysis — Paper 5914: Finding Most Influential Sets

## Evaluation Path
- Entry: `reproduce_benchmark.R`
- Creates synthetic data (n=1e6, univariate regression y ~ 0 + x)
- Fits `lm(y ~ 0 + x, data = d)` 
- Benchmarks `find_miss(m, k = 1e5)` via `microbenchmark(times=100)`
- Parses output: `Median: <value> ms` from stdout

## Critical Code Path (runtime-hot)
1. `find_miss()` → `get_miss_solver()` → Dinkelbach loop (R level) → `order_partial()` → `order_partial_cpp()` (Rcpp)
2. The C++ heap-based top-k scan over n=1e6 elements with k=1e5
3. The Dinkelbach loop runs ~5 iterations per call, each computing scores = num + lambda * den
4. All C++ code is inline via `Rcpp::sourceCpp(code = ...)` in `R/50_order-partial.R`

## Config Path
- No Makevars, no src/ directory
- Rcpp compilation flags: defaults (no -O3, no -march=native)
- Convergence tolerance: tol=1e-16 (in get_miss_solver)
- Rcpp sourceCpp uses default cache directory

## Metric Parser
- Parse `"Median: <value> ms"` from stdout
- `reproduce_benchmark.R` computes median from microbenchmark$time / 1e6

## Safe Modification Targets
1. **R/50_order-partial.R**: Add `-O3 -march=native -ffast-math` to Rcpp::sourceCpp flags
2. **R/50_order-partial.R**: Optimize C++ heap (branchless comparisons, reserve, prefetch)
3. **R/11_miss-enum.R**: Move Dinkelbach loop from R to C++ (eliminate R→C++ call overhead)
4. **R/11_miss-enum.R**: Adjust convergence tolerance (tol: 1e-16 → 1e-12)
5. **reproduce_benchmark.R**: Set Rcpp cache dir, pre-compile, avoid recompilation cost
6. **R/50_order-partial.R**: Fuse score computation into top-k scan (avoid separate pass)

## Risky Files (do not modify)
- `reproduce_benchmark.R` — evaluation protocol (can only add compilation flags, not change metric computation)
- Metric parsing logic — must not change
- Test data generation — must use same seed/fn

## Reusable Resources
- None; all data is synthetic, generated on-the-fly

## Baseline
- Primary metric: 278.04 ms median (n=1e6, k=1e5, 100 runs)
- Direction: lower is better
- Tolerance: 5% regression (~14 ms)
