# Code Analysis — Paper 2393: Fast k-means Seeding Under The Manifold Hypothesis

## Evaluation Path

- **Binary**: `./bin/run_comparison` (compiled from `src/bin/run_comparison.cpp`)
- **Config**: `configs/eval_comparison.json`
- **Evaluation command**: `./bin/run_comparison configs/eval_comparison.json`
- **Output**: `results/eval_comparison.csv` (CSV with columns: dataset, method, k, seeding_cost, seeding_time_ms)
- **Metric parsing**: Row where `method=qkmeans` and `k=10`. Cost = `seeding_cost / 1e11`, Time = `seeding_time_ms`

## Architecture

```
src/
  algorithms/
    qkmeans.hpp          ← Main QKMEANS algorithm (header-only)
    afkmc2.hpp           ← AFKMC2 baseline
    kmeanspp.hpp         ← KMeans++ baseline
    prone.hpp            ← PRONE baseline + D2SegmentTree
  core/
    dataset.hpp          ← Dataset class (I/O, clustering_cost)
  bin/
    run_comparison.cpp   ← Comparison runner (main eval binary)
    run_single.cpp       ← Single algorithm runner
    run_sweep.cpp        ← Hyperparameter sweep runner
```

## Safe Modification Targets

1. `src/algorithms/qkmeans.hpp` — Main algorithm. Safe to modify:
   - `run()` signature to accept M, ef_construction (line ~110)
   - Inner rejection loop (lines 184-204) for adaptive chain length
   - First center selection (lines 143-146) for better initialization
   - Acceptance ratio computation (lines 194-197)

2. `src/bin/run_comparison.cpp` — Eval binary. Safe to modify:
   - Add config parsing for M, ef_construction (lines 62-64)
   - Pass new params to QKMEANS::run()

3. `configs/eval_comparison.json` — Config. Safe to add new keys.

## Risky Files (Do Not Modify)

- `src/core/dataset.hpp` — Dataset I/O and clustering_cost (evaluation metric)
- Datasets (`datasets/mnist.txt`, `datasets/mnist_labels.txt`) — Test data

## Baseline Metrics

- QKMEANS: Cost=2.80, Time=14.31ms (from scores.jsonl iter=0)
- AFKMC2: Cost=2.74, Time=30.65ms
