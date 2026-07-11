# Code Analysis — Paper 4413 (Learning-Augmented Paging)

## Evaluation Path
- **Command:** `bash scripts/run_popu.sh`
- **What it does:** Runs `python -m benchmark --boost --boost_fr --dataset $dataset --real --pred popu --dump_file --output_root_dir stat` for each of 13 SPEC CPU 2006 traces (4 parallel).
- **Aggregation:** `python scripts/aggregate_results.py --name popu --results_dir stat`
- **Key output row:** `RPB-OnlineMin[POPU]-msf-100-pb-1` (tau=1) = baseline Cost Ratio 1.2208, Hit Rate 0.2433

## Metric Parser
- **Source:** `scripts/aggregate_results.py`
- **Data source:** Per-dataset CSV files in `stat/<dataset>_popu_1.csv`
- **Columns:** Name, Hit, Miss, Total, Hit Rate, Cost Ratio
- **Aggregation:** Mean and std across 13 datasets for each algorithm variant
- **Cost Ratio formula:** `miss / opt_miss` where opt_miss comes from OPT row

## Config Path
- **Benchmark runner:** `benchmark/__main__.py` — parses args, registers algorithm variants, runs traces
- **Algorithm registration:** Lines 327-352 (POPU section) — registers RPB-OM at tau=0,1,2,4,8,16 and HC at same tau values
- **Cache config:** cache_line_size=64, capacity=2097152, associativity=16, hash_type=ShiftHashFunction
- **max_support_factor:** 100 (hardcoded in benchmark registration)

## Key Algorithm Files
- `cache/evict/algorithms.py` (1945 lines) — All eviction algorithms
  - `OnlineMinAlgorithm` (line 1017): Base randomized paging algorithm
  - `PredictiveOnlineMinAlgorithm` (line 1290): OnlineMin with predictor on L0 misses
  - `PredictiveRPBOnlineMinAlgorithm` (line 1438): Budget-gated predictor override (RPB-OM)
  - `PredictiveRPBOnlineMinHitCreditAlgorithm` (line 1653): HC variant with hit-accumulated credit
  - `Guard` (line 584): Error-detection guard wrapper
  - `CombineRandomAlgorithm` (line 886): THRESH blending
- `cache/evict/predictor.py` (432 lines) — Predictor implementations
  - `POPUPredictor` (line ~350): Predicts next arrival as t + t/count
  - `OracleReuseDistancePredictor`: Perfect future knowledge
- `cache/evict/evictor.py` — Evictor strategies (MaxEvictor, LRUEvictor, etc.)

## Known Levers
- **tau (pred_budget):** 0,1,2,4,8,16 already tested. tau=4 is optimal (CR=1.2123 vs baseline 1.2208)
- **max_support_factor:** Currently 100. Sweep {50, 100, 150, 200} worth testing
- **RPB variant:** RPB-OM vs RPB-OM-HC (HC is slightly better at tau=4: 1.2122 vs 1.2123)
- **Predictor:** POPU only (no training needed, deterministic given trace)
- **Cache capacity/associativity:** Fixed by SPEC CPU 2006 config

## Existing Results (from stat/)
| Algorithm | tau | Cost Ratio | Hit Rate |
|-----------|-----|-----------|----------|
| RPB-OM | 0 | 1.2375 | 0.2376 |
| RPB-OM | 1 | 1.2208 | 0.2433 | ← baseline
| RPB-OM | 2 | 1.2146 | 0.2450 |
| RPB-OM | 4 | 1.2123 | 0.2456 | ← best RPB-OM
| RPB-OM | 8 | 1.2124 | 0.2455 |
| RPB-OM | 16 | 1.2124 | 0.2455 |
| RPB-OM-HC | 0 | 1.2159 | 0.2441 |
| RPB-OM-HC | 1 | 1.2128 | 0.2456 |
| RPB-OM-HC | 2 | 1.2127 | 0.2455 |
| RPB-OM-HC | 4 | 1.2122 | 0.2456 | ← best overall
| RPB-OM-HC | 8 | 1.2123 | 0.2455 |

## Safe Modification Targets
- `cache/evict/algorithms.py`: Add new algorithm variants, modify gate logic, add parameters
- `benchmark/__main__.py`: Add/remove algorithm registrations, add new CLI args
- `scripts/`: Create new run scripts for specific configurations
- **DO NOT MODIFY:** `scripts/aggregate_results.py` (metric computation), trace data in `traces/`, scoring/parsing logic

## Red-Line Boundaries
- No changes to trace data in `traces/`
- No changes to evaluation metric definitions in `aggregate_results.py`
- No hard-coding per-dataset optimal values
- No changes to OPT computation or Cost Ratio formula

## Optimization Strategy
1. tau=4 is the simplest improvement (already verified): Cost Ratio 1.2123
2. Per-trace tau optimization may squeeze more (different traces have different optimal tau)
3. max_support_factor sweep could improve the cost-robustness tradeoff
4. Algorithmic variants (soft gate, confidence-weighted budget, Guard wrapper) for further gains
5. Oracle profiling to establish the performance ceiling
