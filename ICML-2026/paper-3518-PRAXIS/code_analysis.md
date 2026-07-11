# PRAXIS Code Analysis for SOTA Optimization

## Evaluation Path
- **Script**: `/repo/eval_praxis.py`
- **Command**: `python3 eval_praxis.py`
- **Metrics parsed from stdout**: recall, time_s, peak_mb, n_trees, n_trees_exact, min_objective
- **Timeout**: 10 minutes

## Key Files
- `/repo/eval_praxis.py` — Evaluation script (read-only, do not modify)
- `/repo/src/praxis/__init__.py` — Python wrapper with fit() defaults and visualization
- `/repo/src/praxis/_core.cpp` — pybind11 wrapper bridging Python to C++
- `/repo/src/praxis/cpp/praxis.cpp` — Main C++ implementation (4156 lines)
- `/repo/examples/compas_binarized.csv` — Compas dataset

## Config Path
- Fit parameters are passed programmatically in eval_praxis.py
- lambda_reg=0.02, depth_budget=5, rashomon_mult=0.03
- lookahead_k=1, key_mode="hash", proxy_style=0, proxy_caching=True
- NOT configurable via config file

## Key Architecture
1. PRAXIS::fit() (praxis.cpp:667-850): Entry point, bit-packed data, best objective, trie construction
2. construct_trie() (praxis.cpp:1330-1508): Recursive AND/OR graph with budget pruning
3. generalized_lickety_split() (praxis.cpp:2386-2504): LicketySPLIT proxy with lookahead
4. train_greedy() (praxis.cpp:1791-1880): Greedy split selection with caching
5. build_histograms_post() (praxis.cpp:363-400): Post-order histogram for tree counting

## Caches
- greedy_cache: K2 to int (train_greedy results)
- lickety_cache_k2/kla: K2/KLA to int (generalized_lickety_split results)
- trie_cache: K3 to shared_ptr (construct_trie results, enabled via trie_cache_enabled)

## Safe Modification Targets
- Python: fit() default parameters, evaluation script wrapper
- C++ hot path: construct_trie pruning, generalized_lickety_split, cache structures
- Build: pip install -e . from /repo to rebuild after C++ changes

## Risky Files (do not modify)
- eval_praxis.py — evaluation protocol
- examples/compas_binarized.csv — test data
- src/praxis/_threshold_guessing.py — unrelated binarization

## Baseline Metrics
- Recall: 1.0, Time: 0.103s, Peak MB: 213.72
- n_trees: matches exact count at lookahead_k=4
