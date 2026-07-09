# Code Analysis — FairRARI SOTA Optimization

## Evaluation Path
- `eval_fairrari.py` → loads graph via `init_graph()`, computes original PR via `nx.pagerank()`, runs `fairPageRank.sum_fair_FairRARI()`, computes TV and KendallTau.
- Metric output: `FINAL_METRICS: TV=<float> KendallTau=<float> AchievedFairness=<float>`

## Key Files
| File | Role | Safe to modify |
|------|------|---------------|
| `eval_fairrari.py` | Evaluation harness | Yes — pass new params, track additional diagnostics |
| `fairPageRank.py` | Core algorithm (FairRARI + post-processing) | Yes — algorithmic variants |
| `init_graph.py` | Graph loading | No — changes test data |
| `utils.py` | Utility functions | Yes — but not usually needed |

## Safe Modification Targets
1. `fairPageRank.py::sum_fair_FairRARI()` — the main in-processing algorithm
   - Lines 117-119: initialization (`nstart`)
   - Lines 122-126: personalization vector (`p`)
   - Line 131: PR update step (momentum, acceleration)
   - Lines 134-137: convergence check (commented out)
   - After line 135: EMA smoothing

2. `fairPageRank.py::projection_sum_fair_simplex()` — projection step
   - Bisection tolerance (currently hard-coded 1e-6)
   - Per-group tolerance scaling

3. `eval_fairrari.py` — pass `nstart`, `personalization`, track loss

## Risky Files (do not modify)
- `init_graph.py` — defines data loading, protected node assignment
- `datasets/` — test data
- Scoring scripts, metric definitions

## Key Findings from Convergence Analysis
- Algorithm converges at iteration **14** (err=7.6e-5 < N*tol=9.2e-5)
- TV stabilizes to 0.329548 at iteration 14 and stays constant through 10000
- ~99.86% of iterations are wasted with no change in result
- The structural bottleneck is NOT iteration count — it is the initialization and teleportation strategy

## Baseline Metrics
- TV: 0.3296, KendallTau: 0.4247, AchievedFairness: 0.800003
- Original phi: 0.4714, Target phi: 0.8000
- Graph: PolBooks, 92 vertices, 374 edges, undirected
