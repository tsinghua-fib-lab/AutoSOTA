# Code Analysis — Paper 3498 SOTA Preparation Repair

## Original Failure

The SOTA preparation failed because:
1. `git` was not installed in the container image `autosota/paper-3498:reproduced`
2. `apt-get` failed due to proxy/network issues (502 Bad Gateway from Ubuntu mirrors)
3. First Docker run attempt with `--network host` was rejected by Docker auth plugin
4. Second Docker run succeeded, but `apt-get` still failed until proxy env vars were unset

## Repair

- Container: `autosota_sota_paper_3498` from `autosota/paper-3498:reproduced`
- Repo path: `/repo`
- Git installed via `unset HTTP_PROXY ... && apt-get install git`
- Baseline committed and tagged `_baseline`
- `record_score.sh` copied to `/tools/record_score.sh`

## Corrected Evaluation Command

```bash
cd /repo && python3 eval.py
```

Runs inside container. No host-side wrapping needed.

## Baseline Verification

| Metric | Manifest | Actual | Match |
|--------|----------|--------|-------|
| Spearman_Objective_Cost | 120,826 | 120,826 | ✅ |
| Kendall_Tau_Objective_Cost | 95,181 | 95,181 | ✅ |

## Optimization Targets

The core optimization target is the `eval.py` script, specifically:
- `solve_assignment_lp()` — LP solver for fair assignment (lines 130-270)
- `algorithm_3()` — Main Algorithm 3 with forward/reverse directions (lines ~430-520)
- `main()` — Evaluation entry point with BFI baseline and metrics output (lines ~530-369)

### Safe Optimization Surface

1. **LP solver parameters:** `method='highs'`, options dict, tolerance values
2. **Objective function:** Replace `leftCostRank` with `trueCostRank` or hybrid
3. **Post-processing:** Local search, tie-breaking, direction fusion after LP
4. **Algorithm structure:** Split-points, remaining-position assignment method
5. **Solver restart:** Multi-start with cost perturbation

### No-Go Zones

- Data loading (`parse_input`)
- Metric computation (`get_obj_cost`, `get_kt_obj_cost`)
- Fairness definition (`is_fair`, ALPHA/BETA)
- Dataset files (`Movielens/movielens.in`)
- Evaluation protocol (output format, metric names)

## Reusable Resources

No pre-downloaded data mounts. The Movielens dataset is included in the repo at `/repo/Movielens/movielens.in` (268 candidates, 8 genre groups, 7 rankings). No external downloads needed.

## Implemented Optimizations

1. **ID-01: Two-Phase Lexicographic LP** — After first LP (leftCostRank), solve second LP with trueCostRank objective constrained to leftCostRank-optimal value
2. **ID-02: Multi-Restart LP** — Perturb objective vector with small noise (1e-4), solve N times, pick best trueCostRank
3. **ID-04: Forward-Reverse Direction Fusion** — Fuse sigma1 and sigma2 position-wise, picking best item per position
4. **ID-05: Systematic Local Search** — Check all adjacent swaps, then windowed swaps, with perturbation restart
5. **ID-08: Borda Warmstart** — Compare Borda ordering vs LP for remaining-position assignment

## Best Configuration

- 3 LP restarts with 1e-4 perturbation
- Two-phase lexicographic LP enabled
- Direction fusion enabled
- Borda comparison for remaining positions
- Systematic adjacent-swap LS (30 passes) + windowed LS (50-pos window, 10 passes) + perturbation restart (3 rounds)
- Runtime: ~27 seconds
