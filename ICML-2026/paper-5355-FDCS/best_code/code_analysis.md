# Code Analysis — Paper 5355

## Evaluation Path
- `python3 experiments/reproduce_fico.py` — main eval script
- Parses metrics from stdout: `Accuracy (our method, max_acc):`, `PPV (our method, common p):`, `FOR (our method, common q):`
- Also writes `/repo/experiments/fico_results.json`

## Key Files
- `experiments/reproduce_fico.py` — Data loading, GroupScoreDistribution construction, boundary trace, metric reporting
- `experiments/boundary_trace.py` — Core algorithm: GroupScoreDistribution class, Algorithm 1, Algorithm 2
- `experiments/plotting.py` — Visualization only

## Data Flow
1. Download FICO CSV data (cached at /datasets/fico/)
2. Construct score distributions: s = 1 - performance/100, w = diff(CDF)/100
3. Create GroupScoreDistribution objects for White and Black groups
4. Run trace_intersection to find optimal sufficient classifier
5. Report accuracy, PPV, FOR, delta_separation

## Safe Modification Targets
- reproduce_fico.py: Score distribution construction (s, w estimation from CSVs)
- boundary_trace.py: Loss function, preprocessing tolerances, boundary resolution
- Group definitions: multi-group extension using Hispanic/Asian data

## Baseline Metrics
- Accuracy: 0.8676, PPV: 0.9116, FOR: 0.2346, Delta_Separation: 0.0702
