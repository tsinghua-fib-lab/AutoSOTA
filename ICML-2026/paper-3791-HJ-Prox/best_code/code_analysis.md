# Code Analysis for Paper 3791: Operator Splitting with HJ-based Proximals

## Evaluation Path
- **Entry point**: `python3 evaluate.py` from `/repo`
- **What it does**: Generates synthetic LASSO data (n=250, p=500) with seed=112, runs analytical PGD (soft-thresholding, gold standard), then PGD-HJ (HJ-Prox for L1 proximal), reports final objective value.
- **Runtime**: ~45s on CPU (10k iterations x 1000 MC samples)

## Metric Parser
- **Primary metric**: `Objective Value` (lower is better)
- **Parse from stdout**: `grep "PGD-HJ final objective:"` and extract numeric value
- **Alternative**: Read `reproduction_results.json` key `pgd_hj_objective`
- **Also tracked**: `pgd_hj_best` (best objective observed, not the primary report metric)
- **Baseline**: 10.826
- **PGD analytical floor**: 10.750 (cannot beat this with approximate proximal)

## Config Path (all hardcoded in evaluate.py)
- `SEED = 112`, `DIM = 500`, `N_OBS = 250`
- `NOISE_LEVEL = 0.1`, `LAMBDA_1 = 1.0`
- `MAX_ITERS = 10000`, `NUM_SAMPLES = 1000`
- `EPS = 1e-5`, `STEP_FACTOR = 0.085`, `DELTA_FLOOR = 0.01`

## Key Files
- **evaluate.py**: Main evaluation script (data generation, PGD + PGD-HJ loops, result reporting)
- **hj_prox.py**: HJ-Prox implementation (MC sampling, softmax weighting, overflow handling)

## Safe Modification Targets
- `hj_prox.py`: MC sampling distribution (antithetic, QMC, importance-weighted)
- `hj_prox.py`: Softmax overflow handling (numerically stable softmax)
- `evaluate.py`: Delta schedule, step size, N schedule, momentum
- `evaluate.py`: Iterate averaging, best tracking

## Risky Files (DO NOT MODIFY)
- Data generation (SEED, DIM, N_OBS, NOISE_LEVEL, LAMBDA_1, x_true) — these define the problem
- Metric computation (lasso_objective) — must match paper definition
- Scoring scripts (/tools/record_score.sh)
- Evaluation protocol (what is reported as primary metric vs diagnostic)

## Paper Data
- No external datasets — all data generated procedurally from seed=112
- No model weights to download

## Notes
- PyTorch 2.1.2 produces different random numbers than paper's env with same seed
- Seed=112 chosen to match PGD baseline (10.750 vs paper's 10.751)
- Delta floor=0.01 prevents late-iteration instability
- The gap between PGD (10.750) and PGD-HJ (10.826) is due to MC proximal error
- Best iterate tracked as `best_f` but final iterate reported as primary metric
