# Code Analysis — Paper 4975: Learning to Rank from Incomplete Rankings

## Evaluation Path
- **Entry point**: `eval_pl_mcar.py` — imports config inline (not from configs/)
- **Experiment runner**: `experiment/experiment.py` → `Experiment.run()`
- **Metric**: `loss/kendall_tau.py` — normalized top-k Kendall tau distance (lower=better)
- **Output parsing**: Parse stdout line `Primary metric (PIRATE avg normalized kendall): <value>`

## Train/Inference Path
- All algorithms are online (no separate training phase): feedback → predict
- `Experiment.run()` iterates samples, calls `algo.feedback(feedback)`, periodically calls `algo.predict()` and computes loss
- Four algorithms: PIRATE, RankCentrality, BordaCount, PL Pairwise MLE
- Adapter chain: RankBreakableToPreferenceAdapter wraps RankCentrality/BordaCount/PLPairwiseMLE to accept PartialOrderFeedback

## Config Path
- `eval_pl_mcar.py` contains all config inline (n=10, num_samples=500000, k=10, num_runs=5, seed=42)
- PL parameters: `np.random.uniform(0, 1, size=n)` with seed 42

## Metric Parser
- Parse from stdout: `Primary metric (PIRATE avg normalized kendall): 0.022200`
- Per-algorithm lines: `PIRATE: mean=0.022200 std=0.000000`
- Results also saved to `output/eval_results.json`

## Reusable Resources
- No `/paper_data` mount, no external datasets/models
- Pure synthetic data generation from PlackettLuceRankingModel
- Cache mounts: `/autosota_cache`, `/datasets`, `/models`

## Key Algorithm Files

### PIRATE (`algos/pirate.py`) — PRIMARY OPTIMIZATION TARGET
- **Bottleneck**: Boolean majority `E[i,j] = mu_ij > mu_ji` discards continuous win-ratio info
- **Transitive closure**: Floyd-Warshall via boolean matrix ops amplifies one wrong boolean edge
- **Out-degree scoring**: Counting boolean reachable nodes treats all wins equally
- **Safe modifications**: `predict()` method, `feedback()` win-tracking

### PL Pairwise MLE (`algos/pl_pairwise_mle.py`)
- **Bottleneck**: `lambda_reg=0.01` hardcoded, may be suboptimal for near-tie pairs
- **Safe modifications**: `lambda_reg` parameter, optimization options

### RankCentrality (`algos/rank_centrality.py`)
- Good reference implementation — already achieves 0.0133 on seed 42 vs PIRATE's 0.0222
- Uses continuous stationary distribution from win-ratio transition matrix

### BordaCount (`algos/borda_count.py`)
- Simple continuous scoring: wins/total_comparisons per item
- Also achieves 0.0133 on seed 42

## Risky Files (DO NOT MODIFY)
- `loss/kendall_tau.py` — metric definition
- `experiment/experiment.py` — evaluation protocol
- `feedback/*` — feedback generation
- `ranking_models/*` — data generation
- `data_generator/*` — data generation
- `pos_filter/*` — sampling mechanism
- `eval_pl_mcar.py` — primary evaluation script (ok to change algorithm imports additively)

## Safe Modification Targets
- `algos/pirate.py` — predict() method (continuous scoring)
- `algos/pl_pairwise_mle.py` — lambda_reg, optimization settings
- New algorithm files in `algos/` — additive changes only
