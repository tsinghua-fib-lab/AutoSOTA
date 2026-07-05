# Code Analysis — Paper 3011: Asymmetric Perturbation in Bilinear Saddle-Point Optimization

## Evaluation Path
- **Entry point**: `/repo/nfg/main.py` — Hydra-based main with multiprocessing
- **Evaluation command**: `cd /repo/nfg && uv run main.py n_trials=100 game=biased_rps num_iters=100000 algorithm@player0=PGD player0.random_init=true player0.perturbation_strength=1.0 player0.learning_rate=0.01 algorithm@player1=GD player1.random_init=true player1.learning_rate=0.01`
- **Output**: CSV files at `nfg/logs/biased_rps_/<alg_names>/nash_conv_last_iterate.csv` — 100 columns (one per trial), last row (t=99999) averaged across columns gives NashConv
- **Individual metrics**: `player0_individual_nash_conv.csv`, `player1_individual_nash_conv.csv` at same path

## Metric Parsing
- NashConv: computed as `max(payoff @ s1) + max(-payoff.T @ s0)` — sum of both players' best-response exploitabilities
- Individual NashConvs: `[max(-payoff.T @ s0) - game_values[1], max(payoff @ s1) - game_values[0]]`
- Game value precomputed from known Nash equilibrium or via LP solver
- Output: last row (t=99999) of CSV, averaged across 100 trial columns

## Config Path
- `/repo/nfg/conf/config.yaml` — main config with defaults
- `/repo/nfg/conf/algorithm/GD.yaml` — GD config
- `/repo/nfg/conf/algorithm/OGD.yaml` — OGD config
- `/repo/nfg/conf/algorithm/MWU.yaml` — MWU config
- `/repo/nfg/conf/algorithm/OMWU.yaml` — OMWU config
- `/repo/nfg/conf/algorithm/PGD.yaml` — PGD config
- `/repo/nfg/conf/game/biased_rps.yaml` — BRPS game config

## Algorithm Files (safe modification targets)
- `/repo/nfg/algorithms/md.py` — MD base class, GD, MWU: **SAFE** — can add entropy regularization parameter
- `/repo/nfg/algorithms/omd.py` — OMD base, OGD, OMWU: **SAFE** — same modifications possible
- `/repo/nfg/algorithms/pgd.py` — PGD: **SAFE** — perturbation mechanism
- `/repo/nfg/runner/runner.py` — training loop: **SAFE** — can modify update order, add logging

## Risky Files (DO NOT MODIFY)
- `/repo/nfg/games/matrix_game.py` lines 78-95 — NashConv and individual_nash_convs computation
- `/repo/nfg/main.py` — evaluation output format
- `/repo/nfg/runner/logger.py` — log format
- `/tools/record_score.sh` — scoring script

## /paper_data Resources
- None needed — BRPS is a synthetic 3x3 matrix defined inline in matrix_game.py:biased_rps()

## Key Finding: Root Cause of NashConv=0.16
The PGD perturbation `-mu * strategy` is PROPORTIONAL to the strategy vector. Since PGD uses euclidean Bregman divergence with simplex projection, the perturbation term `(1 - lr*mu) * strategy` is normalized back to `strategy` by the projection (dividing by the sum). This means:

1. Player 0's FIXED POINT under PGD IS the original Nash equilibrium [0.2, 0.6, 0.2]
2. At this equilibrium, payoff * [0.2, 0.6, 0.2] = [0, 0, 0] (game value = 0 for BRPS)
3. Player 1's gradient (-payoff)^T * s0 = [0, 0, 0] — zero for ALL strategies
4. Zero gradient means NO mirror descent variant (GD, OGD, MWU, OMWU) can change strategy
5. Player 1 freezes at [0.28, 0.6, 0.12] — determined by the TRAJECTORY of Player 0's convergence

## Verified: Failed Approaches
- ALGO-01 (PGD+OGD): NashConv = 0.16 — OGD also freezes (gradient zero, strategy_hat unchanged)
- ALGO-02 (PGD+OMWU): NashConv = 0.16 — entropy update s*exp(0) = s, no change
- ALGO-03 (two-time-scale LR): NashConv = 0.16 — same attractor regardless of convergence speed

## Proposed Fix: Entropy-Regularized Objective for Player 1
Add entropy bonus to Player 1's gradient: `g_effective = g_payoff + tau * (log(s) + 1)`
- The entropy term is NOT proportional to strategy, so it survives simplex projection
- Provides a self-generated gradient that persists even when payoff gradient is zero
- Regularized equilibrium approaches Nash equilibrium as tau → 0
- Implementation: modify MD.add_gradient() to optionally include entropy term
