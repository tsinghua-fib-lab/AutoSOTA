# Code Analysis — Paper 741 (RCDP-UCB)

## File Map

| File | Lines | Role |
|------|-------|------|
| `main.py` | ~120 | CLI entry point; argparse + delegation |
| `experiments.py` | ~400 | Experiment registry, driver, plotting, CSV output |
| `contextual_dueling_bandit.py` | ~800 | Core: env, learners, MLE, simulation loop |

## Evaluation Path

```
main.py run post-serving --phi sinusoidal --delay stochastic --d 10 --K 10 --corruption_budget 25 --mean 100 --std 100 --n_runs 10 --T 2000
  → experiments.run_one("post-serving", cfg, T=2000, n_runs=10)
    → experiments.run_mapping(exp, cfg, d=10, e=20, K=10, phi="sinusoidal", T=2000, n_runs=10)
      → For each run: builds env + _ours(d=10, e=20) learner → run_simulation(env, learner, T)
      → Collects regrets[method][run, T] arrays
    → _save(exp, label, names, regrets) → writes CSV + PDF to results/post-serving/
    → Prints ranked final regrets to stdout
```

## Key Classes

- `DuelingGLMLearner` (lines ~240-400): RCDP-UCB (OURS). Main optimization target.
- `NeuralApproximator` (lines ~230-250): MLP `[d→64→64→e]` with ReLU activations.
- `run_simulation()` (lines ~490-560): The simulation loop.

## Configurable Parameters (Hard-Coded)

| Parameter | Value | Location | Safe to Tune? |
|-----------|-------|----------|---------------|
| MLP hidden units | [64, 64] | contextual_dueling_bandit.py:234-236 | Yes |
| MLP learning rate | 1e-3 | contextual_dueling_bandit.py:262 | Yes |
| MLP epochs/round | 2 | contextual_dueling_bandit.py:399 | Yes |
| MLP loss | MSELoss | contextual_dueling_bandit.py:263 | Yes |
| MLP optimizer | Adam | contextual_dueling_bandit.py:262 | Yes |
| Batch size | full-batch | contextual_dueling_bandit.py:311 | Yes |
| Gradient clipping | None | — | Yes |
| LR scheduler | None | — | Yes |
| UCB alpha | 0.1 | experiments.py:54 (passed) | Yes |
| lambda_reg | 1.0 | experiments.py:54 | Yes |
| MLE max_iter | 50 | contextual_dueling_bandit.py:280 | Yes |
| MLE tol | 1e-7 | contextual_dueling_bandit.py:280 | Yes |
| FIXED_C | 25.0 | experiments.py:33 | Environmental (match corruption_budget) |
| FIXED_LAMBDA | 10000.0 | experiments.py:33 | Yes (robustness trade-off) |
| FIXED_MU_TAU | 100.0 | experiments.py:33 | Yes (delay mean) |
| KAPPA | 0.25 | experiments.py:33 | Yes |
| base_seed | 42+... (4041) | experiments.py:132 | DO NOT CHANGE (reproducibility) |

## Metric Parsing

The evaluation command prints to stdout:
```
  saved -> results/post-serving/<label>.{pdf,csv}
    1. RCDP-UCB (Ours)      9375.1
    2. ColSTIM+PS          17266.1
    ...
```

Cumulative Regret for RCDP-UCB = the mean regret at T=2000 across n_runs.
Also available in CSV: results/post-serving/<label>.csv, filter method="RCDP-UCB (Ours)" and round=2000.

## Safe Modification Targets

1. `NeuralApproximator.__init__()` — architecture (hidden layers, activation)
2. `DuelingGLMLearner.__init__()` — optimizer, loss, LR, scheduler setup
3. `DuelingGLMLearner.train_approximator()` — training loop (batch size, epochs)
4. `DuelingGLMLearner.select_arms()` — UCB alpha computation
5. `DuelingGLMLearner.observe_context()` — epochs passed to train_approximator
6. `_ours()` in experiments.py — constructor parameters
7. `FIXED_LAMBDA`, `KAPPA` in experiments.py — robustness parameters

## Risky Files (DO NOT MODIFY)

- `main.py` — evaluation protocol
- `ContextualDuelingBanditEnv` — environment generation (would change evaluation)
- `run_simulation()` — simulation loop (would change metric computation)
- `_save()` / `run_one()` — metric aggregation and output
- Score parsing logic — would change how we report results
