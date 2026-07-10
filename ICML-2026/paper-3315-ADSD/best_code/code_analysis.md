# Code Analysis — Paper 3315

## Overview
Sequential hypothesis testing for detecting strategic deviations in multi-agent systems.
Uses supermartingales (e-values) to test whether players deviate from Nash equilibrium strategies.

## Key Files

| File | Role | Safe to modify? |
|------|------|-----------------|
| `reproduce_fwer.py` | Main eval: measures FWER under H0 | Yes — modify supermartingale update, threshold logic |
| `normal-form.ipynb` | Reference: paper experiments with FWER+FDR+H1 | Read-only reference |
| `predator-prey.ipynb` | Reference: predator-prey game experiments | Read-only |
| `soccer.ipynb` | Reference: soccer game experiments | Read-only |
| `README.md` | Paper metadata | No |

## Evaluation Path

- **Command**: `python3 reproduce_fwer.py`
- **Metrics output**: `Empirical FWER: X.XXX` on stdout
- **Timeout**: 5 minutes (300s)
- **Current baseline**: Empirical FWER = 0.010 (3/300 rejections)

## Metric Parser

Parse from stdout:
- `Empirical FWER: X.XXX` — proportion of runs where any supermartingale ≥ threshold
- `Rejections: N / 300` — absolute rejection count

## Optimization Objective

**Constraint satisfaction**: Minimize avg detection time (H1) while keeping Empirical FWER ≤ α (nominal level).

### Guardrail: Empirical FWER
- Must remain ≤ α at all tested (λ, α) pairs
- Current: 0.010 at λ=0.05, α=0.2 (well within constraint)

### Core: Avg Detection Time
- Mean stopping time under H1 with various (η, λ, α) combinations
- Baseline: ~1236 rounds at η=0.05, α=0.2, λ=0.05

## Safe Modification Targets

1. **Supermartingale update** (line ~53-58 in reproduce_fwer.py): `M1[...] = M1[...] * (1 - lambda_val * X1)` — can change to mixture, log-space, truncated, or adaptive λ
2. **Rejection threshold**: `threshold = m / alpha` — can adjust threshold calibration
3. **Parameter grids**: λ ∈ {0.05, 0.1, 0.15, 0.4}, α ∈ {0.05, 0.1, 0.2}
4. **New evaluation scripts**: Can create new scripts for H1 timing, FDR evaluation

## Red-Line Boundaries

- Do NOT change payoff matrices U1, U2
- Do NOT change Nash equilibrium strategies pi_ne
- Do NOT change alternative strategies pi_alts
- Do NOT change the statistical validity of the test (supermartingale property must hold)
- Do NOT change test data, run counts, or evaluation protocol

## Known Levers

- **λ (betting parameter)**: 0.01–0.5. Higher λ → faster detection but higher FWER.
- **α (significance level)**: 0.01–0.5. Higher α → lower threshold = m/α, faster detection.
- **Acceptable region**: Empirical FWER ≤ α at all tested pairs.
