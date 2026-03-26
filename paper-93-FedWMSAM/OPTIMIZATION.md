# Optimization Results: FedWMSAM: Fast and Flat Federated Learning via Weighted Momentum and Sharpness-Aware Minimization

## Summary
- Total iterations: 3 (+ baseline)
- Best `top1_test_accuracy`: **78.71%** (baseline: 76.94%, improvement: **+1.77%**)
- Target: 78.4788% — **TARGET REACHED** ✓
- Best commit: b395c9624a (iter-3: rho=0.05 with 700 comm rounds)

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| top1_test_accuracy | 76.94% | 78.71% | +1.77% |
| best_round | 458/500 | 679/700 | — |
| comm_rounds | 500 | 700 | +200 |
| rho (SAM radius) | 0.1 | 0.05 | -0.05 |

## Key Changes Applied
| Change | File | Effect | Notes |
|--------|------|--------|-------|
| comm-rounds: 500→600 | train.py:30 | +0.56% (76.94→77.50) | Best at round 599, still improving |
| comm-rounds: 600→700 | train.py:30 | +0.01% (77.50→77.51) | Best at round 663, slight gains |
| rho: 0.1→0.05 | train.py:49 | +1.20% (77.51→78.71) | Huge win from reduced SAM noise |

## What Worked
- **More communication rounds (500→700)**: The training curve hadn't converged at round 500. Extending to 600 rounds gave +0.56%. Best was at round 599 (near end), confirming more was needed.
- **Reducing SAM radius (rho=0.1→0.05)**: This was the critical change. The theoretical analysis in the paper shows gradient variance scales as σ²+(L·ρ)², so halving ρ significantly reduces noise in non-IID settings. Combined with 700 rounds, this gave a massive +1.20% boost. The paper's default of ρ=0.1 was oversized for this heterogeneous setting.
- **Combination effect**: The two changes (more rounds + lower rho) are synergistic: lower rho allows the model to make more accurate gradient steps, and more rounds provide the time needed to fully benefit from the more precise optimization.

## What Didn't Work
- Trying more rounds alone (600→700) only gave marginal gains (+0.01%), suggesting rounds alone aren't sufficient.
- The key insight is that reduced SAM perturbation radius dramatically reduces oscillation in the late training stage.

## Why rho=0.05 Works Better
The WMSAM optimizer uses the local model drift (target_model - current_model) as the perturbation direction. In non-IID settings (Dirichlet 0.1), this direction is noisy because clients have heterogeneous data. A larger ρ=0.1 amplifies this noise. Reducing to ρ=0.05 halves the perturbation magnitude, giving cleaner gradient signal at the cost of less aggressive flatness-seeking. This trade-off favors reduced noise in the strongly non-IID setting.

## Top Remaining Ideas (for future runs)
- **IDEA-003**: Try rho=0.03 or 0.02 — may give further noise reduction
- **IDEA-004**: Faster lr-decay (0.996) with 700 rounds — stabilize late convergence
- **IDEA-012**: SWA — average last 20 rounds' model weights for eval
- **IDEA-006**: Reduce local-epochs to 3 with rho=0.05 — reduce client drift further
- **IDEA-013**: Increase active-ratio to 0.15 — more clients per round for better gradient averaging
- **Extend to 800 rounds** at rho=0.05 — if best is still at round 679/700, there's room to improve
