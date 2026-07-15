# Code Analysis — Paper 5638 (DisjunctiveNet) SOTA Optimization

## Evaluation Path
- **Script**: `/repo/experiments/pbmc3k_dnf.jl`
- **Eval command**: `julia --project=. experiments/pbmc3k_dnf.jl`
- **Metrics parsed from stdout**: Lines matching `DNF accuracy: ...`, `DNF macro_f1: ...`, `DNF macro_prec: ...`, `DNF macro_rec: ...`, `DNF CSAT: ...`
- **Aggregation**: Mean ± std over N_RUNS=3 random seeds, then printed with `round(μ, digits=4)`
- **Eval data**: Fixed test split (10% of PBMC3k, seed=42). Training uses 0.005 fraction (~12 cells).

## Training Path
- Base MLP training: 500 epochs, AdamW(lr=3e-3), batch_size=1 (sample-by-sample SGD)
- DNF projection: Applied at inference time only (NOT during training)
- No gradient flows through the projection layer in the current script

## Inference Path
- Per-sample: `get_active_rules()` → `project_sample()` → `project()`
- Projection rebuilds JuMP model from scratch for each sample
- Active rules determined by gene expression threshold comparison

## Config Path
- All config is inline in `pbmc3k_dnf.jl` (const declarations at top)
- Key params: HIDDEN_DIM=8, BASE_EPOCHS=500, LEARNING_RATE=3e-3, RHO=0.3
- Regularization: Y_REG=1e-4, YCOPY_REG=1e-4, GAMMA_REG=1e-4, ANCHOR_REG=1e-3

## Metric Parser
- `compute_metrics()`: argmax-based classification, macro-averaged precision/recall/F1
- `compute_csat()`: per-sample feasibility check (LP solve) + rule satisfaction check
- Metrics are deterministic given fixed seeds

## Safe Modification Targets
1. **HIDDEN_DIM**: Change from 8 to 16/32/64 (ALGO-003)
2. **LEARNING_RATE**: Tune or use schedule (ALGO-007)
3. **RHO**: Sweep [0.1, 0.2, 0.3, 0.4, 0.5] (PARAM-001)
4. **Training loop**: Add penalty term, fine-tuning phase (ALGO-002, ALGO-005)
5. **Gene thresholds**: Change fixed thresholds to percentile-based (CODE-004)
6. **Marker rules**: Add multi-gene conjunctions (ALGO-004)
7. **Training epochs**: Increase or add second phase

## Risky Files (do NOT modify)
- `/repo/src/backend/differentiation.jl` — ChainRules definitions (already patched)
- `/repo/src/modeling/` — Model building internals
- `/repo/src/formulations/` — Convex hull formulations
- `/repo/test/` — Test files
- `/datasets/pbmc3k/` — Fixed evaluation data

## Key Observation: Gradient Flow Exists
The `flux_end2end.jl` example demonstrates that `Flux.withgradient(model)` works through
`ConstrainedFluxModel` with DNF projection layer. The ChainRules `rrule` in
`differentiation.jl` defines backward pass via DiffOpt reverse differentiation.

**However**, the current experiment builds per-sample JuMP models with different constraints,
which cannot use a single `constrained_model()` wrapper. The per-sample rule activation
requires a different architecture for gradient-based training.

## Best Strategy
1. Start with safe config changes (HIDDEN_DIM, LR schedule)
2. Add penalty-based constraint-aware training (avoids AD issues)
3. Sweep ρ with best architecture
4. Improve biological rules if time permits
