# Final Report: paper-5638

- Title: DisjunctiveNet: Neural Symbolic Learning via Differentiable Convexified Optimization Layers
- Primary metric: `Accuracy` (higher)
- Records: 7
- Generated: 2026-07-14T06:15:55Z

## Best Result

- Iteration: 3
- Idea: PARAM-001 — RHO=0.5 with multi-gene rules and HIDDEN_DIM=32
- Primary metric: 0.75
- Commit: `da1776b535192195b65a2c871b794967a88a0658`
- Notes: PARAM-001: RHO=0.5 with multi-gene rules and HIDDEN_DIM=32. Accuracy 0.75 (+72.8% over baseline 0.434), Macro F1 0.7056 (+47.3%). RHO sweep revealed that higher rho (>0.3) dramatically improves metrics because stronger biological constraints force more confident correct predictions. CSAT maintained at 1.0. All metrics improved over iter-2 (rho=0.3): Acc +5.5%, F1 +14.2%, Rec +20.6%.
