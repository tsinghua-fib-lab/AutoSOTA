# S2F search history summary

The supplied report describes **seven rounds and 89 iterations** from August 1
to September 8, 2026. This appendix preserves that round-level record. Individual
iteration logs and the original idea list were not supplied, so this is not a
complete iteration-by-iteration log. See the [research report](research_report.md)
for method details, full-set results, and development-data exposure.

| Round | Iterations | Focus | Intermediate dev-40 result |
|---|---:|---|---:|
| R1 | 26 | GVP architecture, geometry features, and training hyperparameters; 20-epoch screening | 0.4239 → 0.4318; reproduction: 0.4265 |
| R2 | 21 | 100-epoch training, losses, regularization, and longer pretraining | 0.4446 → 0.4446 |
| R3 | 6 | Restore sequence-branch logits in inference and combine them with structure scores | 0.4450 → 0.4468 |
| R4 | 10 | Unmasked wild-type marginal, long-protein windows, and training-seed checks | 0.4468 → 0.4500 |
| R5 | 10 | Disjoint sel-80 selection set and cached fusion-weight search | 0.4500 → 0.4525 |
| R6 | 14 | Henikoff-weighted PSSM, Neff-adaptive gating, and five-model ESM-1v scoring | 0.4525 → 0.4653 |
| R7 | 2 | Final reproduction and original-checkpoint full-204 comparison | Final: 0.4653 |
| **Total** | **89** | Approximately 47 training iterations, 30 inference iterations, and remaining infrastructure/validation work | — |

Researchers specified the directions, mechanism hypotheses, and acceptance
constraints. AutoSOTA implemented and searched candidates and ran evaluations.
Some rounds reset their own baseline, so their gains must not be compounded.

The final selection used ProteinGym subset labels, including sel-80; full-204
results were also observed during development. An apparent gain on a small
selection set did not always transfer to dev-40. Training-seed variation and
negative ablations are included in the report.

R7 ended after key measurements when API funds were exhausted. A separate run on
September 24–25, 2026 subsequently completed all 217 assays and 2,465,767 mutants.
Its function/UniProt-balanced Spearman was 0.46633717, compared with published
original S2F's 0.45560865. These grouped scores are distinct from the earlier
full-204 arithmetic means.
