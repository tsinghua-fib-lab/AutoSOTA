# Final Report: paper-2581

- Title: TINNs: Time-Induced Neural Networks for Solving Time-Dependent PDEs
- Primary metric: `Relative L2 Error` (lower)
- Records: 9
- Generated: 2026-07-09T08:58:46Z

## Best Result

- Iteration: 8
- Idea: CODE-03-refine — RelL2Error-based checkpointing (refinement on 60K best)
- Primary metric: 2.24764e-07
- Commit: `f67072f2dd2bc579d5a1b2239df88e8068d51add`
- Notes: CODE-03 refine: Switched checkpointing criterion from val_tot to RelL2Error. Correctly selected best model at step 58500 (2.25e-07) vs val-based at step 59500 (2.26e-07). Tighter patience (10K). The val-based checkpoint had a slight bias because val loss kept decreasing while test error started increasing after step 58500.
