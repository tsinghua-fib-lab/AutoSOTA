# Final Report: paper-3345

- Title: Transporting Task Vectors across Different Architectures without Training
- Primary metric: `Accuracy` (higher)
- Records: 12
- Generated: 2026-07-10T11:12:30Z

## Best Result

- Iteration: 11
- Idea: PARAM-06 — num_batches=10 + whiten=0.4 (final config)
- Primary metric: 68.18
- Commit: `41226667dcd6e065d4b12b3af5520704252cd27c`
- Notes: Final best config: center_acts=True, whiten_power=0.4, num_batches=10, batch_size=32. Test accuracy: 68.18% (+10.51pp over baseline 57.67%). Total improvement from systematic optimization of Procrustes alignment parameters. Parsed from stdout: SVHN 0.376969 0.681762 1.808537
