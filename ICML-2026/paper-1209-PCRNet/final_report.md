# Final Report: paper-1209

- Title: PCRNet: Phase-aware Complex Refinement Network for EEG-based Auditory Attention Decoding
- Primary metric: `Accuracy` (higher)
- Records: 8
- Generated: 2026-07-07T06:42:48Z

## Best Result

- Iteration: 5
- Idea: ALGO-05 — Label smoothing eps=0.05 from iter-1 baseline
- Primary metric: 0.867
- Commit: `d9783aca736c4ad4ac5bf75323b6489d431a9969`
- Notes: Label smoothing epsilon=0.05 on CrossEntropyLoss from iter-1 baseline. Accuracy 86.70% vs iter-1 86.34% (+0.36pp). Hard subjects benefited most: S1 +3.3pp, S9 +4.1pp, S13 +6.3pp. Easy subjects preserved. Training longer (~75 epochs vs ~40) due to smoothed loss floor. Parsed from stdout.
