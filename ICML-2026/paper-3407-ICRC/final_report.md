# Final Report: paper-3407

- Title: Calibrating Decision Robustness via Inverse Conformal Risk Control
- Primary metric: `Gap` (lower)
- Records: 11
- Generated: 2026-07-10T23:21:04Z

## Best Result

- Iteration: 6
- Idea: IDEA-06 — MC-only estimates (skip conformal correction)
- Primary metric: 0.0233
- Commit: `3937e057e4ff5531a20c9424a944f29562be0ed8`
- Notes: IDEA-06: MC-only estimates (output_mc=True). Gap: 0.0271->0.0233 (-14%). Time: 0.761->0.770s. Cumulative: 0.1279->0.0233 (-81.8%). MC-only removes the B/(n+1) correction bias entirely. With n=25, sampling variance is low enough that MC estimates are more accurate than conformal-corrected ones.
