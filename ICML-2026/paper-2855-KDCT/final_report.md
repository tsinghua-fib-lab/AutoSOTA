# Final Report: paper-2855

- Title: Are Two Datasets Close Enough With Statistical Significance? A Kernel Distributional Closeness Testing Approach
- Primary metric: `NAMMD_Test_Power` (higher)
- Records: 9
- Generated: 2026-07-16T21:36:04Z

## Best Result

- Iteration: 5
- Idea: idea-original-N-12000 — Original script + N=12000 (no construction changes)
- Primary metric: 0.997
- Commit: `89f6aed57f087df84205fb13782de9d85278d7fa`
- Notes: Original reproduce_final.py with N=12000. NAMMD=0.997±0.002 (+8.4% vs baseline 0.920). MMD=0.981±0.004 (+16.5% vs baseline 0.842). Key finding: increased N alone beats optimized construction. Construction optimizations may produce unstable MMD results.
