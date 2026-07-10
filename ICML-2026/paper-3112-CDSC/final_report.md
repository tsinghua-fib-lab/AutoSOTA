# Final Report: paper-3112

- Title: A Strictly Proper Scoring Rule and a Calibration Metric for Interval-Censored Data Analysis
- Primary metric: `IC-Cal` (lower)
- Records: 12
- Generated: 2026-07-09T18:55:06Z

## Best Result

- Iteration: 10
- Idea: ALGO-01b — Single model (no ensemble) with CDSC + Turnbull bins
- Primary metric: 0.007171
- Commit: `441901f18a5446f6714ebd31d545bbafdb2766eb`
- Notes: Removed ensemble (single model per fold). SIC-Log improved from 1.4008 to 1.3944 (-0.5%). IC-Cal improved from 0.007230 to 0.007171 (-0.8%). PARETO-DOMINANT over iter9 ensemble! CDSC provides sufficient regularization; ensemble averaging actually hurt slightly. Best result overall: SIC-Log -30.5% vs baseline, IC-Cal -13.2% vs baseline.
