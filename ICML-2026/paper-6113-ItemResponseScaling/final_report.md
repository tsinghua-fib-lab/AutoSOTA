# Final Report: paper-6113

- Title: Item Response Scaling Laws: A Measurement Theory Approach for Efficient and Generalizable Neural Scaling Estimation
- Primary metric: `RMSE` (lower)
- Records: 13
- Generated: 2026-07-14T21:56:09Z

## Best Result

- Iteration: 11
- Idea: ALGO-04-noclamp — Fisher weights no-clamp + phi=300
- Primary metric: 0.04434
- Commit: `967a6fca1bfda90390779b5746935cd20745a795`
- Notes: Fisher weights WITHOUT clamping + phi=300. RMSE=0.04434 (baseline 0.04842, -8.4%). Pearson rho=0.99887. Removing the [0.1, 10.0] weight clamp improves RMSE by ~0.2% vs clamped version. Clamp sweep confirmed no-clamp is best. phi=300 confirmed optimal without clamping.
