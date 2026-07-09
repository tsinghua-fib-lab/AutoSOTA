# Final Report: paper-5267

- Title: PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA
- Primary metric: `mawps_accuracy` (higher)
- Records: 7
- Generated: 2026-07-08T21:08:29Z

## Best Result

- Iteration: 5
- Idea: COMBINED — Combined: seed fix + debias + noise decay
- Primary metric: 0.8193
- Commit: `1255bd859fd789898d92f509b2842c026236385b`
- Notes: Combined IDEA-08+09+02. BEST RESULT: Avg 0.5883 (+0.025 vs baseline). GSM8K 0.4450, AQuA 0.4488 (+0.051!), MAWPS 0.8193 (+0.017), SVAMP 0.6400 (+0.033). All metrics above baseline. Best combination of seed fix + dp_debias + noise decay.
