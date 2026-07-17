# Final Report: paper-3454

- Title: CORRECT: COndensed eRror RECognition via knowledge Transfer in multi-agent systems
- Primary metric: `Acc@0` (higher)
- Records: 12
- Generated: 2026-07-16T23:06:20Z

## Best Result

- Iteration: 3
- Idea: IDEA-09-k5 — k=5 schemata (fewer schemata reduce noise)
- Primary metric: 12.07
- Commit: `cb8fafd37a4b2584966dcbcfa10e1fe69e56b9da`
- Notes: Reduced num_schemata from 10 to 5. Acc@0 improved from 10.34 to 12.07 (+1.73%). Acc@1 improved from 17.24 to 20.69 (+3.45%). Fewer schemata reduce noise — model focuses on most relevant patterns. Essentially matches paper reported 12.1% CORRECT result. Key insight: Qwen-2.5-7B performs better with fewer, higher-quality schemata.
