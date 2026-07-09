# Final Report: paper-2141

- Title: RealtimeTool: Parallel Decoding for Real-Time LLM Function Calling
- Primary metric: `Overall Accuracy` (higher)
- Records: 9
- Generated: 2026-07-08T05:11:25Z

## Best Result

- Iteration: 7
- Idea: BUGFIX-param-order — Fix parameter ordering mismatch in get_param_order + CODE-2
- Primary metric: 89.5
- Commit: `90f41e220cb3f13513d83a64e2f5e85c14e1ddb6`
- Notes: CRITICAL BUG FIX: get_param_order() used required-first+alpha ordering but model outputs in definition order. Fixed to use definition order. Corrected true baseline: 89.2%%. With CODE-2 on top: 89.5%%. Recovered 12 misaligned samples. Remaining: 54 wrong_value + 11 missing_required.
