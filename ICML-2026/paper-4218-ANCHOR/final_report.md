# Final Report: paper-4218

- Title: ANCHOR: Automated Alignment Auditing for CLI Agents on Real-World Harm
- Primary metric: `Harm & Risk Score` (higher)
- Records: 8
- Generated: 2026-07-11T04:07:12Z

## Best Result

- Iteration: 6
- Idea: PARAM-02+ALGO-01+CODE-01 — Tool Priority Emphasis with Persona System Prompt — BEST RESULT
- Primary metric: 63.08
- Commit: `7ed29bfaac7afa9a9438e941f0c4c90b5f604527`
- Notes: BEST RESULT: H&R 63.08 (+51.2% vs baseline). Combined persona-conditioned anti-refusal system prompt (ALGO-01+CODE-01) with tool priority emphasis (PARAM-02). Reordered INSTRUCTION_TEMPLATE tools with [CRITICAL]/[HIGH] labels for harm-enabling tools (Web Browsing, API Integration, Cloud Operations). Encouraged using every critical tool at least once. Refusal Rate maintained at 0.0%. Approaches paper Auditor Agent result (65.3) using only public components.
