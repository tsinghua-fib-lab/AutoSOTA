# Final Report: paper-369

- Title: Profiling the Irrational Agent: Cognitive Modeling of LLM Behaviors in Sequential Jailbreaks
- Primary metric: `IAR` (lower)
- Records: 3
- Generated: 2026-07-04T18:10:40Z

## Best Result

- Iteration: 2
- Idea: ALGO-02 — NEUTRAL system prompt for Regret scenario
- Primary metric: 0.6
- Commit: `b43d44c4eaf5bdf79bf1f1de99b553d3e8b5cf5a`
- Notes: Changed Regret scenario system_key from GAME to NEUTRAL. Also reverted GAME prompt to original. IAR improved from 0.90 to 0.60 but worse than ALGO-01 (0.00). NTF=4.2, Mean ASR=0.158. Demonstrates that removing game framing helps but is less effective than explicit ethical framing.
