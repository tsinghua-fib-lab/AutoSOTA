# Final Report: paper-4989

- Title: Graph Alignment via Dual-Pass Spectral Encoding and Latent Space Communication
- Primary metric: `Hit@1` (higher)
- Records: 13
- Generated: 2026-07-13T02:54:04Z

## Best Result

- Iteration: 1
- Idea: I-09 — Float32 precision consistency in eval
- Primary metric: 97.58
- Commit: `df5518807aa5c62901317e0f85fef3995e18dfa4`
- Notes: Changed torch.cdist from float64 (.double()) to float32 during evaluation. Hit@1 improved from 94.10 to 97.58 (+3.48 pp). Guardrail metrics at ceiling.
