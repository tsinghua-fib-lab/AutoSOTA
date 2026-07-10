# Final Report: paper-2585

- Title: Control Consistency Losses for Diffusion Bridges
- Primary metric: `kl_to_solution` (lower)
- Records: 10
- Generated: 2026-07-09T14:27:21Z

## Best Result

- Iteration: 9
- Idea: PARAM-01 — Extended Training (8000 iterations) with Cosine LR
- Primary metric: 0.002293
- Commit: `5c363a8d1d023d668c5c8b8979616ea2be1ae77e`
- Notes: Extended num_outer_iterations from 4000 to 8000 with cosine LR decay. kl_to_solution: 0.00386→0.00229 (-40% from iter4, -95.5% from baseline). Incredibly consistent: [0.00252, 0.00224, 0.00198, 0.00228, 0.00244], std=0.00019. kl_to_reference_learned=7.07 stays near true 7.02. Extended training with cosine LR enables full convergence — the model benefits from extended fine-tuning at low LR.
