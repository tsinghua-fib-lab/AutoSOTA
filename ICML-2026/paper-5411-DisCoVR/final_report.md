# Final Report: paper-5411

- Title: Variational Learning of Disentangled Representations
- Primary metric: `NLL` (lower)
- Records: 8
- Generated: 2026-07-13T04:53:54Z

## Best Result

- Iteration: 6
- Idea: IDEALIB-5411-4 — InfoNCE contrastive loss on z (lambda=0.05)
- Primary metric: 1.765
- Commit: `bfbfebfb4e64a169f84622d9d58bb216d5d3aa33`
- Notes: InfoNCE contrastive loss on z latent (lambda=0.05, tau=0.1) with cyclical KL and gradient clipping. NLL=1.765 (versus 1.766 from cyclical KL alone). Delta-Bayes=0.001 (versus 0.002). Both metrics improved — contrastive loss helps z become more label-invariant while preserving reconstruction quality. NEW BEST.
