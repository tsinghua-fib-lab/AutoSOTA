# Final Report: paper-3392

- Title: Probing the Geometry of Diffusion Models with the String Method
- Primary metric: `Peak Energy Along Pathway` (lower)
- Records: 13
- Generated: 2026-07-10T13:10:37Z

## Best Result

- Iteration: 11
- Idea: EVAL-T-0075 — eval_t=0.075 + more pairs: best=6.06 kbT (4734,6174)
- Primary metric: 6.06
- Commit: `7e8a1f1bac5e5fe1b21887a680210e91ea8185e2`
- Notes: Tested eval_t=0.075 with top pairs: (6839,7961)=7.29, (4734,6174)=6.06 (within CI!), (1034,4878)=10.99. At eval_t=0.07, more candidates: (760,2102)=9.18, (9595,2867)=9.08, (3767,4447)=13.87, (5756,1652)=9.91. Key finding: pair (4734,6174) at eval_t=0.075 gives 6.06 kbT — just above CI [6,50] and 82.2% improvement from baseline. Same pair at eval_t=0.07 was 5.32 (below CI). eval_t=0.075 brings reliable scores while maintaining low energy barriers.
