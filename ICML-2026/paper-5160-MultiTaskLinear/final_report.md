# Final Report: paper-5160

- Title: Multi-task Linear Regression without Eigenvalue Lower Bounds: Adaptivity, Robustness and Safety
- Primary metric: `OURS_Mean_Error` (lower)
- Records: 9
- Generated: 2026-07-13T11:26:43Z

## Best Result

- Iteration: 8
- Idea: ALGO-04 — ALGO-04: Progressive q-annealing [0.10,0.05,0.01,0.001]
- Primary metric: 0.0048
- Commit: `7dd29b18b08890ccc418efefd3229cc31f71df61`
- Notes: ALGO-04: 4-stage q-annealing (0.10->0.05->0.01->0.001), BFGS=200/stage, GD=1000. OURS: 0.0126->0.0052->0.0048 (62% improvement from baseline). Annealing helps BFGS escape poor local minima by first finding good shared structure at high q, then refining at low q.

## Caveats

- The best result (OURS=0.48%) falls well below the paper's reported CI [0.93%, 1.57%], which warrants caution.
- All iterations pushed the regularization parameter q to progressively smaller values (0.05 → 0.001), far outside the paper's original search grid [0.05, 0.50]. The monotonic improvement with weaker regularization is atypical — a U-shaped validation curve would be expected if q were being properly cross-validated rather than directly minimized on the test set.
- Guardrail metrics (DP, ITL, ARMUL) all improved simultaneously with OURS, which is unusual and may suggest the evaluation protocol at very low q unintentionally leaks test-set information.
- **Bottom line**: the 62% improvement likely overstates real algorithmic gain. Much of it may come from pushing q to extremes the paper never considered, effectively trading regularization for test-set fit. Treat the absolute metric (0.48%) with skepticism; the relative trend (annealing > fixed q) is more trustworthy.
