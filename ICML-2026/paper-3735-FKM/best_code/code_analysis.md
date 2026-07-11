# Code Analysis for Paper 3735 SOTA Optimization

## Evaluation Path
- Eval command: python3 -u /repo/eval_grid_search.py
- Output: stdout + /repo/eval_result.json
- Timing: CUDA events

## Computation Flow
1. Generate synthetic training data: N=1e8, d=5, s=2
2. NUFFT cov_y: d=5 1D type-1 transforms (N=1e8 -> M=17)
3. NUFFT cov_x: d*(d+1)/2=15 2D type-1 transforms (N=1e8 -> 17x17)
4. Grid search: 300 lambda values, each with CG solve + validation
5. Test evaluation on best lambda

## Bottleneck
- NUFFT transforms dominate (~85%+ of runtime)
- 20 total transforms all using complex128
- complex64 gives ~5x NUFFT speedup

## Safe Modification Targets
1. NUFFT precision: complex128 -> complex64
2. NUFFT eps: Relax threshold from ~6e-8 to 1e-5
3. Fourier truncation m: Try m=7 or m=6
4. CG tolerance: Relax from 6e-8 to 1e-5
