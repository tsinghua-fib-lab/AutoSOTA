# Code Analysis: CLASP Online Linear Regression (Paper 6078)

## Preparation Failure

The SOTA preparation failed because:
1. `git` was not installed in the `autosota/paper-6078:reproduced` Docker image
2. `apt-get install git` failed due to broken proxy connectivity (172.17.0.1:17890 returning 502)
3. The `.git` directory existed from the repo clone, but there was no `git` binary

**Fix**: Copied host `/usr/bin/git` (v2.25.1) into the container at `/usr/bin/git`. Both host and container are Ubuntu 20.04 x86_64 with compatible libraries.

## Corrected Evaluation Command

```bash
cd /repo && python3 evaluate.py
```

Runs from inside container `autosota_sota_paper_6078`. No external data needed — generates synthetic data on the fly. Takes ~9-10 minutes for 50 trials. Pure CPU — no GPU required.

## Baseline Verification

The repaired baseline evaluation produced CLASP-I metrics consistent with the manifest:
- cumulative_loss: 664.23 (manifest: 658.03, within stochastic noise)
- CCVT_1: 52.23 (manifest: 50.99, within stochastic noise)
- CCVT_2: 84.73 (manifest: 82.95, within stochastic noise)

The experiment is inherently stochastic (no seed control, random data generation each trial), so exact reproduction is impossible. The qualitative ranking matches the paper.

## Code Structure

### Algorithms (all in evaluate.py):
- **CLASP-I**: Step-size ηt = 1/√(t+1), projects using CVXPY QP per step, most-violated-constraint selection
- **CLASP-F**: Same ηt, simpler box projection + constraint gradient step, most-violated-constraint
- **AdaGrad**: Adaptive step-size with cumulative gradient norms, exponential penalty
- **RECOO**: CVXPY optimization per step with adaptive alpha/eta
- **Switch**: Two-phase: Phase 1 collects violated constraints via CVXPY projection; Phase 2 uses AdaGrad-style updates
- **FW (Frank-Wolfe)**: Block-based FW with exponential penalty

### Key Parameters:
- `n=10, k=4, T=100`: Problem dimensions
- `S=50`: Number of trials
- `D=√2, F=n²=100, G=√(4n)≈6.32`: Assumption parameters
- Step-size: `ηt = 1/√(t+1)` (hardcoded in CLASP-I and CLASP-F)

### Optimization Targets:
1. **Step-size schedule**: Tune coefficient c in ηt = c/√(t+1), or exponent p in ηt = 1/(t+1)^p
2. **CVXPY solver**: Add warm_start=True, try different solvers
3. **Constraint selection**: Project w.r.t. all violated constraints, not just the worst one
4. **Initial point**: Use fixed center point x₀=0.5 instead of random
5. **Algorithm parameters**: Tune AdaGrad/Switch α, β, λ
6. **Combined strategies**: Best individual improvements combined
