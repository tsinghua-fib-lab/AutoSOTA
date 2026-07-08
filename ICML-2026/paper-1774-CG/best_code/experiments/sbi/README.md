# Bayesian-inference benchmark (paper §6.1)

The paper evaluates CBG on the simulation-based-inference benchmark of
[Lueckmann et al. (2021)](https://arxiv.org/abs/2101.04653): analytic prior +
likelihood, so the true posterior is available and methods are scored by **C2ST**
against reference samples (Table 1, Figure 3). CBG's gradient-free estimator
improves monotonically with compute and achieves the best distributional fit.

## Status

The full benchmark (the 5 sbibm tasks + the likelihood-based/likelihood-free
baselines) is added separately. This module ships a clean, self-contained
**analytic demo** that runs the exact estimator + flow-matching sampler used in
the paper on a closed-form Gaussian task, where the true posterior is known
exactly. It demonstrates calibration and is the template the full benchmark
plugs into.

## Run

```bash
python run.py sbi --estimator reinforce --num-particles 128
python run.py sbi --estimator reparam   --num-particles 64

# calibration trend: error shrinks as K grows
for k in 4 16 64 256; do python run.py sbi --num-particles $k; done
```

It reports MMD to the analytic posterior and the posterior mean/std errors, and
logs them to the unified W&B project (`calibrated-guidance`, group `sbi`). CPU is
fine. No checkpoints or data downloads needed.
