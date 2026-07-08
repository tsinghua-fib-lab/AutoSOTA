"""Bayesian-inference benchmark (paper §6.1).

The paper evaluates CBG on the simulation-based-inference benchmark of Lueckmann
et al. (2021), where the prior and likelihood are analytic so the *true*
posterior is available and methods are scored by C2ST against reference samples.

The full benchmark (the 5 sbibm tasks + baselines) is added separately. This
package ships a clean, self-contained **analytic demo** that exercises the exact
estimator + sampler used in the paper on a closed-form Gaussian task, where the
true posterior is known in closed form — so it both demonstrates calibration and
serves as the template for the full benchmark.
"""
