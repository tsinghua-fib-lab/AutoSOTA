"""Black-hole imaging experiment (paper §6.2).

Calibrated Bayesian Guidance on the InverseBench black-hole imaging task, in two
flavours (Table 2):

* ``--posterior meanflow`` — fast one-step mean-flow "renoise" sampler for
  ``p(x0|xt)`` (the new version).
* ``--posterior ddpm``     — slow DDPM inner-loop sampler (the original version).

The heavy engine (forward operator, scheduler, diffusion sampler, dataset,
evaluator, diffusion prior) is the vendored InverseBench under
``third_party/InverseBench``; this package adds the clean, uniform entry point.
"""
