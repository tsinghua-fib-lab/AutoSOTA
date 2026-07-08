"""Unified entry points for the experiments in *Calibrated Test-Time Guidance
for Bayesian Inference*.

Three experiment lines live here, each exposing a ``cli()`` entry point and a
``run(...)`` function with a uniform shape:

* ``experiments.sbi``               — Bayesian-inference benchmark (analytic priors).
* ``experiments.black_hole``        — black-hole imaging (InverseBench engine).
* ``experiments.super_resolution``  — 4x ImageNet super-resolution (pixel mean-flow).

All three share :mod:`experiments.common` for Weights & Biases logging, seeding
and CLI helpers, so that runs are consistent and comparable across experiments.
"""
