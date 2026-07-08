"""4x ImageNet super-resolution experiment (paper §6.3a).

Reconstructs 256x256 images from 64x64 low-resolution observations on
ImageNet-val with a pixel mean-flow (pMF) prior and the gradient-free
Calibrated Bayesian Guidance estimator. Each denoising step draws K candidate
x0 reconstructions from the one-step pMF posterior, re-weights them by the
Gaussian super-resolution likelihood (Eq. 20), and never computes a likelihood
gradient.

The validated SR sampling core (forward operator / bicubic 4x downsampler,
likelihood, pMF adapter, single-image guided sampler) is vendored verbatim
under ``pixel_space_inverse_problems`` and made importable by ``_engine``; this
package adds the clean, flag-driven driver and the unified W&B logging.
"""
