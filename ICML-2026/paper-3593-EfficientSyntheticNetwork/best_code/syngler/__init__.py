"""SyNGLER: Efficient Synthetic Network Generation via Latent Embedding Reconstruction.

Top-level subpackages:
  - syngler.lsm        : Latent Space Model fitting (PGD on Theta_ij = alpha_i + alpha_j + Z_i^T Z_j).
  - syngler.res        : SyNG-R — bootstrap resampling of fitted LSM latents.
  - syngler.diff       : SyNG-D — diffusion (ForestDiffusion or MLP) over fitted LSM latents.
  - syngler.utils      : shared utilities.
  - syngler.evaluation : paper metrics (triangle density, GCC, degree, eigenvalues, orbits, MMD/energy).
"""
__version__ = "0.1.0"
