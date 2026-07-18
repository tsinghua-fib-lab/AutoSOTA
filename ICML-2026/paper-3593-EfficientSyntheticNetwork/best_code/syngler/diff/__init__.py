"""SyNG-D — diffusion-based generation of LSM latents.

Two backends:
  - syngler.diff.forest : ForestDiffusion (paper default; CPU-only, fast).
  - syngler.diff.mlp    : DDPM with residual MLP (GPU; SyNG-D(MLP) variant).
"""
from syngler.diff import forest, mlp

__all__ = ["forest", "mlp"]
