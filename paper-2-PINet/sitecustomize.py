"""Patch cvxpylayers to work with JAX >= 0.6.

Temporary shim for packages that still import jax.core.Primitive on JAX >= 0.6.
NOTE: We can remove this once cvxpylayers has been released with the patch.
"""

import sys
import types

try:
    import jax

    # Only do this if jax.extend exists (JAX >= 0.6)
    if hasattr(jax, "extend"):
        # Build a fake module "jax.core" that mirrors jax.extend.core
        shim = types.ModuleType("jax.core")
        shim.__dict__.update(jax.extend.core.__dict__)
        sys.modules["jax.core"] = shim
        setattr(jax, "core", shim)
except Exception:
    pass
