"""Make the vendored super-resolution core importable.

The original pixel-space SR code is a flat package, ``pixel_space_inverse_problems``,
whose modules import each other by the absolute name
``from pixel_space_inverse_problems.X import Y`` and whose pMF adapter inserts
``<adapter_dir>/external/pMF`` on ``sys.path`` to resolve ``from pmf import
pixelMeanFlow``. We vendor it verbatim under
``experiments/super_resolution/pixel_space_inverse_problems`` and put *this
experiment directory* on ``sys.path`` here so ``import
pixel_space_inverse_problems`` resolves without editing any vendored file.

Import this module before importing any ``pixel_space_inverse_problems.*``
symbol. The shared ``calibrated_guidance`` library resolves separately because
the repo root is on ``PYTHONPATH`` (or the package is installed).
"""

from __future__ import annotations

import os
import sys

# Headless-safe matplotlib in case any vendored/lazy module imports pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

# This experiment directory; its child ``pixel_space_inverse_problems`` is the
# vendored core package. Putting THIS dir on sys.path makes the package's own
# absolute self-imports (``from pixel_space_inverse_problems.X import Y``) work.
SUPER_RESOLUTION_ROOT = os.path.abspath(os.path.dirname(__file__))
PIXEL_SPACE_ROOT = os.path.join(SUPER_RESOLUTION_ROOT, "pixel_space_inverse_problems")
# Where ``pixel_mean_flow_adapter`` expects the public pMF clone (download.sh).
PMF_REPO_ROOT = os.path.join(PIXEL_SPACE_ROOT, "external", "pMF")
DEFAULT_CHECKPOINT = os.path.join(SUPER_RESOLUTION_ROOT, "checkpoints", "pMF-B-16.pt")

if SUPER_RESOLUTION_ROOT not in sys.path:
    sys.path.insert(0, SUPER_RESOLUTION_ROOT)
