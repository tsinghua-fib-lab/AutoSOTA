"""Repo path bootstrap.

Importing this module makes first-party packages in the repo importable,
regardless of which directory a script or notebook is launched from. It prepends
the repo root and its package directories to ``sys.path``:

- ``<root>``                  -> ``hide_and_seek.*``
- ``<root>/utils``            -> bare ``tools``, ``params``, ``loading_runs``, ``Data_Generation``
- ``<root>/packages``         -> ``INVASE_master.*``, ``realx_main.*``, ``L2X.*``
- ``<root>/packages/lime-master`` -> ``lime``
- ``<root>/experiments``      -> ``tests_mnist.classifier_for_lime``

Usage (from any script or notebook, after putting the repo root on sys.path):

    import os, sys
    sys.path.insert(0, os.path.abspath('../..'))   # reach the repo root once
    import repo_paths  # noqa: F401  -- adds the rest
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

_PATHS = (
    _ROOT,
    os.path.join(_ROOT, "utils"),
    os.path.join(_ROOT, "packages"),
    os.path.join(_ROOT, "packages", "lime-master"),
    os.path.join(_ROOT, "experiments"),
)

for _p in _PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)
