"""Compatibility shim for pickle deserialization of old models.

Old models were pickled with classes from baselines.posteriors.
This module re-exports from posteriors/ so those pickles can still load.
"""

from posteriors import *  # noqa: F401,F403
