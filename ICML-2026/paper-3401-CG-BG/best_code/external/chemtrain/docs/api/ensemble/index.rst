:modulename:`ensemble`
==================================

The ``ensemble`` module provides tools for sampling from molecular ensembles.
The :mod:`chemtrain.ensemble.sampling` module implements routines to sample
via MD simulations.
The :mod:`chemtrain.ensemble.reweighting` module implements thermodynamic
potential theory to transfer information between perturbed and unperturbed
ensembles.

.. toctree::
   :titlesonly:

   evaluation
   reweighting
   sampling


Utilities
-----------

.. autoclass:: chemtrain.ensemble.templates.StatePoint
