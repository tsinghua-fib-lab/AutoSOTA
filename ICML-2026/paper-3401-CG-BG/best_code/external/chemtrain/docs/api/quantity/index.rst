``quantity``
======================

Atomistic
---------

**chemtrain** provides machine learning routines for predicting atomistic
advanced atomistic properties. For example, the :mod:`property_prediction`
module transforms energy models to predict per-atom and per-molecule properties.

.. toctree::
   :titlesonly:

   property_prediction

Macroscopic
------------

Ensemble theory connects the microscopic dynamics of a system to its macroscopic
properties via observables.
To enable perturbation-based training of these properties via DiffTRe,
**chemtrain** provides simple and advanced observables based on weighted
ensemble averages.
:class:`util.TargetBuilder` provides a simple interface to initialize the
observables and compute functions simultaneously.

.. toctree::
   :titlesonly:

   targets
   observables

Utilities
---------

.. toctree::
   :titlesonly:

   constants
