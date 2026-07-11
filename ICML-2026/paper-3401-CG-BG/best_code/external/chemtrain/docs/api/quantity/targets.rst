:modulename:`quantity.targets`
========================================

This module creates DiffTRe targets, initializing the required observables
and the computations of necessary snapshot quantities.

.. currentmodule:: chemtrain.quantity.targets


Intramolecular
--------------

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   init_dihedral_distribution_target


Intermolecular
--------------

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   init_radial_distribution_target
   init_angular_distribution_target


Thermodynamic
--------------

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   init_relative_entropy_target


Build Targets and Compute Functions
------------------------------------

.. autoclass:: TargetBuilder

.. autoclass:: InitArguments
   :members:

.. autofunction:: split_target_dict
