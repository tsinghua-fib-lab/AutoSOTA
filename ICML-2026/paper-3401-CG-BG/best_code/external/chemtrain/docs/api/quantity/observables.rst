:modulename:`quantity.observables`
============================================

.. currentmodule:: chemtrain.quantity.observables

.. automodule:: chemtrain.quantity.observables

This module contains methods to compute ensemble quantities given a trajectory
of instantaneous quantities.

Ensemble Averages
------------------

Ensemble averages of instantaneous quantities, optionally via a reweighting
approach.

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   init_traj_mean_fn
   init_linear_traj_mean_fn


Fluctuation Quantities
-----------------------

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   init_heat_capacity_nvt
   init_heat_capacity_npt
   init_born_stiffness_tensor
   stiffness_tensor_components_cubic_crystal
   stiffness_tensor_components_hexagonal_crystal

State-Space Quantities
-----------------------

Quantities which are no ensemble averages.

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   init_relative_entropy_traj_fn

Utility
--------

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   init_identity_fn
   dynamic_statepoint
