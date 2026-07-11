:modulename:`quantity.property_prediction`
====================================================

.. currentmodule:: chemtrain.quantity.property_prediction

.. automodule:: chemtrain.quantity.property_prediction


Molecular Property Predictors
-------------------------------

This wrapper function is at the core of the molecular property prediction
module by transforming models for (atom-wise) potential energy prediction to
molecular property predictors for atom and molecule level properties.

.. autofunction:: molecular_property_predictor


Properties
___________

**chemtrain** provides the following property predictors:

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   partial_charge_prediction
   potential_energy_prediction


Protocols
__________

.. autoclass:: PropertyPredictor
   :members: __call__

.. autoclass:: SinglePropertyPredictor
   :members: __call__


Examples
_________

We provide an example to transform DimeNet++ to a partial charge predictor,
which enforces charge neutrality of its predictions. For a real-world
application of this partial charge predictor in an active learning context,
see this `code <https://github.com/tummfm/mof-al>`_ of
`Thaler et al. (2024) <https://www.nature.com/articles/s41524-024-01277-8>`_.


Utilities
__________

.. autosummary:: apply_model


Snapshot Quantities
---------------------

Based on the predicted properties, we can compute other physical snapshot
quantities. For example, with the predicted partial charges, we can compute
the dipole moment of a system.

The following function transforms a property predictor on a graph into a
snapshot compute function:

.. autofunction:: snapshot_quantity

To extract features only once for all derived snapshot quantities, the
following function can be used in combination with
:func:`chemtrain.trajectory.traj_util.quantity_map`:

.. autofunction:: init_feature_pre_computation

**chemtrain** provides the following snapshot quantities:

.. autosummary::
    :toctree: _autosummary
    :template: function.rst

    init_dipole_moment
