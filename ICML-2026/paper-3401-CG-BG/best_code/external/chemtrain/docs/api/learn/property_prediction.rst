:modulename:`quantity.property_prediction`
====================================================

.. currentmodule:: chemtrain.learn.property_prediction

.. automodule:: chemtrain.learn.property_prediction


Pre-processing
-------------------------------
Utility functions for initialization of the dataset and the model as well as
a masked loss function for masking virtual atoms that are added to conserve
array shapes.

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   build_dataset
   init_model
   init_loss_fn

Post-processing
-------------------------------
Property prediction at the atom level is often highly class-imbalanced.
To account for this imbalance, the following functions evaluate prediction
accuracy for each atom species:

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   per_species_results
   per_species_box_errors
