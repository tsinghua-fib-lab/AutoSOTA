:modulename:`learn.force_matching`
============================================

.. currentmodule:: chemtrain.learn.force_matching

.. automodule:: chemtrain.learn.force_matching

Dataset
-------

Utility functions to process datasets of per-snapshot quantities.
These utilities check the dataset for consistent keys, differentiating between
inputs to the model (e.g., atomic positions) and targets (e.g., atomic forces).

.. autofunction:: AtomisticDataset

.. autofunction:: build_dataset

Model
-----

The input to the learning problems is always an ``energy_fn_template`` function.
To match forces and/or virials, the computation of these quantities from the
energy function must first be initialized, e.g., using the following
functions.

.. autofunction:: init_model


Loss
----

These are functions to initialize advanced loss functions, e.g., combining the
losses for force and energy predictions into a single loss value.

.. autofunction:: init_loss_fn
