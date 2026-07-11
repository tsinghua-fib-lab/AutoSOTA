:modulename:`learn.max_likelihood`
============================================

.. currentmodule:: chemtrain.learn.max_likelihood

.. automodule:: chemtrain.learn.max_likelihood

Loss Functions
---------------

These functions are masked implementations of common loss functions.

.. autofunction:: mse_loss

.. autofunction:: mae_loss


Dataset Predictions
--------------------

Algorithms such as force matching requires evaluation of the loss function on
many samples instead of a single snapshot. Therefore, **chemtrain** provides
functions to efficiently parallelize these evaluations, using vectorization
and parallelization.

.. autofunction:: pmap_update_fn

.. autofunction:: shmap_update_fn

.. autofunction:: shmap_loss_fn

.. autofunction:: init_val_predictions

.. autofunction:: init_val_loss_fn
