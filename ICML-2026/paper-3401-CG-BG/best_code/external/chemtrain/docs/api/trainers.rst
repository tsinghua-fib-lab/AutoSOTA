:modulename:`trainers`
================================

.. automodule:: chemtrain.trainers

.. currentmodule:: chemtrain.trainers

The ``trainers`` module provides a high-level interface to train models via
**chemtrain**'s algorithms.
The :ref:`base_trainers` provide the core algorithms, while
:ref:`combining_trainers`, allows the construction of more
advanced training schemes from multiple algorithms.

.. _base_trainers:

Base Trainers
--------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   Difftre
   DifftreParallel
   PropertyPrediction
   ForceMatching
   RelativeEntropy
   SGMCForceMatching

.. _combining_trainers:

Combining Trainers
-------------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   InterleaveTrainers

Train Multiple Models
----------------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   EnsembleOfModels

Trainer Templates
------------------

.. currentmodule:: chemtrain.trainers.base

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   MLETrainerTemplate
   DataParallelTrainer
   EarlyStopping

Extensions
------------

**chemtrain** provides some extensions to the :ref:`base_trainers`,
e.g., to log data to services allowing live tracking of the training process.

Logging to `Weights and Biases <https://wandb.ai/site>`_
________________________________________________________

.. currentmodule:: chemtrain.trainers.extensions

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   wandb_log_difftre

