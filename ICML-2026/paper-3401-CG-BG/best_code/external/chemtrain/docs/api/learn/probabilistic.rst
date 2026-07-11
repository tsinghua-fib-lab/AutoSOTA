:modulename:`learn.probabilistic`
===========================================

.. currentmodule:: chemtrain.learn.probabilistic

.. automodule:: chemtrain.learn.probabilistic

Bayesian UQ
-------------------------------

Defines the Bayesian UQ problem by building prior and likelihood functions as well as
combining them to prepare the problem in order to initialize an SGMCMC force matching
trainer.

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   uniform_prior
   init_elementwise_prior_fn
   init_likelihood
   init_log_posterior_fn
   init_force_matching

Dropout Monte Carlo
-------------------------------
Utility functions to perform forward UQ of a model trained via Dropout.

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   init_dropout_uq_fwd
   dropout_uq_predictions

Propagate Uncertainty
-------------------------------

Propagate the uncertainty of probabilistic potentials to obtain UQ of MD observables.

.. autofunction:: uq_trajectories


UQ Postprocessing
-------------------------------

Utility functions to assess statistics of Markov chains and compute error metrics.

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   mcmc_statistics
   validation_mae_params_fm
   test_rmse_params_fm
