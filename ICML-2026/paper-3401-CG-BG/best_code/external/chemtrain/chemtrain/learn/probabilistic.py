# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains methods for training and evaluation of uncertainty-aware
 neural network potentials trained bottom-up via energy / force matching."""
from functools import partial

from jax import (lax, vmap, pmap, checkpoint, random, device_count,
                 scipy as jscipy, numpy as jnp, jit)
from jax_sgmc import data, potential
from scipy import stats
import tree_math

import chemtrain.data.data_loaders
from chemtrain import (util)
from chemtrain.data import preprocessing
from chemtrain.learn import force_matching, max_likelihood
from jax_md_mod.model import dropout


# Modeling


def uniform_prior(sample):
    """Uniform improper prior function.

    Args:
        sample: Dummy argument

    Returns:
        A uniform logpdf value of 0.
    """
    del sample
    return 0.


def init_elementwise_prior_fn(scale,
                              distribution=jscipy.stats.norm.logpdf,
                              loc=0.):
    """Initializes a prior distribution that acts on all parameters
    independently.

    Args:
        scale: Scale of the prior distribution.
        distribution: Prior logpdf, defaults to Gaussian.
        loc: Location of prior mean. Usually default of 0.

    Returns:
        Prior function
    """
    def prior_fn(params):
        params = tree_math.Vector(params)
        log_pdf = tree_math.unwrap(distribution, vector_argnums=0)(params,
                                                                   loc=loc,
                                                                   scale=scale)
        return log_pdf.sum()
    return prior_fn


def init_likelihood(energy_fn_template,
                    nbrs_init,
                    energy_scale=1.,
                    force_scale=1.,
                    virial_scale=1.,
                    virial_fn=None,
                    distribution=jscipy.stats.norm.logpdf):
    """Returns the likelihood function for Bayesian potential optimization
    based on a force-matching formulation.

    The scales of the likelihood components are normalized to be on same scale
    as energy_params.

    Args:
        energy_fn_template: Energy function template
        nbrs_init: Neighbor list to be used for initialization
        energy_scale: Prior scale of energy data.
        force_scale: Prior scale of force components.
        virial_scale: Prior scale of virial components.
        virial_fn: Virial function compatible ẃith virial data type,
            e.g. initialized via force_matching.init_virial_fn.
        distribution: Likelihood distribution. Defaults to a Gaussian logpdf,
            but any jax.scipy logpdf with the same signature can be provided.
    """
    single_prediction = force_matching.init_model(
        nbrs_init, energy_fn_template, virial_fn)

    def sum_log_likelihood(predictions, targets, std):
        likelihoods = distribution(predictions, loc=targets, scale=std)
        return jnp.sum(likelihoods)

    def likelihood_fn(sample, observation):
        prediction = single_prediction(sample['params'], observation)

        likelihood = 0.
        if 'U' in observation.keys():  # energy likelihood component
            likelihood += sum_log_likelihood(
                prediction['U'], observation['U'],
                sample['U_scale_multiple'] * energy_scale)
        if 'F' in observation.keys():  # forces likelihood component
            likelihood += sum_log_likelihood(
                prediction['F'], observation['F'],
                sample['F_scale_multiple'] * force_scale)
        if 'p' in observation.keys():  # virial likelihood component
            likelihood += sum_log_likelihood(
                prediction['p'], observation['p'],
                sample['p_scale_multiple'] * virial_scale)
        return likelihood
    return likelihood_fn


def _set_or_draw_exponential_rv(rv_scale, draw=True):
    """Returns a random variable, either drawn from the exponential distribution
     or set to its mean value. Also ensures the mean is defined.
     """
    assert rv_scale is not None
    if draw:
        rv = stats.expon.rvs(scale=rv_scale)
    else:
        rv = rv_scale  # exponential distribution mean is scale = 1 / lambda
    return jnp.array(rv, dtype=jnp.float32)


def init_force_matching(
        energy_param_prior, energy_fn_template, nbrs_init, init_params,
        position_data, energy_data=None, energy_scale=None, force_data=None,
        force_scale=None, virial_data=None, virial_scale=None, kt_data=None,
        box_tensor=None, train_ratio=0.7, val_ratio=0.1,
        likelihood_distribution=jscipy.stats.norm.logpdf,
        prior_scale_distribution=jscipy.stats.expon.logpdf,
        prior_scale_init_multiple=1., shuffle=False):
    """Initializes a compatible set of prior, likelihood, initial MCMC samples
    as well as train and validation loaders  for learning probabilistic
    potentials via force-matching.

    Data scales are used for parametrization of the exponential prior
    distributions for the standard deviations of the (energy, force and/or
    virial) likelihood distributions.
    This allows accounting for the different scales of energies, forces and
    virial, similar to the loss weights in standard force matching.
    Additionally, the scales of the likelihood components are normalized to be
    on same scale as energy_params to facilitate learning.

    Note that scale = 1 / lambda for the common parametrization of the
    exponential  distribution via the rate parameter lambda.
    See the :class:`scipy.stats.expon` documentation for more details.

    Args:
        energy_param_prior: Prior function for , e.g. as generated from
            ``'init_elementwise_prior_fn'``.
        energy_fn_template: Energy function template
        nbrs_init: Initial neighbor list
        init_params: Initial energy params
        position_data: (N_snapshots x N_particles x dim) array of particle
            positions
        energy_data: (N_snapshots,) array of corresponding energy values,
            if applicable
        energy_scale: Prior scale of energy data.
        force_data: (N_snapshots x N_particles x dim) array of corresponding
            forces acting on particles, if applicable.
        force_scale: Prior scale of force components.
        virial_data: (N_snapshots,) or (N_snapshots, dim, dim) array of
            corresponding virial (tensor) values (without kinetic contribution),
            if applicable.
        virial_scale: Prior scale of virial components.
        kt_data: Temperature corresponding to each data point. For learning
            temperature-dependent (coarse-graned) models.
        box_tensor: Box tensor, only needed if virial_data used.
        train_ratio: Ratio of dataset to be used for training. The remaining
            data can be used for validation.
        val_ratio: Ratio of dataset to be used for validation. The remaining
            data will be used for testing.
        likelihood_distribution: Log-likelihood distribution, defaults to
            Gaussian log-likelihood.
        prior_scale_distribution: Log-prior distribution of the likelihood scale
            parameter. Defaults to exponential distribution.
        prior_scale_init_multiple: Initial value of prior scale multiple. Can be
            used to initialize prior scales larger or smaller than prior mean or
            to cunteract the smaller scale in the gamma distribution compared to
            the mean.
        shuffle: Whether to shuffle data before splitting into train-val-test.

    Returns:
        A tuple (prior_fn, likelihood_fn, init_samples, train_loader,
        val_loader, test_loader, test_set). Prior and likelihood can be used to
        construct the potential function for Hamiltonian MCMC formulations.
        Init_samples is a list of initial values for multiple MCMC chains,
        e.g. for SGMCMCTrainer. The data loaders are jax-sgmc NumpyDataloaders
        used for training, validation and testing. The test_set can be used for
        further analyses of the trained model on unseen data.
    """
    dataset, _ = force_matching.build_dataset(position_data, energy_data,
                                              force_data, virial_data, kt_data)
    train_loader, val_loader, test_loader = chemtrain.data.data_loaders.init_dataloaders(
        dataset, train_ratio, val_ratio, shuffle=shuffle)

    virial_fn = force_matching.init_virial_fn(virial_data, energy_fn_template,
                                              box_tensor)

    likelihood_fn = init_likelihood(energy_fn_template, nbrs_init, energy_scale,
                                    force_scale, virial_scale, virial_fn,
                                    likelihood_distribution)

    def prior_fn(sample):
        prior = energy_param_prior(sample['params'])
        if 'U_scale_multiple' in sample:
            prior += prior_scale_distribution(
                sample['U_scale_multiple'] * energy_scale, scale=energy_scale)
        if 'F_scale_multiple' in sample:
            prior += prior_scale_distribution(
                sample['F_scale_multiple'] * force_scale, scale=force_scale)
        if 'p_scale_multiple' in sample:
            prior += prior_scale_distribution(
                sample['p_scale_multiple'] * virial_scale, scale=virial_scale)
        return prior

    if not isinstance(init_params, list):  # canonicalize init_params
        init_params = [init_params]

    init_samples = []
    for energy_params in init_params:
        sample = {'params': energy_params}
        scale_multiple = jnp.array(prior_scale_init_multiple, dtype=jnp.float32)
        if energy_data is not None:
            assert energy_scale is not None
            sample['U_scale_multiple'] = scale_multiple
        if force_data is not None:
            assert force_scale is not None
            sample['F_scale_multiple'] = scale_multiple
        if virial_data is not None:
            assert virial_scale is not None
            sample['p_scale_multiple'] = scale_multiple
        init_samples.append(sample)

    return (prior_fn, likelihood_fn, init_samples, train_loader, val_loader,
            test_loader)


def init_log_posterior_fn(likelihood, prior, train_loader, batch_size,
                          batch_cache):
    """Initializes the log-posterior function.

    Initializes a function that computes the log-posterior value of a parameter
    sample. The full log-posterior is computed batch-wise to avoid out-of-memory
    errors during the forward and backward pass.

    Args:
        likelihood: Likelihood function
        prior: Prior function
        train_loader: Train data loader
        batch_size: Batch-size for batch-wise computation of the sub-likelihoods
        batch_cache: Number of batches to store in cache

    Returns:
        Log-posterior function
    """
    # re-materialization of the likelihood for each batch of data allows
    # circumventing the enormous memory requirements of backpropagating
    # through the full potential - at the expense of additional
    # computational cost.
    likelihood = checkpoint(likelihood)  # avoid OOM for grad over whole dataset
    full_potential_fn = potential.full_potential(prior, likelihood,
                                                 strategy='vmap')
    data_map, _ = data.full_data_mapper(train_loader, batch_cache, batch_size)

    def log_posterior_fn(sample):
        potential_val, _ = full_potential_fn(sample, None, data_map)
        return -potential_val  # potential is negative posterior
    return log_posterior_fn


def init_dropout_uq_fwd(batched_model, meta_params, n_dropout_samples=8):
    """Initializes a function that predicts a distribution of predictions for
    different dropout configurations, e.g. for uncertainty quantification.

    Args:
        batched_model: A model with signature model(params, batch), which
            was trained using dropout.
        meta_params: Final trained meta_params
        n_dropout_samples: Number of predictions to run

    Returns:
        The function predict_distribution(key, model_input) predicts
        n_dropout_samples predictions for different dropout configurations to
        be used e.g. for uncertainty quantification.
    """
    n_devices = device_count()
    batch_per_device = int(n_dropout_samples / n_devices)
    util.assert_distributable(n_dropout_samples, n_devices, batch_per_device)
    haiku_params, _ = dropout.split_dropout_params(meta_params)

    def meta_param_model(key, model_input):
        dropout_params = dropout.build_dropout_params(haiku_params, key)
        return batched_model(dropout_params, model_input)

    def predict_distribution(key, model_input):
        keys = random.split(key, n_dropout_samples)
        keys = keys.reshape((n_devices, batch_per_device, 2))  # 2 per key
        key_batched_model = pmap(vmap(meta_param_model, (0, None)))
        model_input = util.tree_replicate(model_input)
        predictions = key_batched_model(keys, model_input)
        # reshape to move from multiple devices back to single device after pmap
        vectored_predictions = predictions.reshape((-1, *predictions.shape[2:]))
        # swap axes such that batch is along axis 0, which is needed for
        # full-data-map
        vectored_predictions = jnp.swapaxes(vectored_predictions, 0, 1)
        return vectored_predictions
    return predict_distribution


def dropout_uq_predictions(batched_model, meta_params, val_loader,
                           init_rng_key=random.PRNGKey(0),
                           n_dropout_samples=8, batch_size=1,
                           batch_cache=10, include_without_dropout=True):
    """Returns forward UQ predictions for a trained model on a validation
    dataset.

    Args:
        batched_model: A model with signature model(params, batch), which
             was trained using dropout.
        meta_params: Final trained meta_params
        val_loader: Validation data loader
        init_rng_key: Initial PRNGKey to use for sampling dropout configurations
        n_dropout_samples: Number of predictions with different dropout
            configurations for each data observation.
        batch_size: Number of input observations to vectorize. n_dropout_samples
            are already vmapped over.
        batch_cache: Number of input observations cached in GPU memory.
        include_without_dropout: Whether to also output prediction with Dropout
            disabled.

    Returns:
        A tuple (uncertainties, no_dropout_predictions) containing for each data
        observation n_dropout_samples dropout predictions as well as the mean
        prediction with dropout disabled. If include_without_dropout is False,
        only uncertainties are returned.
    """
    data_map, release = data.full_data_mapper(val_loader, batch_cache,
                                              batch_size)

    uq_fn = init_dropout_uq_fwd(batched_model, meta_params, n_dropout_samples)
    no_drop_params, _ = dropout.split_dropout_params(meta_params)

    def mapping_fn(batch, key):
        key, drop_key = random.split(key, 2)
        uq_samples = uq_fn(drop_key, batch)
        if include_without_dropout:
            prediction = batched_model(no_drop_params, batch)
            return (uq_samples, prediction), key
        else:
            return uq_samples, key

    # need to jit data mapper for correct dataloading under pmap
    mapper = jit(partial(data_map, mapping_fn))
    output, _ = mapper(init_rng_key)
    release()  # free all references to val_loader data generated by data_map
    return output


def uq_calibration(uq_samples, targets, mask=None):
    """Returns the scaling factor alpha, such that alpha * sigma is a
    UQ estimate that is calibrated on the validation data set.

    Args:
        uq_samples: A (n_samples, n_uq_estimates, *) array of UQ predictions
        targets: A (n_samples, *) array of target labels
        mask: : A boolean (n_samples, *) mask array such that padded predictions
            are treated correctly.
    """
    m = uq_samples.shape[1]
    predicted_mean = jnp.mean(uq_samples, axis=1)
    predicted_var = jnp.var(uq_samples, axis=1, ddof=1)
    squared_errors = (predicted_mean - targets)**2

    if mask is not None:
        predicted_var = predicted_var[mask]
        squared_errors = squared_errors[mask]

    variance_multples = squared_errors / predicted_var
    mean_error_multiple = jnp.mean(variance_multples)
    alpha_sq = -1. / m + (m - 3) / (m - 1) * mean_error_multiple
    return jnp.sqrt(alpha_sq)


def uq_trajectories(param_sets, init_state, trajectory_generator,
                    vmap_simulations=1, kt_schedule=None, n_dropout=16):
    """Compute multiple trajectories in parallel for evaluation of parameter
    uncertainty.

    Args:
        param_sets: Energy_params stacked along axis 0.
            For Dropout only a single parameter set.
        init_state: Either a single sim_state (compatible with the
            trajectory_generator) or sim_states stacked along axis 0 to
            start each energy_param set from a different sim_state.
        trajectory_generator: Trajectory generator as initialized from
            trajectory_generator_init.
        vmap_simulations: Number of simulations to run vectorized.
        kt_schedule: kbT schedule for simulations. If None, uses encoded
            temperature in simulator_template.
        n_dropout: Number of Dropout samples to evaluate.

    Returns:
        A trajectory state that contains all generated trajectories stacked
        along axis 0.
    """
    def single_trajectory(inputs):
        params, state = inputs
        traj = trajectory_generator(params, state, kT=kt_schedule)
        return traj

    batched_params = util.tree_vmap_split(param_sets, vmap_simulations)

    if util.tree_multiplicity(param_sets) != util.tree_multiplicity(init_state):
        batched_traj_fn = vmap(single_trajectory, in_axes=(0, None))
    else:  # same numer of init_states as param_sets
        batched_traj_fn = vmap(single_trajectory)
        init_state = util.tree_vmap_split(init_state, vmap_simulations)

    bachted_trajs = lax.map(batched_traj_fn, (batched_params, init_state))
    uq_trajs = util.tree_combine(bachted_trajs)
    return uq_trajs


def mcmc_statistics(uq_predictions):
    statistics = {}
    for quantity_key in uq_predictions:
        quantity_samples = uq_predictions[quantity_key]
        statistics[quantity_key] = {'mean': jnp.mean(quantity_samples, axis=0),
                                    'std': jnp.std(quantity_samples, axis=0)}
    return statistics


def _init_virial_if_applicable(val_loader, energy_fn_template, box_tensor):
    """Returns virial_fn according to the virial data type, otherwise None."""
    test_batch = val_loader.initializer_batch(1)
    if 'p' in test_batch:
        test_virial = test_batch['p']
    else:
        test_virial = None

    virial_fn = force_matching.init_virial_fn(test_virial, energy_fn_template,
                                              box_tensor)
    return virial_fn


def validation_mae_params_fm(params, val_loader, energy_fn_template, nbrs_init,
                             box_tensor=None, batch_size=1, batch_cache=1):
    """Evaluates the mean absolute error of a list of parameter sets generated
    via sampling-based methods, based on validation data and a force-matching
    likelihood.
    """

    virial_fn = _init_virial_if_applicable(val_loader, energy_fn_template,
                                           box_tensor)
    mae_fn, release = force_matching.init_mae_fn(
        val_loader, nbrs_init, energy_fn_template,
        batch_size, batch_cache, virial_fn
    )

    maes = []
    for i, param_set in enumerate(params):
        param_set = util.tree_replicate(param_set)
        mae = mae_fn(param_set)
        maes.append(mae)
        for key, mae_value in mae.items():
            print(f'Parameter set {i}: {key}: MAE = {mae_value:.4f}')

    release()
    return maes


def test_rmse_params_fm(param_set, test_loader, energy_fn_template, nbrs_init,
                        box_tensor=None):
    """Evaluates the root mean squared error of a set of parameters generated
    via sampling-based methods, based on test data and a force-matching
    objective.
    """
    test_batch = test_loader.initializer_batch(1)
    virial_fn = _init_virial_if_applicable(test_loader, energy_fn_template,
                                           box_tensor)
    model = force_matching.init_model(nbrs_init, energy_fn_template, virial_fn)
    param_batched_model = jit(vmap(model, in_axes=(0, None)))
    init_data_state, get_batch, release = data.random_reference_data(
        test_loader, 1, 1)
    data_state = init_data_state(shuffle=True, in_epochs=True)
    observation_count = test_loader.static_information['observation_count']

    mse_f = 0.
    mse_u = 0.
    mse_p = 0.

    for i in range(observation_count):
        data_state, batch = get_batch(data_state)
        batch = util.tree_get_single(batch)
        prediction_dict = param_batched_model(param_set, batch)

        # print(f'Target {batch["p"]}, prediction: {prediction_dict["p"]}')

        if 'U' in batch:
            mse_u += max_likelihood.mse_loss(
                jnp.mean(prediction_dict['U'], axis=0), batch['U'])
        if 'F' in batch:
            mse_f += max_likelihood.mse_loss(
                jnp.mean(prediction_dict['F'], axis=0), batch['F'])
        if 'p' in batch:
            mse_p += max_likelihood.mse_loss(
                jnp.mean(prediction_dict['p'], axis=0), batch['p'])

    rmse_dict = {}
    if 'U' in test_batch:
        rmse_dict['U'] = jnp.sqrt(mse_u / observation_count)
    if 'F' in test_batch:
        rmse_dict['F'] = jnp.sqrt(mse_f / observation_count)
    if 'p' in test_batch:
        rmse_dict['p'] = jnp.sqrt(mse_p / observation_count)

    release()
    return rmse_dict


