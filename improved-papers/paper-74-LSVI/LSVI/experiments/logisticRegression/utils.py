import jax
import jax.numpy as jnp
from particles import datasets


def get_dataset(flip=True, dataset="Pima"):
    if dataset == "Pima":
        dataset = datasets.Pima()
    elif dataset == "Sonar":
        dataset = datasets.Sonar()
    data = dataset.preprocess(dataset.raw_data, return_y=not flip)
    return data


def logistic(x):
    # return 1 / (1 + jnp.exp(-x))
    return jax.scipy.special.expit(x)


def normal_cdf(x):
    return 0.5 * (1 + jax.scipy.special.erf(x / jnp.sqrt(2)))


def get_tgt_log_density(flipped_predictors, my_prior_log_density, cdf=logistic):
    """
    Define the log target density of the posterior distribution of the logistic regression model,
    assuming a Gaussian prior.
    """
    if cdf == logistic:
        def tgt_log_density(beta):
            logcdf = - jnp.log1p(jnp.exp(-flipped_predictors @ beta.T))
            logcdf = jnp.nan_to_num(logcdf, False, nan=0.0, posinf=0.0, neginf=0.0)
            log_likelihood = jnp.sum(logcdf, axis=-1)
            return log_likelihood + my_prior_log_density(beta)
    else:
        def tgt_log_density(beta):
            logcdf = jnp.log(cdf(flipped_predictors @ beta.T))
            logcdf = jnp.nan_to_num(logcdf, False, nan=0.0, posinf=0.0, neginf=0.0)
            log_likelihood = jnp.sum(logcdf, axis=-1)
            return log_likelihood + my_prior_log_density(beta)

    return tgt_log_density


def multilogistic_get_tgt_log_density(predictors, labels, my_prior_log_density):
    def tgt_log_density(beta):
        beta_reshaped = beta.reshape((predictors.shape[1], labels.shape[1]))
        C = jnp.trace(predictors @ beta_reshaped @ labels.T)
        sumlogsoftmax = jnp.sum(jax.scipy.special.logsumexp(- predictors @ beta_reshaped, axis=-1), axis=-1)
        loglikelihood = C + sumlogsoftmax
        return loglikelihood + my_prior_log_density(beta)

    return tgt_log_density
