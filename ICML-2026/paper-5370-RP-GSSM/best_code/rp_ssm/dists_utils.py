from typing import TYPE_CHECKING, Optional, Union

import jax
import jax.numpy as np
import jax.random as jr
from jax import Array, vmap

from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as MVN
)

from dynamax.linear_gaussian_ssm import parallel_lgssm_smoother, LinearGaussianSSM

from rp_ssm import utils
from rp_ssm.types import *

if TYPE_CHECKING:
    from dists import LGChainDistParam


def transitions_to_marginals(
    params: dict[str, Array],
    num_timesteps: int,
    invariant: bool,
    num_samples: int = 0,
    key: Optional[Array] = None,
) -> Union["LGChainDistParam", tuple["LGChainDistParam", Array]]:
    """
    Convert transition distributions p(z_1), p(z_t+1|z_t)
    to marginals p(z_t).

    If invariant is True, we assume that p(z_t)=p(z_1)
    for all t.

    If params has keys 'As' and 'bs', assume that these
    are action-conditioned parameters A(a_t) and b(a_t)
    for all t.

    For now assume that Q is never action-dependent.

    If num_samples is 0, don't return any samples.
    Otherwise, sample chains z_1^n,...z_T^n for
    n=1,...,num_samples.
    """
    from rp_ssm.dists import LGChainDistParam

    if "Qs" in params:
        raise NotImplementedError

    if invariant:
        assert "As" and "bs" not in params
        means = np.tile(params["m1"][None], (num_timesteps, 1))
        covs = np.tile(params["Q1"][None], (num_timesteps, 1, 1))
        cross_covs = params["A"] @ covs[:-1]  # Cov(z_t+1,z_t) = A @ Sigma_t

    else:
        # if "As" is stationary (so has same number of dimensions as "A"), tile it
        if params.get("As", np.zeros_like(params["A"])).ndim == params["A"].ndim:
            As = np.tile(params["A"][None], (num_timesteps, 1, 1))
        else:
            As = np.concatenate([params["A"][None], params["As"]])
        # if "bs" is stationary (so has same number of dimensions as "b"), tile it
        if params.get("bs", np.zeros_like(params["b"])).ndim == params["b"].ndim:
            bs = np.concatenate(
                [params["m1"][None], np.tile(params["b"][None], (num_timesteps - 1, 1))]
            )
        else:
            bs = np.concatenate([params["m1"][None], params["bs"]])
        Qs = np.concatenate(
            [params["Q1"][None], np.tile(params["Q"][None], (num_timesteps - 1, 1, 1))]
        )

        def _step(carry, x):
            mean, cov = carry
            A, b, Q = x
            mean = A @ mean + b
            cov = A @ cov @ A.T + Q
            return (mean, cov), (mean, cov)

        means, covs = jax.lax.scan(
            _step,
            init=(
                np.zeros_like(params["m1"]),
                np.zeros_like(params["Q1"]),
            ),  # As[0] can be arbitrary
            xs=(As, bs, Qs),
        )[1]

        cross_covs = As[1:] @ covs[:-1]  # Cov(z_t+1,z_t) = A_t+1 @ Sigma_t

    chain_dist = LGChainDistParam(means=means, covs=covs, cross_covs=cross_covs)

    if num_samples > 0:
        assert key is not None

        def _sample_single_chain(key):
            time_keys = jr.split(key, num_timesteps)
            noises = vmap(lambda kt, bt, Qt: jr.multivariate_normal(kt, bt, Qt))(
                time_keys, bs, Qs
            )  # TxK

            def _sample_next_step(carry, x):
                zt = carry
                At, noise = x
                ztt = At @ zt + noise
                return ztt, ztt

            inputs = (As[1:], noises[1:])
            init = noises[0]
            _, zs = jax.lax.scan(_sample_next_step, init, inputs)

            return np.concatenate([init[None], zs], axis=0)

        seq_keys = jr.split(key, num_samples)
        samples = vmap(_sample_single_chain)(seq_keys)
        return chain_dist, samples

    else:
        return chain_dist


def chain_kl(q: dict[str, Array], p: dict[str, Array]) -> float:
    """Compute KL(q(z_1:T)||p(z_1:T))"""

    def kl_t(muq, mup, Sq, Sp):
        return MVN(muq, Sq).kl_divergence(MVN(mup, Sp))

    def kl_tt(muqt, muqtt, mupt, muptt, Sqt, Sqtt, Spt, Sptt, Sqx, Spx):

        muq = np.concatenate((muqt, muqtt))
        mup = np.concatenate((mupt, muptt))
        Sq = np.block([[Sqt, Sqx.T], [Sqx, Sqtt]])
        Sp = np.block([[Spt, Spx.T], [Spx, Sptt]])
        return kl_t(muq, mup, Sq, Sp)

    marginal = vmap(kl_t)(
        q["means"][1:-1],
        p["means"][1:-1],
        q["covs"][1:-1],
        p["covs"][1:-1]
    )
    pairwise = vmap(kl_tt)(
        q["means"][:-1],
        q["means"][1:],
        p["means"][:-1],
        p["means"][1:],
        q["covs"][:-1],
        q["covs"][1:],
        p["covs"][:-1],
        p["covs"][1:],
        q["cross_covs"],
        p["cross_covs"],
    )

    return np.sum(pairwise) - np.sum(marginal)


def parallel_smoother(inference_params, factors, latent_dim):

    if "As" in inference_params:
        dynamics_weights = inference_params["As"]
    else:
        dynamics_weights = inference_params["A"]
    if "bs" in inference_params:
        dynamics_bias = inference_params["bs"]
    else:
        dynamics_bias = inference_params["b"]
        
    lgssm = LinearGaussianSSM(latent_dim, latent_dim)    
    lgssm_params, _ = lgssm.initialize(
        initial_mean=inference_params["m1"],
        initial_covariance=inference_params["Q1"],
        dynamics_weights=dynamics_weights,
        dynamics_bias=dynamics_bias,
        dynamics_covariance=inference_params["Q"],
        emission_weights=np.eye(latent_dim),
        emission_covariance=factors.params["cov"]
    )

    smoother_out = parallel_lgssm_smoother(
        lgssm_params, factors.params["mean"]
    )._asdict()

    # in rare cases the smoothed covariances have min. evalues ~-1e-6,
    # so add a correction to be safe (checked that this has no 
    # effect on experiments that were already stable)
    smoother_out["smoothed_covariances"] += 1e-5 * np.eye(latent_dim)

    filtered_cov = smoother_out["filtered_covariances"]
    smoothed_cov = smoother_out["smoothed_covariances"]
    As, Q = dynamics_weights, inference_params["Q"]

    if As.ndim == 2:
        in_axes = (0, 0, None)
    elif As.ndim == 3:
        in_axes = (0, 0, 0)
    
    cross_covs = vmap(lambda S, F, A: utils.inv_quad_form(Q + A @ F @ A.T, S.T, A @ F), in_axes=in_axes)(
        smoothed_cov[1:], filtered_cov[:-1], As
    )

    return (smoother_out["smoothed_means"], smoothed_cov, cross_covs)