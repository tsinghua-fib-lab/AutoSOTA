import copy
from typing import Any

import jax.numpy as np
import jax.random as jr
from jax import Array, vmap
from jax.lax import stop_gradient as stopgrad
from jax.scipy.special import logsumexp

import optax

from rp_ssm.config import Config
from rp_ssm.datasets import TrainData
from rp_ssm.dists import (
    AllParam,
    LGChainDistParam,
    LGStationaryParam,
    NatParam
)
from rp_ssm.recognition import RPMRecognition
from rp_ssm.dists_utils import parallel_smoother
from rp_ssm.types import *


class RPSSM:
    def __init__(self, prior: LGStationaryParam, recognition: list[RPMRecognition]):
        self.prior = prior
        self.recognition = recognition
        self.latent_dim = self.prior.latent_dim

    def init(self, key: Array, data: TrainData) -> AllParams:
        J = len(data.obs)
        prior_key, *rec_keys = jr.split(key, J + 1)

        prior_params = self.prior.init(prior_key, data.actions)
        rec_params = [
            enc.init(k, x[0, 0])
            for enc, k, x in zip(self.recognition, rec_keys, data.obs)
        ]

        params = (prior_params, *rec_params)
        return params

    def get_factors(
        self, rec_params: list[NetworkParams], obs: tuple[Array]
    ) -> NatParam:
        # assume obs is list of length J with each element being BxTxN
        outs = [
            vmap(vmap(lambda x: rec.apply(p, x)))(o)
            for rec, p, o in zip(self.recognition, rec_params, obs)
        ]
        return type(outs[0])(
            **{
                k: np.stack([out.params[k] for out in outs])
                for k in outs[0].params.keys()
            }
        )


class ConstrainedIVFreeEnergy:
    def __init__(self, model: RPSSM):
        self.model = model

    def init(
        self, key: Array, data: TrainData, config: Config
    ) -> tuple[AllParams, list[optax.OptState], list[optax.GradientTransformation]]:
        self.num_timesteps = data.obs[0].shape[1]
        self.batch_size = config.batch_size
        self.num_factors = len(data.obs)
        params = self.model.init(key, data)

        def map_nested_fn(fn):
            """Recursively apply `fn` to key-value pairs of a nested dict."""
            def map_fn(nested_dict):
                return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                        for k, v in nested_dict.items()}
            return map_fn
    
        label_fn = map_nested_fn(lambda k, _: k)

        # apply separate learning rates for action-dependent and action-
        # independent prior parameters
        prior_opt = optax.transforms.partition(
            {
                'Q': optax.adamw(config.prior_lr, weight_decay=1e-4),
                'A': optax.adamw(config.prior_lr, weight_decay=1e-4),
                'b': optax.adamw(config.prior_lr, weight_decay=1e-4),
                'm1': optax.adamw(config.prior_lr, weight_decay=1e-4),
                'Q1': optax.adamw(config.prior_lr, weight_decay=1e-4),
                'kernel': optax.adamw(config.act_lr, weight_decay=1e-4),
                'bias': optax.adamw(config.act_lr, weight_decay=1e-4),
                'C': optax.adamw(config.act_lr, weight_decay=1e-4),
                'c': optax.adamw(config.act_lr, weight_decay=1e-4)
            },
            label_fn
        )
        
        opts = [
            prior_opt,
            *[optax.adamw(lr, weight_decay=1e-4) for lr in config.rec_lr],
        ]
        opt_states = [opt.init(p) for opt, p in zip(opts, params)]
        return params, opt_states, opts

    def loss(
        self, params: AllParams, data: TrainData, beta: float, em: bool, key: Array
    ) -> tuple[float, Any]:
        prior_params, *rec_params = params

        prior = self.model.prior.update(prior_params)

        ### E-step
        prior_chain, factors_nat, posterior = self.get_posterior(
            key, prior, rec_params, data
        )
        if em:
            posterior = stopgrad(posterior)

        ### M-step
        kl_qf, log_Gamma, kl_qp = self.get_loss_terms(
            prior_chain, factors_nat, posterior
        )

        Z = self.batch_size * self.num_timesteps * self.num_factors
        loss = -(log_Gamma - kl_qf - beta * kl_qp) / Z

        aux = {
            "posterior": posterior,
            "factors_nat": factors_nat,
            "kl_qp": kl_qp / Z,
            "kl_qf": kl_qf / Z,
            "log_Gamma": log_Gamma / Z,
        }

        return loss, aux

    def get_posterior(self, key, prior, rec_params, data):
        factors_nat = self.model.get_factors(rec_params, data.obs)  # JxBxTxK
        factors_tot = AllParam(factors_nat.sum(axis=0))  # BxTxK
        timesteps = data.obs[0].shape[1]
        B = data.obs[0].shape[0]
        prior_chain, inference_params = prior.to_chain(
            timesteps, data.actions
        )  # with actions: BxTxK; without actions: TxK
        dist_param = copy.copy(factors_tot.dist_param)

        if prior_chain.params["means"].ndim == 2:
            in_axes = (0, None)
        elif prior_chain.params["means"].ndim == 3:
            in_axes = (0, 0)
        
        means, covs, cross_covs = vmap(
            lambda f, p: parallel_smoother(p, f, self.model.latent_dim),
            in_axes=in_axes
        )(
            dist_param,
            inference_params
        )  # BxTxK

        posterior = LGChainDistParam(means=means, covs=covs, cross_covs=cross_covs)

        return prior_chain, factors_nat, posterior

    def get_loss_terms(self, prior_chain, factors_nat, posterior):

        if prior_chain.params["means"].ndim == 2:
            kl_qp = vmap(lambda qtk: qtk.kl(prior_chain))(posterior)  # B
            prior_chain = prior_chain.all_param  # TxK

        elif prior_chain.params["means"].ndim == 3:
            kl_qp = vmap(lambda qtk, ptk: qtk.kl(ptk))(posterior, prior_chain)  # B
            prior_chain = prior_chain.moment_match.all_param  # BxTxK

        posterior = posterior.all_param

        kl_qf = vmap(
            lambda fntk: vmap(vmap(lambda qtk, ftk: qtk.kl(qtk + ftk)))(
                posterior.nat_param, fntk
            )
        )(
            factors_nat
        )  # JxBxT
        log_gammas = vmap(
            lambda fntk: vmap(
                lambda qnk, fnk, pk: vmap(
                    lambda qk: vmap(
                        lambda fk: (fk + qk).lognormalizer - (fk + pk).lognormalizer
                    )(fnk)
                )(qnk),
                in_axes=(1, 1, 0),
            )(posterior.nat_param, fntk, prior_chain.nat_param)
        )(
            factors_nat
        )  # JxTxBxB

        log_Gamma = vmap(vmap(lambda G: np.diag(G) - logsumexp(G, axis=1)))(
            log_gammas
        )  # JxTxB

        return kl_qf.sum(), log_Gamma.sum(), kl_qp.sum()