import pickle
from copy import copy
from typing import Any, Callable

import jax
import jax.numpy as np
import jax.random as jr
from jax import Array, vmap

import optax

from tqdm import tqdm

from rp_ssm import utils
from rp_ssm.config import Config, GenerativeConfig
from rp_ssm.datasets import TrainData
from rp_ssm.dists import ACLGParam, GaussianDistParam, DistParam, LGChainDistParam
from rp_ssm.generation import GenerativeModel
from rp_ssm.rpm import ConstrainedIVFreeEnergy
from rp_ssm.types import AllParams

EPS = 1e-3


class Trainer:
    params: AllParams
    opt_states: list[optax.OptState]
    opts: list[optax.GradientTransformation]
    itr: int

    def __init__(
        self,
        free_energy: ConstrainedIVFreeEnergy,
        config: Config,
        logger: Callable = lambda *x: {},
    ):
        self.free_energy = free_energy
        self.config = config
        self.logger = logger
        self.itr = 0

    def train_step(
        self,
        params: AllParams,
        opt_states: list[optax.OptState],
        data: TrainData,
        key: Array,
    ) -> tuple[float, Any, AllParams, list[optax.OptState]]:
        beta = self.config.beta_schedule(self.itr)
        em = self.config.em

        (loss, aux), grads = jax.value_and_grad(self.free_energy.loss, has_aux=True)(
            params, data, beta, em, key
        )
        
        new_params, new_opt_states = [], []
        for param, grad, opt_state, opt in zip(params, grads, opt_states, self.opts):
            updates, new_opt_state = opt.update(grad, opt_state, param)
            new_param = optax.apply_updates(param, updates)
            new_params.append(new_param)
            new_opt_states.append(new_opt_state)

        return loss, aux, tuple(new_params), new_opt_states

    def fit(self, data: TrainData, use_pbar: bool = True) -> None:
        N, T = data.obs[0].shape[:2]
        print(f"Training with N={N}, T={T}")

        key, subkey = jr.split(jr.PRNGKey(self.config.seed))
        self.params, self.opt_states, self.opts = self.free_energy.init(
            subkey, data, self.config
        )

        train_step = jax.jit(self.train_step) if self.config.jit else self.train_step

        loss_key = jr.PRNGKey(self.config.seed - 1)

        self.loss_tot = []

        pbar = tqdm(range(self.config.num_iter), disable=not (use_pbar))
        for self.itr in pbar:
            key, subkey = jr.split(key)
            if self.config.batch_size == data.obs[0].shape[0]:
                if self.itr == 0:
                    jax.debug.print("using entire dataset")
                batch_indices = np.arange(self.config.batch_size)
            else:
                batch_indices = jr.randint(
                    subkey, (self.config.batch_size,), 0, data.obs[0].shape[0]
                )
            data_batch = data[batch_indices]

            loss_key, subkey = jr.split(loss_key)
            loss, aux, self.params, self.opt_states = train_step(
                self.params, self.opt_states, data_batch, subkey
            )

            self._stabilize_params()

            self.loss_tot.append(loss)
            to_print = self.logger(self, aux, batch_indices)
            to_print.update({"loss": f"{loss:.3f}"})

            pbar.set_postfix(**to_print)

    def train_continue(self, data: TrainData, new_iter: int, key: Array):
        train_step = jax.jit(self.train_step) if self.config.jit else self.train_step

        pbar = tqdm(range(self.itr, self.itr + new_iter))
        for self.itr in pbar:
            key, subkey = jr.split(key)
            batch_indices = jr.randint(
                subkey, (self.config.batch_size,), 0, data.obs[0].shape[0]
            )
            data_batch = data[batch_indices]

            key, subkey = jr.split(key)
            loss, aux, self.params, self.opt_states = train_step(
                self.params, self.opt_states, data_batch, subkey
            )

            self._stabilize_params()

            self.loss_tot.append(loss)
            to_print = self.logger(self, aux, batch_indices)
            to_print.update({"loss": f"{loss:.3f}"})

            pbar.set_postfix(**to_print)

    def _stabilize_params(self):
        # only stabilize A if it's a full matrix
        # if it's diag with sigmoid, don't need to stabilize
        if self.config.stabilize_A is None:
            return
        if "A" not in self.params[0]:
            return
        if self.params[0]["A"].ndim == 2:
            if self.config.stabilize_A == "scale":
                self.params[0]["A"] = utils.scale_sv(self.params[0]["A"], EPS)
            elif self.config.stabilize_A == "clip":
                self.params[0]["A"] = utils.clip_sv(self.params[0]["A"], EPS)

    def apply(self, data: TrainData) -> tuple[DistParam, LGChainDistParam]:
        prior_params, *rec_params = self.params
        prior = self.free_energy.model.prior.update(prior_params)
        _, factors_nat, posterior = self.free_energy.get_posterior(
            None, prior, rec_params, data
        )
        factors = factors_nat.dist_param
        return factors, posterior

    def rollout(
        self,
        context_data: TrainData,
        extra_actions: Array,
        num_timesteps: int,
        num_rollouts: int = 0,
        key: Array = None
    ):
        """
        Given data of shape TxD, for any T>0,
        compute the posterior p(z_1:T|x_1:T, a_1:T-1)
        and predict future latents using the
        prior: p(z_T+u|x_1:T, a_1:T+u-1), for u=1,...,
        num_timesteps.

        If num_samples > 0, also return
        sample rollouts.
        """
        assert extra_actions.shape[0] == num_timesteps - 1
        # add extra axis (B=1)
        _, posterior = self.apply(context_data[None, ...])

        prior_params = copy(self.params[0])
        prior_params["A"] = prior_params.get(
            "A",
            np.zeros(
                (self.free_energy.model.latent_dim, self.free_energy.model.latent_dim)
            ),
        )
        prior_params["m1"] = posterior.params["means"][0, -1]
        prior_params["Q1"] = posterior.params["covs"][0, -1]
        prior_params["Q"] = (
            np.eye(self.free_energy.model.latent_dim)
            - prior_params["A"] @ prior_params["A"].T
        )

        final_state = ACLGParam(
            action_mapper=self.free_energy.model.prior.action_mapper,
            transition_matrix=self.free_energy.model.prior.transition_matrix,
            transition_bias=self.free_energy.model.prior.transition_bias,
            Q_dist_map=self.free_energy.model.prior.Q_dist_map,
            opt_params=[],
            start_from_invariant=False,
            **prior_params,
        )

        # actions should be TxA, but we need to add an axis
        # to make it BxTxA, where B=1
        chain, _ = final_state.to_chain(
            num_timesteps, extra_actions[None], num_samples=num_rollouts, key=key
        )

        if num_rollouts > 0:
            preds, samples = chain
        elif num_rollouts == 0:
            preds = chain
            samples = None

        # remove extra axis (B=1) from before
        predictive_dist = GaussianDistParam(
            mean=np.concatenate(
                [posterior.params["means"][0], preds.params["means"][0][1:]]
            ),
            cov=np.concatenate(
                [posterior.params["covs"][0], preds.params["covs"][0][1:]]
            ),
        )

        return predictive_dist, samples

    def save_params(self, path: str) -> None:
        """
        Save the model parameters to a file.
        """
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def load_params(self, path: str) -> None:
        """
        Load the model parameters from a file.
        """
        with open(path, "rb") as f:
            self.params = pickle.load(f)


class GenerativeTrainer:

    def __init__(
        self,
        generative_model: GenerativeModel,
        trainer: Trainer,
        config: GenerativeConfig,
    ):
        self.generative_model = generative_model
        self.trainer = trainer
        self.config = config

    def train_step(self, key, params, opt_state, data):
        num_samples = self.config.num_samples
        trainer = self.trainer

        (loss, aux), grads = jax.value_and_grad(
            self.generative_model.loss, has_aux=True
        )(params, data, key, trainer, num_samples)

        updates, opt_state = self.opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return loss, aux, params, opt_state

    def fit(self, data: TrainData, use_pbar: bool = True) -> None:
        assert (
            self.trainer.config.seed != self.config.seed
            and self.trainer.config.seed + 1 != self.config.seed
        )

        key, subkey = jr.split(jr.PRNGKey(self.config.seed))
        self.params, self.opt_state, self.opt = self.generative_model.init(
            subkey, self.trainer.free_energy.model.latent_dim, self.config
        )

        train_step = jax.jit(self.train_step) if self.config.jit else self.train_step

        self.loss_tot = []

        pbar = tqdm(range(self.config.num_iter), disable=not (use_pbar))
        for self.itr in pbar:
            key, subkey = jr.split(key)
            if self.config.batch_size == data.obs[0].shape[0]:
                if self.itr == 0:
                    jax.debug.print("using entire dataset")
                batch_indices = np.arange(self.config.batch_size)
            else:
                batch_indices = jr.randint(
                    subkey, (self.config.batch_size,), 0, data.obs[0].shape[0]
                )
            data_batch = data[batch_indices]

            loss, aux, self.params, self.opt_state = train_step(
                subkey, self.params, self.opt_state, data_batch
            )

            self.loss_tot.append(loss)

            pbar.set_postfix(loss=f"{loss:.3f}")

    def apply(self, states: Array):
        """
        Given latent states z of shape MxK,
        return the generative model output
        p(x|z) of shape MxD.
        """
        return vmap(lambda z: self.generative_model.apply(self.params, z))(states)

    def reconstruct(self, data: TrainData):
        """
        Given a data sequence of shape TxD, get
        posterior means from RP-SSM and decode
        using generative model.
        """
        _, posterior = self.trainer.apply(
            data[None, ...],
        )
        states = posterior.params["means"][0]
        return self.apply(states)

    def predict(
        self,
        data: TrainData,
        num_timesteps: int,
        num_rollouts: int = 0,
        key: Array = None,
    ):
        """
        Given a data sequence of shape TxD for
        any T>0, get posterior means and
        num_timesteps predicted latent states,
        then decode all into data space.
        """
        predictive_dist, samples = self.trainer.rollout(
            data, num_timesteps, num_rollouts, key
        )
        mean_preds = self.apply(predictive_dist.params["mean"])
        if samples is None:
            sample_preds = None
        else:
            sample_preds = vmap(self.apply)(samples)
        return mean_preds, sample_preds
