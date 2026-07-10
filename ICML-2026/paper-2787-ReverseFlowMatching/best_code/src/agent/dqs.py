import functools
import pickle

import jax
import jax.numpy as jnp
import numpy as onp
import optax

from src.components.lambda_weighter import BasicLambdaWeighter
from src.components.score_estimator import estimate_grad_Rt
from src.components.sde_integration import integrate_sde
from src.components.noise_schedule import noise_h
from src.components.mlp import ScoreMLP
from src.agent.critic import ReturnFunction, update_q
from src.utils.grad_utils import l2_norm, clip_grads


def anneal_temperature(start, end, step, num_steps):
    step = jnp.asarray(step)
    frac = jnp.clip(step / num_steps, 0.0, 1.0)
    log_temp = start + (end - start) * frac
    return jnp.exp(log_temp)


@functools.partial(jax.jit, static_argnames=("module", "optimizer", "max_grad_norm"))
def update_scores(
    module,
    params,
    times,
    actions,
    states,
    estimated_scores,
    lambda_weights,
    optimizer,
    optimizer_state,
    temps,
    max_grad_norm,
):
    def loss_fn(params):
        predicted_scores = module.apply(params, times, actions, states, temps)
        error_norms = jnp.mean(((predicted_scores - estimated_scores) ** 2), axis=-1)
        dqs_loss = jnp.mean(lambda_weights * error_norms)
        return dqs_loss

    loss, grad = jax.value_and_grad(loss_fn)(params)
    if max_grad_norm is not None:
        grad = clip_grads(grad, max_grad_norm)
    updates, optimizer_state = optimizer.update(grad, optimizer_state, params=params)
    params = optax.apply_updates(params, updates)
    return loss, params, optimizer_state, l2_norm(grad)


@functools.partial(jax.jit, static_argnames=("module", "optimizer", "input_shape"))
def init_fn(module, optimizer, input_shape, rng):
    dummy_input = jnp.ones((1, *input_shape))
    params = module.init(rng, dummy_input)
    opt_state = optimizer.init(params)
    return opt_state, params


class DQS:
    def __init__(
        self,
        sample_dim,
        state_dim,
        critic_config,
        actor_hidden_size,
        actor_hidden_layers,
        actor_emb_size,
        actor_lr,
        actor_max_grad_norm,
        num_estimator_mc_samples,
        num_integration_steps,
        num_samples_to_sample_from_buffer,
        sigma_min=0.00001,
        sigma_max=1.0,
        init_temperature=0.05,
        final_temperature=0.05,
        temperature_steps=int(1e6),
        diffusion_scale=1.0,
        seed=0,
    ):
        master_key = jax.random.PRNGKey(seed)
        actor_key, critic_key, runtime_key = jax.random.split(master_key, 3)
        self.key = runtime_key
        self.state_dim = state_dim
        self.sample_dim = sample_dim
        self.net = ScoreMLP(
            actor_hidden_size,
            actor_hidden_layers,
            actor_emb_size,
            sample_dim,
        )

        self.net_params = self.net.init(
            actor_key,
            jnp.zeros((1,)),
            jnp.zeros((1, sample_dim)),
            jnp.zeros((1, state_dim)),
            jnp.zeros((1,)),
        )
        self.net_optimizer = optax.adam(actor_lr)
        self.net_optimizer_state = self.net_optimizer.init(self.net_params)
        self.actor_max_grad_norm = actor_max_grad_norm

        self.energy_function = ReturnFunction(**critic_config, rng=critic_key)

        self.estimate_grad = estimate_grad_Rt

        self.sigma_min = sigma_min
        self.sigma_diff = sigma_max / sigma_min
        self.init_temperature = init_temperature
        self.final_temperature = final_temperature
        self.temperature_steps = temperature_steps

        self.num_estimator_mc_samples = num_estimator_mc_samples
        self.num_integration_steps = num_integration_steps
        self.num_samples_to_sample_from_buffer = num_samples_to_sample_from_buffer

        self.lambda_weighter = BasicLambdaWeighter(
            self.sigma_min, self.sigma_diff, epsilon=1e-3
        )
        self.diffusion_scale = diffusion_scale

        self.step = 0
        self._sample_actions_jit = self._make_sample_actions_jit()
        self._det_sample_actions_jit = self._make_deterministic_sample_actions_jit()
        self._fused_update = self._make_fused_update()

    def _make_deterministic_sample_actions_jit(self):
        net = self.net
        sigma_min = self.sigma_min
        sigma_diff = self.sigma_diff
        sample_dim = self.sample_dim
        num_integration_steps = self.num_integration_steps

        @jax.jit
        def _sample(params, cond, temperature):
            cond_f32 = cond.astype(jnp.float32)
            bits = jax.lax.bitcast_convert_type(cond_f32, jnp.uint32)
            idx = jnp.arange(bits.shape[1], dtype=jnp.uint32)[None, :]

            x = bits + jnp.uint32(0x9E3779B9) * (idx + jnp.uint32(1))
            x = x ^ (x >> 16)
            x = x * jnp.uint32(0x85EBCA6B)
            x = x ^ (x >> 13)
            x = x * jnp.uint32(0xC2B2AE35)
            x = x ^ (x >> 16)

            seeds = jnp.sum(x, axis=1, dtype=jnp.uint32)
            seeds = jnp.where(seeds == 0, jnp.uint32(1), seeds)
            keys = jax.vmap(jax.random.PRNGKey)(seeds)

            base_scale = noise_h(1.0, sigma_min, sigma_diff) ** 0.5

            def _noise(k):
                return jax.random.normal(k, (sample_dim,), dtype=jnp.float32)

            samples = jax.vmap(_noise)(keys) * base_scale

            final_samples, _ = integrate_sde(
                jax.random.PRNGKey(0),
                net,
                params,
                samples,
                cond,
                num_integration_steps,
                sigma_min,
                sigma_diff,
                diffusion_scale=0.0,
                temperature=temperature,
            )
            return final_samples

        return _sample

    def _make_fused_update(self):
        net = self.net
        net_opt = self.net_optimizer
        sigma_min = self.sigma_min
        sigma_diff = self.sigma_diff
        num_samples = self.num_samples_to_sample_from_buffer
        num_mc = self.num_estimator_mc_samples
        lambda_weighter = self.lambda_weighter
        estimate_grad_fn = self.estimate_grad
        energy_fn = self.energy_function
        q_module = self.energy_function.Q_module
        q_opt = self.energy_function.optimizer_q
        discount = self.energy_function.discount
        tau = self.energy_function.tau
        target_update_period = self.energy_function.target_update_period
        critic_max_grad_norm = self.energy_function.max_grad_norm
        sample_fn = self._sample_actions_jit
        start_log_temp = jnp.log(self.init_temperature)
        end_log_temp = jnp.log(self.final_temperature)
        temp_steps = self.temperature_steps
        diff_scale = self.diffusion_scale
        actor_max_grad_norm = self.actor_max_grad_norm

        @jax.jit
        def _update(
            net_params,
            net_opt_state,
            q_online_params,
            q_target_params,
            q_opt_state,
            critic_step,
            agent_step,
            key,
            states,
            actions,
            rewards,
            masks,
            next_states,
        ):
            agent_step += 1

            next_actions, key_after_sample = sample_fn(
                net_params,
                key,
                next_states.shape[0],
                next_states,
                diff_scale,
                anneal_temperature(
                    start_log_temp,
                    end_log_temp,
                    agent_step,
                    temp_steps,
                )
                * jnp.ones(next_states.shape[0]),
            )
            next_actions = jnp.tanh(next_actions)
            key1, key2, key3, key_out = jax.random.split(key_after_sample, 4)

            temperature = anneal_temperature(
                start_log_temp,
                end_log_temp,
                agent_step,
                temp_steps,
            )

            times = jax.random.uniform(key1, (num_samples,))
            temperature_array = jnp.ones(num_samples) * temperature

            noised_actions = actions + (
                jax.random.normal(key2, actions.shape)
                * jnp.expand_dims(jnp.sqrt(noise_h(times, sigma_min, sigma_diff)), -1)
            )

            estimated_scores = estimate_grad_fn(
                key3,
                times,
                noised_actions,
                states,
                energy_fn,
                {"Q_online_params": q_online_params},
                num_mc_samples=num_mc,
                sigma_min=sigma_min,
                sigma_diff=sigma_diff,
                temperature=temperature_array,
            )

            lambda_weights = lambda_weighter(times)

            dqs_loss, net_params, net_opt_state, dqs_grad_norm = update_scores(
                net,
                net_params,
                times,
                noised_actions,
                states,
                estimated_scores,
                lambda_weights,
                net_opt,
                net_opt_state,
                temperature_array,
                actor_max_grad_norm,
            )

            q_loss, q_online_params, q_opt_state, q_grad_norm = update_q(
                q_module,
                q_online_params,
                q_target_params,
                states,
                actions,
                rewards,
                masks,
                next_states,
                next_actions,
                q_opt,
                q_opt_state,
                discount,
                critic_max_grad_norm,
            )

            q_target_params = jax.lax.cond(
                (critic_step % target_update_period) == 0,
                lambda: jax.tree_util.tree_map(
                    lambda p, t: tau * p + (1.0 - tau) * t,
                    q_online_params,
                    q_target_params,
                ),
                lambda: q_target_params,
            )
            critic_step += 1

            metrics = {
                "q_loss": q_loss,
                "q_grad_norm": q_grad_norm,
                "dqs_loss": dqs_loss,
                "dqs_grad_norm": dqs_grad_norm,
                "temperature": temperature,
            }

            key = key_out

            return (
                net_params,
                net_opt_state,
                q_online_params,
                q_target_params,
                q_opt_state,
                critic_step,
                agent_step,
                key,
                metrics,
            )

        return _update

    def _make_sample_actions_jit(self):
        net = self.net
        sigma_min = self.sigma_min
        sigma_diff = self.sigma_diff

        def _sample(params, key, num_samples, cond, diffusion_scale, temperature):
            base_scale = noise_h(1.0, sigma_min, sigma_diff) ** 0.5
            samples = (
                jax.random.normal(key, (num_samples, self.sample_dim)) * base_scale
            )
            final_samples, key = integrate_sde(
                key,
                net,
                params,
                samples,
                cond,
                self.num_integration_steps,
                sigma_min,
                sigma_diff,
                diffusion_scale=diffusion_scale,
                temperature=temperature,
            )
            return final_samples, key

        return jax.jit(_sample, static_argnums=(2,))

    def sample_actions(self, states, deterministic: bool = False):
        single = states.ndim == 1
        if states.ndim == 1:
            states = jnp.expand_dims(states, 0)
        num_samples = states.shape[0]

        temperature = anneal_temperature(
            onp.log(self.init_temperature),
            onp.log(self.final_temperature),
            self.step,
            self.temperature_steps,
        )
        temperature_array = jnp.ones(num_samples) * temperature
        if deterministic:
            actions = self._det_sample_actions_jit(
                self.net_params,
                states,
                temperature_array,
            )
        else:
            actions, self.key = self._sample_actions_jit(
                self.net_params,
                self.key,
                num_samples,
                states,
                self.diffusion_scale,
                temperature_array,
            )
        if single:
            actions = actions[0]
        return onp.tanh(actions), temperature

    def update(self, states, actions, rewards, masks, next_states):

        states = jnp.asarray(states)
        actions = jnp.asarray(actions)
        rewards = jnp.asarray(rewards)
        masks = jnp.asarray(masks)
        next_states = jnp.asarray(next_states)

        (
            self.net_params,
            self.net_optimizer_state,
            self.energy_function.Q_online_params,
            self.energy_function.Q_target_params,
            self.energy_function.optimizer_state_q,
            self.energy_function.step,
            self.step,
            self.key,
            metrics,
        ) = self._fused_update(
            self.net_params,
            self.net_optimizer_state,
            self.energy_function.Q_online_params,
            self.energy_function.Q_target_params,
            self.energy_function.optimizer_state_q,
            self.energy_function.step,
            self.step,
            self.key,
            states,
            actions,
            rewards,
            masks,
            next_states,
        )

        return metrics

    def save(self, path: str) -> None:
        payload = {
            "score_params": self.net_params,
            "critic_online_params": self.energy_function.Q_online_params,
            "critic_target_params": self.energy_function.Q_target_params,
        }
        payload = jax.device_get(payload)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
