import functools
import pickle

import jax
import jax.numpy as jnp
import numpy as onp
import optax

from src.components.mlp import ScoreMLP
from src.agent.critic import ReturnFunction, update_q
from src.utils.grad_utils import l2_norm, clip_grads


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return jnp.linspace(beta_start, beta_end, timesteps)


def vp_beta_schedule(timesteps):
    t = jnp.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
    betas = 1 - alpha
    return betas


def make_beta_schedule(beta_schedule: str, timesteps: int):
    name = str(beta_schedule).lower().strip()
    if name == "cosine":
        betas = cosine_beta_schedule(timesteps)
    elif name == "linear":
        betas = linear_beta_schedule(timesteps)
    elif name == "vp":
        betas = vp_beta_schedule(timesteps)
    else:
        raise ValueError(f"Invalid beta schedule: {beta_schedule}")
    alphas = 1.0 - betas
    alpha_hats = jnp.cumprod(alphas, axis=0)
    return betas, alphas, alpha_hats


@functools.partial(jax.jit, static_argnames=("module", "optimizer", "max_grad_norm"))
def update_actor_qsm(
    module,
    params,
    times,
    noisy_actions,
    states,
    temps,
    eps_target,
    optimizer,
    optimizer_state,
    max_grad_norm,
):
    def loss_fn(network_params):
        eps_pred = module.apply(network_params, times, noisy_actions, states, temps)
        diff = eps_pred - jax.lax.stop_gradient(eps_target)
        return jnp.mean(jnp.mean(diff**2, axis=-1))

    loss, grad = jax.value_and_grad(loss_fn)(params)
    if max_grad_norm is not None:
        grad = clip_grads(grad, max_grad_norm)
    updates, optimizer_state = optimizer.update(grad, optimizer_state, params=params)
    params = optax.apply_updates(params, updates)
    return loss, params, optimizer_state, l2_norm(grad)


class QSM:
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
        num_diffusion_steps: int = 5,
        ddpm_temperature: float = 1.0,
        clip_sampler: bool = True,
        beta_schedule: str = "vp",
        q_grad_scale: float = 1.0,
        target_action_noise: float = 0.1,
        exploration_noise: float = 0.1,
        seed: int = 0,
    ):
        master_key = jax.random.PRNGKey(seed)
        actor_key, critic_key, runtime_key = jax.random.split(master_key, 3)
        self.key = runtime_key

        self.sample_dim = int(sample_dim)
        self.state_dim = int(state_dim)
        self.actor_max_grad_norm = actor_max_grad_norm

        self.net = ScoreMLP(
            actor_hidden_size,
            actor_hidden_layers,
            actor_emb_size,
            self.sample_dim,
        )
        self.net_params = self.net.init(
            actor_key,
            jnp.zeros((1,)),
            jnp.zeros((1, self.sample_dim)),
            jnp.zeros((1, self.state_dim)),
            jnp.zeros((1,)),
        )
        self.net_optimizer = optax.adam(actor_lr)
        self.net_optimizer_state = self.net_optimizer.init(self.net_params)

        self.critic = ReturnFunction(**critic_config, rng=critic_key)

        self.num_diffusion_steps = int(num_diffusion_steps)
        self.ddpm_temperature = float(ddpm_temperature)
        self.clip_sampler = bool(clip_sampler)
        self.beta_schedule = str(beta_schedule)
        self.q_grad_scale = float(q_grad_scale)
        self.target_action_noise = float(target_action_noise)
        self.exploration_noise = float(exploration_noise)

        self.betas, self.alphas, self.alpha_hats = make_beta_schedule(
            self.beta_schedule, self.num_diffusion_steps
        )

        self.step = 0

        self._sample_actions_jit = self._make_sample_actions_jit()
        self._det_sample_actions_jit = self._make_deterministic_sample_actions_jit()
        self._fused_update = self._make_fused_update()

        self.init_temperature = float(self.ddpm_temperature)
        self.final_temperature = float(self.ddpm_temperature)
        self.temperature_steps = 1

    def _make_sample_actions_jit(self):
        net = self.net
        alphas = self.alphas
        alpha_hats = self.alpha_hats
        betas = self.betas
        num_steps = self.num_diffusion_steps
        clip_sampler = self.clip_sampler
        sample_dim = self.sample_dim

        @jax.jit
        def _sample(params, key, states, temperature):
            batch_size = states.shape[0]
            key, key_init = jax.random.split(key)
            x = jax.random.normal(key_init, (batch_size, sample_dim), dtype=jnp.float32)
            temp = jnp.broadcast_to(temperature, (batch_size,))

            def step_fn(carry, t_idx):
                x, key = carry
                t_batch = jnp.full((batch_size,), t_idx, dtype=jnp.float32)
                eps_pred = net.apply(params, t_batch, x, states, temp)

                alpha_1 = 1.0 / jnp.sqrt(alphas[t_idx])
                alpha_2 = (1.0 - alphas[t_idx]) / jnp.sqrt(1.0 - alpha_hats[t_idx])
                x = alpha_1 * (x - alpha_2 * eps_pred)

                key, key_noise = jax.random.split(key)
                z = jax.random.normal(key_noise, x.shape, dtype=jnp.float32)
                x = x + jnp.where(
                    t_idx > 0, jnp.sqrt(betas[t_idx]) * (temp[:, None] * z), 0.0
                )
                if clip_sampler:
                    x = jnp.clip(x, -1.0, 1.0)
                return (x, key), None

            idxs = jnp.arange(num_steps - 1, -1, -1)
            (x, key), _ = jax.lax.scan(step_fn, (x, key), idxs)
            if clip_sampler:
                x = jnp.clip(x, -1.0, 1.0)
            return x, key

        return _sample

    def _make_deterministic_sample_actions_jit(self):
        ddpm_temp = self.ddpm_temperature

        @jax.jit
        def _sample(params, states):
            states_f32 = states.astype(jnp.float32)
            bits = jax.lax.bitcast_convert_type(states_f32, jnp.uint32)
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

            def _one(key, s):
                s_b = s[None, :]
                a_b, _ = self._sample_actions_jit(params, key, s_b, ddpm_temp)
                return a_b[0]

            return jax.vmap(_one)(keys, states)

        return _sample

    def _make_fused_update(self):
        net = self.net
        net_opt = self.net_optimizer
        actor_max_grad_norm = self.actor_max_grad_norm

        q_module = self.critic.Q_module
        q_opt = self.critic.optimizer_q
        discount = self.critic.discount
        tau = self.critic.tau
        target_update_period = self.critic.target_update_period
        critic_max_grad_norm = self.critic.max_grad_norm

        alpha_hats = self.alpha_hats
        num_steps = self.num_diffusion_steps
        ddpm_temperature = self.ddpm_temperature
        q_grad_scale = self.q_grad_scale
        target_action_noise = self.target_action_noise

        sample_fn = self._sample_actions_jit

        def _critic_action_gradients(q_params, states, actions):
            def grad_single(params):
                def q_sum(a):
                    inputs = jnp.concatenate([states, a], axis=-1)
                    q_vals = q_module.apply(params, inputs)
                    return jnp.sum(q_vals)

                return jax.grad(q_sum)(actions)

            grads = jax.vmap(grad_single, in_axes=0)(q_params)
            return grads

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
            key, key_next, key_time, key_noise, key_target_noise = jax.random.split(
                key, 5
            )

            next_actions, key_next = sample_fn(
                net_params, key_next, next_states, ddpm_temperature
            )
            if target_action_noise > 0:
                noise = (
                    jax.random.normal(key_target_noise, next_actions.shape)
                    * target_action_noise
                )
                next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

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
            critic_step += 1

            batch_size = states.shape[0]
            times = jax.random.randint(key_time, (batch_size,), 0, num_steps)
            alpha_hat = alpha_hats[times]
            alpha_1 = jnp.sqrt(alpha_hat)
            alpha_2 = jnp.sqrt(1.0 - alpha_hat)

            noise = jax.random.normal(key_noise, actions.shape)
            noisy_actions = alpha_1[:, None] * actions + alpha_2[:, None] * noise

            critic_grads = _critic_action_gradients(
                q_online_params, states, noisy_actions
            )
            critic_jacobian = jnp.mean(critic_grads, axis=0)
            eps_target = -q_grad_scale * jax.lax.stop_gradient(critic_jacobian)

            temp_batch = ddpm_temperature * jnp.ones((batch_size,))
            actor_loss, net_params, net_opt_state, actor_grad_norm = update_actor_qsm(
                net,
                net_params,
                times.astype(jnp.float32),
                noisy_actions,
                states,
                temp_batch,
                eps_target,
                net_opt,
                net_opt_state,
                actor_max_grad_norm,
            )
            agent_step += 1

            q_target_params = jax.lax.cond(
                (critic_step % target_update_period) == 0,
                lambda: jax.tree_util.tree_map(
                    lambda p, t: tau * p + (1.0 - tau) * t,
                    q_online_params,
                    q_target_params,
                ),
                lambda: q_target_params,
            )

            metrics = {
                "q_loss": q_loss,
                "q_grad_norm": q_grad_norm,
                "actor_loss": actor_loss,
                "actor_grad_norm": actor_grad_norm,
                "ddpm_temperature": ddpm_temperature,
            }

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

    def sample_actions(self, states, deterministic: bool = False):
        single = states.ndim == 1
        if single:
            states = jnp.expand_dims(states, 0)

        if deterministic:
            actions = self._det_sample_actions_jit(self.net_params, states)
        else:
            self.key, key_sample, key_noise = jax.random.split(self.key, 3)
            actions, _ = self._sample_actions_jit(
                self.net_params, key_sample, states, self.ddpm_temperature
            )
            if self.exploration_noise > 0:
                noise = (
                    jax.random.normal(key_noise, actions.shape) * self.exploration_noise
                )
                actions = jnp.clip(actions + noise, -1.0, 1.0)

        if single:
            actions = actions[0]
        return onp.array(actions), self.ddpm_temperature

    def update(self, states, actions, rewards, masks, next_states):
        states, actions = jnp.asarray(states), jnp.asarray(actions)
        rewards, masks = jnp.asarray(rewards), jnp.asarray(masks)
        next_states = jnp.asarray(next_states)

        (
            self.net_params,
            self.net_optimizer_state,
            self.critic.Q_online_params,
            self.critic.Q_target_params,
            self.critic.optimizer_state_q,
            self.critic.step,
            self.step,
            self.key,
            metrics,
        ) = self._fused_update(
            self.net_params,
            self.net_optimizer_state,
            self.critic.Q_online_params,
            self.critic.Q_target_params,
            self.critic.optimizer_state_q,
            self.critic.step,
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
            "critic_online_params": self.critic.Q_online_params,
            "critic_target_params": self.critic.Q_target_params,
        }
        payload = jax.device_get(payload)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
