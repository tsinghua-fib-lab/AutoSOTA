import functools
import math
import pickle

import jax
import jax.numpy as jnp
import numpy as onp
import optax

from src.components.mlp import ScoreMLP
from src.agent.critic import ReturnFunction, get_q_values, update_q, update_q_sac
from src.utils.grad_utils import l2_norm, clip_grads


def anneal_temperature(start, end, step, num_steps):
    step = jnp.asarray(step)
    frac = jnp.clip(step / num_steps, 0.0, 1.0)
    log_temp = start + (end - start) * frac
    return jnp.exp(log_temp)


def _cosine_t_max(s: float = 0.008, beta_max: float = 999.0) -> float:
    return math.atan(beta_max * (1.0 + s) / math.pi) * 2.0 * (1.0 + s) / math.pi - s


def cosine_alpha_hat(t, s: float = 0.008):
    t = jnp.asarray(t)
    f = jnp.cos((t + s) / (1.0 + s) * jnp.pi * 0.5) ** 2
    f0 = jnp.cos(s / (1.0 + s) * jnp.pi * 0.5) ** 2
    return f / f0


def log_snr_from_alpha_hat(alpha_hat, eps: float = 1e-5):
    alpha_hat = jnp.clip(alpha_hat, eps, 1.0 - eps)
    return jnp.log(alpha_hat) - jnp.log1p(-alpha_hat)


def make_cosine_schedule(
    num_steps: int,
    t_min: float,
    t_max: float,
    s: float = 0.008,
):
    steps = jnp.linspace(t_min, t_max, num_steps + 1, dtype=jnp.float32)
    alpha_hat_prevs = cosine_alpha_hat(steps, s=s)
    alpha_hats = alpha_hat_prevs[1:]
    alphas = alpha_hats / alpha_hat_prevs[:-1]
    betas = 1.0 - alphas
    return alpha_hat_prevs, alpha_hats, alphas, betas


def _replicate_states(states, num_particles: int):
    if num_particles == 1:
        return states
    b, sd = states.shape
    return jnp.broadcast_to(states[:, None, :], (b, num_particles, sd)).reshape(
        b * num_particles, sd
    )


@functools.partial(jax.jit, static_argnames=("module", "optimizer", "max_grad_norm"))
def update_actor_maxentdp(
    module,
    params,
    log_snr,
    noisy_actions,
    states,
    temps,
    eps_target,
    optimizer,
    optimizer_state,
    max_grad_norm,
):
    def loss_fn(network_params):
        eps_pred = module.apply(network_params, log_snr, noisy_actions, states, temps)
        diff = eps_pred - jax.lax.stop_gradient(eps_target)
        return jnp.mean(jnp.mean(diff**2, axis=-1))

    loss, grad = jax.value_and_grad(loss_fn)(params)
    if max_grad_norm is not None:
        grad = clip_grads(grad, max_grad_norm)
    updates, optimizer_state = optimizer.update(grad, optimizer_state, params=params)
    params = optax.apply_updates(params, updates)
    return loss, params, optimizer_state, l2_norm(grad)


class MaxEntDP:
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
        num_diffusion_steps: int = 20,
        init_temperature: float = 0.1,
        final_temperature: float = 0.1,
        temperature_steps: int = int(1e6),
        clip_sampler: bool = True,
        backup_entropy: bool = True,
        log_prob_samples: int = 50,
        log_prob_steps: int = 20,
        eval_action_selection: bool = True,
        eval_candidate_num: int = 10,
        score_samples_num: int = 500,
        t_min: float = 1e-3,
        t_max: float | None = None,
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
        self.clip_sampler = bool(clip_sampler)
        self.backup_entropy = bool(backup_entropy)
        self.log_prob_samples = int(log_prob_samples)
        self.log_prob_steps = int(log_prob_steps)
        self.eval_action_selection = bool(eval_action_selection)
        self.eval_candidate_num = int(eval_candidate_num)
        self.score_samples_num = int(score_samples_num)

        self.t_min = float(t_min)
        if t_max is None:
            self.t_max = float(_cosine_t_max())
        else:
            self.t_max = float(t_max)

        self.init_temperature = float(init_temperature)
        self.final_temperature = float(final_temperature)
        self.temperature_steps = int(temperature_steps)

        (
            self.alpha_hat_prevs,
            self.alpha_hats,
            self.alphas,
            self.betas,
        ) = make_cosine_schedule(self.num_diffusion_steps, self.t_min, self.t_max)

        (
            self.lp_alpha_hat_prevs,
            self.lp_alpha_hats,
            self.lp_alphas,
            self.lp_betas,
        ) = make_cosine_schedule(self.log_prob_steps, self.t_min, self.t_max)

        self.step = 0

        self._sample_actions_jit = self._make_sample_actions_jit()
        self._det_sample_actions_jit = self._make_deterministic_sample_actions_jit()
        self._fused_update = self._make_fused_update()

    def _make_sample_actions_jit(self):
        net = self.net
        alpha_hat_prevs = self.alpha_hat_prevs
        alpha_hats = self.alpha_hats
        alphas = self.alphas
        betas = self.betas
        num_steps = self.num_diffusion_steps
        clip_sampler = self.clip_sampler
        sample_dim = self.sample_dim

        @jax.jit
        def _sample(params, key, states, temperature):
            batch_size = states.shape[0]
            key, key_init = jax.random.split(key)
            x = jax.random.normal(key_init, (batch_size, sample_dim), dtype=jnp.float32)

            temp_batch = jnp.broadcast_to(temperature, (batch_size,))

            def step_fn(carry, t_idx):
                x, key = carry
                alpha_hat = alpha_hats[t_idx]
                alpha_hat_prev = alpha_hat_prevs[t_idx]
                alpha = alphas[t_idx]
                beta = betas[t_idx]

                log_snr = log_snr_from_alpha_hat(alpha_hat)
                t_batch = jnp.full((batch_size,), log_snr, dtype=jnp.float32)
                eps_pred = net.apply(params, t_batch, x, states, temp_batch)

                x_start = (x - jnp.sqrt(1.0 - alpha_hat) * eps_pred) / jnp.sqrt(
                    alpha_hat
                )
                if clip_sampler:
                    x_start = jnp.clip(x_start, -1.0, 1.0)

                mean_coef1 = jnp.sqrt(alpha_hat_prev) * beta / (1.0 - alpha_hat)
                mean_coef2 = (
                    jnp.sqrt(alpha) * (1.0 - alpha_hat_prev) / (1.0 - alpha_hat)
                )
                mean = mean_coef1 * x_start + mean_coef2 * x

                key, key_noise = jax.random.split(key)
                z = jax.random.normal(key_noise, x.shape, dtype=jnp.float32)
                x = mean + jnp.where(t_idx > 0, jnp.sqrt(beta) * z, 0.0)
                return (x, key), None

            idxs = jnp.arange(num_steps - 1, -1, -1)
            (x, key), _ = jax.lax.scan(step_fn, (x, key), idxs)
            if clip_sampler:
                x = jnp.clip(x, -1.0, 1.0)
            return x, key

        return _sample

    def _make_deterministic_sample_actions_jit(self):
        n = self.eval_candidate_num if self.eval_action_selection else 1
        q_module = self.critic.Q_module

        @jax.jit
        def _sample(params, q_params, states, temperature):
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
                s_rep = jnp.broadcast_to(s[None, :], (n, s.shape[0]))
                actions, _ = self._sample_actions_jit(params, key, s_rep, temperature)
                if n == 1:
                    return actions[0]

                q_vals = get_q_values(q_module, q_params, s_rep, actions)
                if q_vals.ndim == 2 and q_vals.shape[-1] == 1:
                    q_vals = q_vals[:, 0]
                best_idx = jnp.argmax(q_vals, axis=0)
                return actions[best_idx]

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

        alpha_hat_prevs = self.alpha_hat_prevs
        alpha_hats = self.alpha_hats
        lp_alpha_hat_prevs = self.lp_alpha_hat_prevs
        lp_alpha_hats = self.lp_alpha_hats

        num_score_samples = self.score_samples_num
        t_min = self.t_min
        t_max = self.t_max

        start_log_temp = jnp.log(self.init_temperature)
        end_log_temp = jnp.log(self.final_temperature)
        temp_steps = self.temperature_steps

        clip_sampler = self.clip_sampler
        backup_entropy = self.backup_entropy
        log_prob_samples = self.log_prob_samples
        log_prob_steps = self.log_prob_steps

        sample_fn = self._sample_actions_jit

        def _calc_log_prob(key, params, observations, actions, temperature):
            batch_size, act_dim = actions.shape
            num_samples = log_prob_samples
            steps = log_prob_steps

            key, key_noise = jax.random.split(key)
            noise_sample = jax.random.normal(
                key_noise, (batch_size, num_samples, steps, act_dim), dtype=jnp.float32
            )

            time = jnp.arange(steps, dtype=jnp.int32)[None, None, :, None]
            time = jnp.broadcast_to(time, (batch_size, num_samples, steps, 1))

            alpha_h = lp_alpha_hats[time]
            alpha_h_prev = lp_alpha_hat_prevs[time]
            alpha_h = jnp.clip(alpha_h, 1e-5, 1.0 - 1e-5)
            log_snr = jnp.log(alpha_h) - jnp.log1p(-alpha_h)

            alpha_1 = jnp.sqrt(alpha_h)
            alpha_2 = jnp.sqrt(1.0 - alpha_h)

            actions_rep = jnp.broadcast_to(
                actions[:, None, None, :],
                (batch_size, num_samples, steps, act_dim),
            )
            obs_rep = jnp.broadcast_to(
                observations[:, None, None, :],
                (batch_size, num_samples, steps, observations.shape[-1]),
            )
            noisy_actions = alpha_1 * actions_rep + alpha_2 * noise_sample

            flat_noisy = noisy_actions.reshape((-1, act_dim))
            flat_obs = obs_rep.reshape((-1, observations.shape[-1]))
            flat_log_snr = log_snr.reshape((-1,))

            temp = jnp.broadcast_to(temperature, (batch_size,))
            temp = jnp.broadcast_to(
                temp[:, None, None], (batch_size, num_samples, steps)
            )
            flat_temp = temp.reshape((-1,))

            eps_pred = net.apply(
                params,
                flat_log_snr,
                flat_noisy,
                flat_obs,
                flat_temp,
            ).reshape((batch_size, num_samples, steps, act_dim))

            x_start = (noisy_actions - jnp.sqrt(1.0 - alpha_h) * eps_pred) / jnp.sqrt(
                alpha_h
            )
            if clip_sampler:
                x_start = jnp.clip(x_start, -1.0, 1.0)

            eps_pred = (noisy_actions - jnp.sqrt(alpha_h) * x_start) / jnp.sqrt(
                1.0 - alpha_h
            )

            weight = (alpha_h_prev - alpha_h) / (2.0 * alpha_h * (1.0 - alpha_h))
            weight = weight[:, 0, :, 0]
            alpha_h_first = alpha_h[:, 0, :, 0]

            error = jnp.sum((eps_pred - noise_sample) ** 2, axis=-1)
            error_mean = jnp.mean(error, axis=1)
            alphahat_minus_error = (alpha_h_first * act_dim - error_mean) * weight
            log_prob = jnp.sum(alphahat_minus_error, axis=1) - 0.5 * act_dim * jnp.log(
                2.0 * jnp.pi * jnp.exp(1.0)
            )

            return log_prob, key

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
            key, key_next, key_t, key_noise, key_score, key_logprob = jax.random.split(
                key, 6
            )

            rl_temp = anneal_temperature(
                start_log_temp, end_log_temp, agent_step, temp_steps
            )

            next_actions, key_next = sample_fn(
                net_params, key_next, next_states, rl_temp
            )

            if backup_entropy:
                log_prob, key_logprob = _calc_log_prob(
                    key_logprob, net_params, next_states, next_actions, rl_temp
                )
                q_loss, q_online_params, q_opt_state, q_grad_norm = update_q_sac(
                    q_module,
                    q_online_params,
                    q_target_params,
                    states,
                    actions,
                    rewards,
                    masks,
                    next_states,
                    next_actions,
                    log_prob,
                    q_opt,
                    q_opt_state,
                    discount,
                    rl_temp,
                    critic_max_grad_norm,
                )
            else:
                log_prob = jnp.asarray(0.0, dtype=jnp.float32)
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
            times = jax.random.uniform(key_t, (batch_size,), minval=t_min, maxval=t_max)
            alpha_hat = cosine_alpha_hat(times)
            log_snr = log_snr_from_alpha_hat(alpha_hat)
            alpha_1 = jnp.sqrt(alpha_hat)
            alpha_2 = jnp.sqrt(1.0 - alpha_hat)
            alpha_2 = jnp.maximum(alpha_2, 1e-5)

            noise = jax.random.normal(key_noise, actions.shape)
            noisy_actions = alpha_1[:, None] * actions + alpha_2[:, None] * noise

            noisy_actions_rep = jnp.broadcast_to(
                noisy_actions[:, None, :],
                (batch_size, num_score_samples, actions.shape[1]),
            )
            std = alpha_2 / alpha_1
            std = jnp.maximum(std, 1e-5)
            alpha_2_rep = alpha_2[:, None, None]
            inv_std = (1.0 / std)[:, None, None]
            lower = -noisy_actions_rep / alpha_2_rep - inv_std
            upper = -noisy_actions_rep / alpha_2_rep + inv_std

            key_score, key_norm = jax.random.split(key_score)
            tnormal_noise = jax.random.truncated_normal(
                key_score,
                lower=lower,
                upper=upper,
                shape=noisy_actions_rep.shape,
            )
            normal_noise = jax.random.normal(key_norm, noisy_actions_rep.shape)
            normal_noise_clip = jnp.clip(normal_noise, lower, upper)
            noise_candidates = jnp.where(
                jnp.isnan(tnormal_noise), normal_noise_clip, tnormal_noise
            )

            clean_samples = noisy_actions_rep / alpha_1[:, None, None] + (
                std[:, None, None] * noise_candidates
            )

            states_rep = _replicate_states(states, num_score_samples)
            clean_samples_flat = clean_samples.reshape((-1, actions.shape[1]))
            q_vals = get_q_values(
                q_module, q_target_params, states_rep, clean_samples_flat
            )
            if q_vals.ndim == 2 and q_vals.shape[-1] == 1:
                q_vals = q_vals[:, 0]
            q_vals = q_vals.reshape((batch_size, num_score_samples))

            temp = jnp.maximum(rl_temp, 1e-12)
            logits = q_vals / temp
            logits = logits - jnp.max(logits, axis=1, keepdims=True)
            weights = jax.nn.softmax(logits, axis=1)
            eps_target = -jnp.sum(weights[:, :, None] * noise_candidates, axis=1)

            temp_batch = rl_temp * jnp.ones((batch_size,))
            actor_loss, net_params, net_opt_state, actor_grad_norm = (
                update_actor_maxentdp(
                    net,
                    net_params,
                    log_snr,
                    noisy_actions,
                    states,
                    temp_batch,
                    eps_target,
                    net_opt,
                    net_opt_state,
                    actor_max_grad_norm,
                )
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
                "temperature": rl_temp,
                "log_prob": jnp.mean(log_prob),
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

        temperature = anneal_temperature(
            onp.log(self.init_temperature),
            onp.log(self.final_temperature),
            self.step,
            self.temperature_steps,
        )

        q_params = self.critic.Q_target_params
        if deterministic:
            actions = self._det_sample_actions_jit(
                self.net_params, q_params, states, temperature
            )
        else:
            self.key, subkey = jax.random.split(self.key)
            actions, _ = self._sample_actions_jit(
                self.net_params, subkey, states, temperature
            )

        if single:
            actions = actions[0]
        return onp.array(actions), temperature

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
