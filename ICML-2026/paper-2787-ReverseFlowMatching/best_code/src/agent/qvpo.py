import pickle

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp
import optax

from src.components.mlp import default_init
from src.utils.grad_utils import l2_norm, clip_grads


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


def sinusoidal_pos_emb(t, dim):
    if dim % 2 != 0:
        raise ValueError(f"time embedding dim must be even, got {dim}")
    half_dim = dim // 2
    emb = jnp.log(jnp.asarray(10000.0)) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = t[:, None] * emb[None, :]
    return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    return onp.linspace(beta_start, beta_end, timesteps, dtype=onp.float32)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = onp.linspace(0, steps, steps, dtype=onp.float32)
    alphas_cumprod = onp.cos(((x / steps) + s) / (1 + s) * onp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return onp.clip(betas, a_min=0.0, a_max=0.999).astype(onp.float32)


def vp_beta_schedule(timesteps):
    t = onp.arange(1, timesteps + 1, dtype=onp.float32)
    T = float(timesteps)
    b_max = 10.0
    b_min = 0.1
    alpha = onp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / (T**2))
    return (1.0 - alpha).astype(onp.float32)


def make_diffusion_coeffs(num_timesteps, beta_schedule):
    schedule = str(beta_schedule).lower().strip()
    if schedule == "linear":
        betas = linear_beta_schedule(num_timesteps)
    elif schedule == "cosine":
        betas = cosine_beta_schedule(num_timesteps)
    elif schedule == "vp":
        betas = vp_beta_schedule(num_timesteps)
    else:
        raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

    alphas = 1.0 - betas
    alphas_cumprod = onp.cumprod(alphas, axis=0)
    alphas_cumprod_prev = onp.append(
        onp.array([1.0], dtype=onp.float32), alphas_cumprod[:-1]
    )

    sqrt_alphas_cumprod = onp.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = onp.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = onp.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = onp.sqrt(1.0 / alphas_cumprod - 1.0)

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_log_variance_clipped = onp.log(onp.maximum(posterior_variance, 1e-20))
    posterior_mean_coef1 = (
        betas * onp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
        (1.0 - alphas_cumprod_prev) * onp.sqrt(alphas) / (1.0 - alphas_cumprod)
    )

    to_jnp = lambda x: jnp.asarray(x, dtype=jnp.float32)
    return {
        "betas": to_jnp(betas),
        "alphas": to_jnp(alphas),
        "alphas_cumprod": to_jnp(alphas_cumprod),
        "alphas_cumprod_prev": to_jnp(alphas_cumprod_prev),
        "sqrt_alphas_cumprod": to_jnp(sqrt_alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": to_jnp(sqrt_one_minus_alphas_cumprod),
        "sqrt_recip_alphas_cumprod": to_jnp(sqrt_recip_alphas_cumprod),
        "sqrt_recipm1_alphas_cumprod": to_jnp(sqrt_recipm1_alphas_cumprod),
        "posterior_variance": to_jnp(posterior_variance),
        "posterior_log_variance_clipped": to_jnp(posterior_log_variance_clipped),
        "posterior_mean_coef1": to_jnp(posterior_mean_coef1),
        "posterior_mean_coef2": to_jnp(posterior_mean_coef2),
    }


def _extract(arr, t, x_shape):
    out = jnp.take(arr, t, axis=0)
    return out.reshape((t.shape[0],) + (1,) * (len(x_shape) - 1))


def _deterministic_keys_from_obs(obs: jnp.ndarray) -> jnp.ndarray:
    obs_f32 = obs.astype(jnp.float32)
    bits = jax.lax.bitcast_convert_type(obs_f32, jnp.uint32)
    idx = jnp.arange(bits.shape[1], dtype=jnp.uint32)[None, :]

    x = bits + jnp.uint32(0x9E3779B9) * (idx + jnp.uint32(1))
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x85EBCA6B)
    x = x ^ (x >> 13)
    x = x * jnp.uint32(0xC2B2AE35)
    x = x ^ (x >> 16)

    seeds = jnp.sum(x, axis=1, dtype=jnp.uint32)
    seeds = jnp.where(seeds == 0, jnp.uint32(1), seeds)
    return jax.vmap(jax.random.PRNGKey)(seeds)


class QVPOPolicyNet(nn.Module):
    state_dim: int
    action_dim: int
    hidden_size: int
    hidden_layers: int
    time_dim: int = 32

    @nn.compact
    def __call__(self, actions, t, states):
        t = t.astype(jnp.float32)
        t_emb = sinusoidal_pos_emb(t, self.time_dim)
        t_emb = nn.Dense(self.hidden_size, kernel_init=default_init())(t_emb)
        t_emb = mish(t_emb)
        t_emb = nn.Dense(self.time_dim, kernel_init=default_init())(t_emb)

        x = jnp.concatenate([actions, t_emb, states], axis=-1)
        x = nn.Dense(self.hidden_size, kernel_init=default_init())(x)
        x = mish(x)
        for _ in range(max(1, self.hidden_layers) - 1):
            x = nn.Dense(self.hidden_size, kernel_init=default_init())(x)
            x = mish(x)
        x = nn.Dense(self.action_dim, kernel_init=default_init())(x)
        return x


class QVPOQNet(nn.Module):
    hidden_size: int
    hidden_layers: int

    @nn.compact
    def __call__(self, states, actions):
        x = jnp.concatenate([states, actions], axis=-1)
        for _ in range(max(1, self.hidden_layers)):
            x = nn.Dense(self.hidden_size, kernel_init=default_init())(x)
            x = mish(x)
        x = nn.Dense(1, kernel_init=default_init())(x)
        return x.squeeze(-1)


def q_adv_weights(q, *, v, batch_size):
    if v is None:
        v = jnp.zeros((batch_size, 1), dtype=q.dtype)
    q = q.reshape(batch_size, 1)
    v = v.reshape(batch_size, 1)
    return jnp.maximum(q - v, 0.0)


class QVPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        critic_config,
        actor_hidden_size,
        actor_hidden_layers,
        actor_emb_size,
        actor_lr,
        actor_max_grad_norm,
        beta_schedule="cosine",
        num_diffusion_steps=20,
        noise_ratio=1.0,
        behavior_sample=4,
        target_sample=4,
        eval_sample=32,
        train_sample=64,
        deterministic=False,
        eps_greedy=0.0,
        ema_alpha_mean=0.001,
        ema_alpha_std=0.001,
        action_lr=0.03,
        action_gradient_steps=20,
        action_augmentation=False,
        entropy_alpha=0.02,
        entropy_samples=10,
        seed=0,
    ):
        master_key = jax.random.PRNGKey(seed)
        actor_key, critic_key, runtime_key = jax.random.split(master_key, 3)
        self.key = runtime_key

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.time_emb_size = int(actor_emb_size)

        self.num_timesteps = int(num_diffusion_steps)
        self.noise_ratio = float(noise_ratio)
        self.behavior_sample = int(behavior_sample)
        self.target_sample = int(target_sample)
        self.eval_sample = int(eval_sample)
        self.train_sample = int(train_sample)
        self.deterministic = bool(deterministic)
        self.eps_greedy = float(eps_greedy)
        self.ema_alpha_mean = float(ema_alpha_mean)
        self.ema_alpha_std = float(ema_alpha_std)
        self.action_lr = float(action_lr)
        self.action_gradient_steps = int(action_gradient_steps)
        self.action_grad_norm = float(self.action_dim) * 0.1
        self.action_augmentation = bool(action_augmentation)
        self.entropy_alpha = float(entropy_alpha)
        self.entropy_samples = int(entropy_samples)
        self.actor_max_grad_norm = actor_max_grad_norm

        if self.target_sample <= 0:
            self.target_sample = self.behavior_sample

        self.coeffs = make_diffusion_coeffs(self.num_timesteps, beta_schedule)

        self.policy = QVPOPolicyNet(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_size=actor_hidden_size,
            hidden_layers=actor_hidden_layers,
            time_dim=self.time_emb_size,
        )
        dummy_actions = jnp.zeros((1, self.action_dim), dtype=jnp.float32)
        dummy_t = jnp.zeros((1,), dtype=jnp.float32)
        dummy_states = jnp.zeros((1, self.state_dim), dtype=jnp.float32)
        self.policy_params = self.policy.init(
            actor_key, dummy_actions, dummy_t, dummy_states
        )
        self.policy_target_params = self.policy_params
        self.policy_optimizer = optax.adam(actor_lr)
        self.policy_opt_state = self.policy_optimizer.init(self.policy_params)

        self.q_module = QVPOQNet(
            hidden_size=critic_config["hidden_size"],
            hidden_layers=critic_config["hidden_layers"],
        )
        q1_key, q2_key = jax.random.split(critic_key, 2)
        self.q1_params = self.q_module.init(q1_key, dummy_states, dummy_actions)
        self.q2_params = self.q_module.init(q2_key, dummy_states, dummy_actions)
        self.q1_target_params = self.q1_params
        self.q2_target_params = self.q2_params
        self.q_optimizer = optax.adam(float(critic_config["lr"]))
        self.q1_opt_state = self.q_optimizer.init(self.q1_params)
        self.q2_opt_state = self.q_optimizer.init(self.q2_params)

        self.discount = float(critic_config["discount"])
        self.tau = float(critic_config["tau"])
        self.target_update_period = int(critic_config.get("target_update_period", 1))
        self.critic_max_grad_norm = critic_config.get("max_grad_norm")

        self.running_q_mean = jnp.array(0.0, dtype=jnp.float32)
        self.running_q_std = jnp.array(1.0, dtype=jnp.float32)
        self.critic_step = 0
        self.step = 0

        self._sample_actions_raw_jit = self._make_sample_actions_raw_jit()
        self._sample_actions_jit = self._make_sample_actions_jit(self.behavior_sample)
        self._det_sample_actions_eval_jit = self._make_det_sample_actions_jit(
            self.eval_sample
        )
        self._fused_update = self._make_fused_update()

    def _make_sample_actions_raw_jit(self):
        policy = self.policy
        coeffs = self.coeffs
        action_dim = self.action_dim
        num_timesteps = self.num_timesteps

        def p_mean_variance(params, x, t, states):
            noise_pred = policy.apply(params, x, t, states)
            x_recon = (
                _extract(coeffs["sqrt_recip_alphas_cumprod"], t, x.shape) * x
                - _extract(coeffs["sqrt_recipm1_alphas_cumprod"], t, x.shape)
                * noise_pred
            )
            x_recon = jnp.clip(x_recon, -1.0, 1.0)
            model_mean = (
                _extract(coeffs["posterior_mean_coef1"], t, x.shape) * x_recon
                + _extract(coeffs["posterior_mean_coef2"], t, x.shape) * x
            )
            model_log_variance = _extract(
                coeffs["posterior_log_variance_clipped"], t, x.shape
            )
            return model_mean, model_log_variance

        @jax.jit
        def _sample(policy_params, states, key, noise_ratio):
            batch = states.shape[0]
            key, key_init = jax.random.split(key)
            x = jax.random.normal(key_init, (batch, action_dim), dtype=jnp.float32)

            def step_fn(carry, t_idx):
                x, key = carry
                key, key_noise = jax.random.split(key)
                t_batch = jnp.full((batch,), t_idx, dtype=jnp.int32)
                model_mean, model_log_var = p_mean_variance(
                    policy_params, x, t_batch, states
                )
                noise = jax.random.normal(key_noise, x.shape, dtype=jnp.float32)
                nonzero = (t_idx > 0).astype(jnp.float32)
                x = (
                    model_mean
                    + nonzero * jnp.exp(0.5 * model_log_var) * noise * noise_ratio
                )
                return (x, key), None

            t_seq = jnp.arange(num_timesteps - 1, -1, -1)
            (x, key), _ = jax.lax.scan(step_fn, (x, key), t_seq)
            x = jnp.clip(x, -1.0, 1.0)
            return x, key

        return _sample

    def _make_sample_actions_jit(self, num_candidates: int):
        q_module = self.q_module
        action_dim = self.action_dim
        sample_raw = self._sample_actions_raw_jit
        num_candidates = int(num_candidates)

        @jax.jit
        def _sample(policy_params, q1_params, q2_params, states, key, noise_ratio):
            batch = states.shape[0]
            states_rep = jnp.repeat(states, num_candidates, axis=0)
            actions, key = sample_raw(policy_params, states_rep, key, noise_ratio)
            actions = jnp.clip(actions, -1.0, 1.0)

            q1 = q_module.apply(q1_params, states_rep, actions)
            q2 = q_module.apply(q2_params, states_rep, actions)
            q = jnp.minimum(q1, q2)

            actions = actions.reshape(batch, num_candidates, action_dim)
            q = q.reshape(batch, num_candidates)
            idx = jnp.argmax(q, axis=1)
            best_actions = actions[jnp.arange(batch), idx]
            return best_actions, key

        return _sample

    def _make_det_sample_actions_jit(self, num_candidates: int):
        q_module = self.q_module
        action_dim = self.action_dim
        sample_raw = self._sample_actions_raw_jit
        num_candidates = int(num_candidates)

        def sample_single(policy_params, q1_params, q2_params, state, key, noise_ratio):
            keys = jax.random.split(key, num_candidates)
            states_rep = jnp.broadcast_to(state, (num_candidates, state.shape[0]))

            def sample_one(k):
                acts, _ = sample_raw(policy_params, state[None, :], k, noise_ratio)
                return acts[0]

            actions = jax.vmap(sample_one)(keys)
            q1 = q_module.apply(q1_params, states_rep, actions)
            q2 = q_module.apply(q2_params, states_rep, actions)
            q = jnp.minimum(q1, q2)
            idx = jnp.argmax(q, axis=0)
            return actions[idx]

        @jax.jit
        def _sample(policy_params, q1_params, q2_params, states, keys, noise_ratio):
            return jax.vmap(sample_single, in_axes=(None, None, None, 0, 0, None))(
                policy_params, q1_params, q2_params, states, keys, noise_ratio
            )

        return _sample

    def sample_actions(self, states, deterministic: bool = False):
        single = states.ndim == 1
        if states.ndim == 1:
            states = jnp.expand_dims(states, 0)

        if deterministic:
            keys = _deterministic_keys_from_obs(states)
            noise_ratio = 0.0 if self.deterministic else self.noise_ratio
            actions = self._det_sample_actions_eval_jit(
                self.policy_params,
                self.q1_params,
                self.q2_params,
                states,
                keys,
                noise_ratio,
            )
        else:
            self.key, key = jax.random.split(self.key)
            explore = False
            if self.eps_greedy > 0.0:
                explore = bool(
                    jax.device_get(jax.random.uniform(key, ()) < self.eps_greedy)
                )
            noise_ratio = self.noise_ratio
            if explore:
                actions, self.key = self._sample_actions_raw_jit(
                    self.policy_params, states, self.key, noise_ratio
                )
                actions = actions[0] if single else actions
                return onp.asarray(actions)

            actions, self.key = self._sample_actions_jit(
                self.policy_params,
                self.q1_params,
                self.q2_params,
                states,
                self.key,
                noise_ratio,
            )

        if single:
            actions = actions[0]
        return onp.asarray(actions)

    def _make_fused_update(self):
        policy = self.policy
        q_module = self.q_module
        coeffs = self.coeffs
        num_timesteps = self.num_timesteps
        action_dim = self.action_dim
        noise_ratio = self.noise_ratio
        discount = self.discount
        tau = self.tau
        target_update_period = self.target_update_period
        train_sample = self.train_sample
        alpha_mean = self.ema_alpha_mean
        alpha_std = self.ema_alpha_std
        action_lr = self.action_lr
        action_gradient_steps = self.action_gradient_steps
        action_grad_norm = self.action_grad_norm
        actor_max_grad_norm = self.actor_max_grad_norm
        critic_max_grad_norm = self.critic_max_grad_norm
        action_augmentation = self.action_augmentation
        entropy_alpha = self.entropy_alpha
        entropy_samples = self.entropy_samples
        target_sample = self.target_sample
        q_opt = self.q_optimizer
        policy_opt = self.policy_optimizer

        def q_min(q1_params, q2_params, states, actions):
            q1 = q_module.apply(q1_params, states, actions)
            q2 = q_module.apply(q2_params, states, actions)
            return jnp.minimum(q1, q2)

        def sample_raw(params, states, key):
            batch = states.shape[0]
            key, key_init = jax.random.split(key)
            x = jax.random.normal(key_init, (batch, action_dim), dtype=jnp.float32)

            def step_fn(carry, t_idx):
                x, key = carry
                key, key_noise = jax.random.split(key)
                t_batch = jnp.full((batch,), t_idx, dtype=jnp.int32)
                noise_pred = policy.apply(params, x, t_batch, states)
                x_recon = (
                    _extract(coeffs["sqrt_recip_alphas_cumprod"], t_batch, x.shape) * x
                    - _extract(coeffs["sqrt_recipm1_alphas_cumprod"], t_batch, x.shape)
                    * noise_pred
                )
                x_recon = jnp.clip(x_recon, -1.0, 1.0)
                model_mean = (
                    _extract(coeffs["posterior_mean_coef1"], t_batch, x.shape) * x_recon
                    + _extract(coeffs["posterior_mean_coef2"], t_batch, x.shape) * x
                )
                model_log_variance = _extract(
                    coeffs["posterior_log_variance_clipped"], t_batch, x.shape
                )
                noise = jax.random.normal(key_noise, x.shape, dtype=jnp.float32)
                nonzero = (t_idx > 0).astype(jnp.float32)
                x = (
                    model_mean
                    + nonzero * jnp.exp(0.5 * model_log_variance) * noise * noise_ratio
                )
                return (x, key), None

            t_seq = jnp.arange(num_timesteps - 1, -1, -1)
            (x, key), _ = jax.lax.scan(step_fn, (x, key), t_seq)
            x = jnp.clip(x, -1.0, 1.0)
            return x, key

        def sample_best(params, q1_params, q2_params, states, key, num_candidates):
            batch = states.shape[0]
            states_rep = jnp.repeat(states, num_candidates, axis=0)
            actions, key = sample_raw(params, states_rep, key)
            actions = jnp.clip(actions, -1.0, 1.0)
            q = q_min(q1_params, q2_params, states_rep, actions)
            actions = actions.reshape(batch, num_candidates, action_dim)
            q = q.reshape(batch, num_candidates)
            idx = jnp.argmax(q, axis=1)
            best_actions = actions[jnp.arange(batch), idx]
            return best_actions, key

        def diffusion_loss(params, states, actions, weights, key):
            batch = actions.shape[0]
            key_t, key_noise = jax.random.split(key)
            t = jax.random.randint(key_t, (batch,), 0, num_timesteps)
            noise = jax.random.normal(key_noise, actions.shape, dtype=jnp.float32)
            x_noisy = (
                _extract(coeffs["sqrt_alphas_cumprod"], t, actions.shape) * actions
                + _extract(coeffs["sqrt_one_minus_alphas_cumprod"], t, actions.shape)
                * noise
            )
            noise_pred = policy.apply(params, x_noisy, t, states)
            diff = noise_pred - noise
            per_sample = jnp.mean(diff**2, axis=-1)
            weights = weights.squeeze(-1)
            return jnp.mean(per_sample * weights)

        def action_gradient(states, actions, q1_params, q2_params):
            opt = optax.adam(action_lr)
            opt_state = opt.init(actions)

            def body_fn(_, carry):
                actions, opt_state = carry

                def loss_fn(a):
                    q = q_min(q1_params, q2_params, states, a)
                    return -jnp.mean(q)

                loss, grads = jax.value_and_grad(loss_fn)(actions)
                if action_grad_norm > 0:
                    grads = clip_grads(grads, action_grad_norm)
                updates, opt_state = opt.update(grads, opt_state, params=actions)
                actions = optax.apply_updates(actions, updates)
                actions = jnp.clip(actions, -1.0, 1.0)
                return actions, opt_state

            actions, _ = jax.lax.fori_loop(
                0, action_gradient_steps, body_fn, (actions, opt_state)
            )
            return actions

        @jax.jit
        def _update(
            policy_params,
            policy_opt_state,
            policy_target_params,
            q1_params,
            q2_params,
            q1_target_params,
            q2_target_params,
            q1_opt_state,
            q2_opt_state,
            running_q_mean,
            running_q_std,
            critic_step,
            step,
            key,
            states,
            actions,
            rewards,
            masks,
            next_states,
        ):
            key, key_next, key_actor, key_entropy, key_loss, key_out = jax.random.split(
                key, 6
            )

            next_actions, key_next = sample_best(
                policy_target_params,
                q1_target_params,
                q2_target_params,
                next_states,
                key_next,
                target_sample,
            )
            target_q = q_min(
                q1_target_params, q2_target_params, next_states, next_actions
            )
            target = rewards + discount * masks * target_q

            def q_loss_fn(params, states, actions, target):
                q_val = q_module.apply(params, states, actions)
                return jnp.mean((q_val - target) ** 2)

            q1_loss, q1_grads = jax.value_and_grad(q_loss_fn)(
                q1_params, states, actions, target
            )
            q2_loss, q2_grads = jax.value_and_grad(q_loss_fn)(
                q2_params, states, actions, target
            )
            clip_norm = critic_max_grad_norm
            if clip_norm is not None:
                q1_grads = clip_grads(q1_grads, clip_norm)
                q2_grads = clip_grads(q2_grads, clip_norm)

            q1_updates, q1_opt_state = q_opt.update(
                q1_grads, q1_opt_state, params=q1_params
            )
            q2_updates, q2_opt_state = q_opt.update(
                q2_grads, q2_opt_state, params=q2_params
            )
            q1_params = optax.apply_updates(q1_params, q1_updates)
            q2_params = optax.apply_updates(q2_params, q2_updates)
            critic_step = critic_step + 1

            def actor_update(args):
                (
                    policy_params,
                    policy_opt_state,
                    running_q_mean,
                    running_q_std,
                    key_actor,
                    key_entropy,
                    key_loss,
                ) = args

                if action_augmentation:
                    states_rep = states
                    actions_all, key_actor = sample_raw(
                        policy_params,
                        jnp.repeat(states, train_sample, axis=0),
                        key_actor,
                    )
                    actions_all = jnp.clip(actions_all, -1.0, 1.0)
                    q_all = q_min(
                        q1_params,
                        q2_params,
                        jnp.repeat(states, train_sample, axis=0),
                        actions_all,
                    )
                    q_all = q_all.reshape(states.shape[0], train_sample)
                    actions_all = actions_all.reshape(
                        states.shape[0], train_sample, action_dim
                    )
                    mean = jnp.mean(q_all)
                    std = jnp.std(q_all)
                    v = jnp.mean(q_all, axis=1, keepdims=True)
                    idx = jnp.argmax(q_all, axis=1)
                    best_actions = actions_all[jnp.arange(states.shape[0]), idx]
                    q_sel = q_all[jnp.arange(states.shape[0]), idx]
                    states_sel = states_rep

                else:
                    q_before = q_min(q1_params, q2_params, states, actions)
                    mean = jnp.mean(q_before)
                    std = jnp.std(q_before)
                    v = None
                    states_sel = states
                    best_actions = action_gradient(
                        states, actions, q1_params, q2_params
                    )
                    q_sel = q_min(q1_params, q2_params, states_sel, best_actions)

                running_q_std = running_q_std + alpha_std * (std - running_q_std)
                running_q_mean = running_q_mean + alpha_mean * (mean - running_q_mean)

                weights = q_adv_weights(
                    q_sel.reshape(-1, 1),
                    v=v,
                    batch_size=states.shape[0],
                )
                if entropy_alpha > 0.0:
                    rand_states = jnp.repeat(states_sel, entropy_samples, axis=0)
                    rand_actions = jax.random.uniform(
                        key_entropy,
                        (rand_states.shape[0], action_dim),
                        minval=-1.0,
                        maxval=1.0,
                    )
                    rand_weights = (
                        jnp.repeat(weights, entropy_samples, axis=0) * entropy_alpha
                    )
                    states_sel = jnp.concatenate([states_sel, rand_states], axis=0)
                    best_actions = jnp.concatenate([best_actions, rand_actions], axis=0)
                    weights = jnp.concatenate([weights, rand_weights], axis=0)

                loss, grads = jax.value_and_grad(diffusion_loss)(
                    policy_params, states_sel, best_actions, weights, key_loss
                )
                clip_norm = actor_max_grad_norm
                if clip_norm is not None:
                    grads = clip_grads(grads, clip_norm)
                updates, policy_opt_state = policy_opt.update(
                    grads, policy_opt_state, params=policy_params
                )
                policy_params = optax.apply_updates(policy_params, updates)
                grad_norm = l2_norm(grads)
                return (
                    policy_params,
                    policy_opt_state,
                    running_q_mean,
                    running_q_std,
                    loss,
                    grad_norm,
                )

            (
                policy_params,
                policy_opt_state,
                running_q_mean,
                running_q_std,
                policy_loss,
                policy_grad_norm,
            ) = actor_update(
                (
                    policy_params,
                    policy_opt_state,
                    running_q_mean,
                    running_q_std,
                    key_actor,
                    key_entropy,
                    key_loss,
                )
            )

            def update_targets():
                new_q1 = jax.tree_util.tree_map(
                    lambda p, t: tau * p + (1.0 - tau) * t,
                    q1_params,
                    q1_target_params,
                )
                new_q2 = jax.tree_util.tree_map(
                    lambda p, t: tau * p + (1.0 - tau) * t,
                    q2_params,
                    q2_target_params,
                )
                new_policy = jax.tree_util.tree_map(
                    lambda p, t: tau * p + (1.0 - tau) * t,
                    policy_params,
                    policy_target_params,
                )
                return new_policy, new_q1, new_q2

            def keep_targets():
                return policy_target_params, q1_target_params, q2_target_params

            policy_target_params, q1_target_params, q2_target_params = jax.lax.cond(
                (critic_step % target_update_period) == 0,
                update_targets,
                keep_targets,
            )

            metrics = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q_loss": q1_loss + q2_loss,
                "q1_grad_norm": l2_norm(q1_grads),
                "q2_grad_norm": l2_norm(q2_grads),
                "policy_loss": policy_loss,
                "policy_grad_norm": policy_grad_norm,
                "running_q_mean": running_q_mean,
                "running_q_std": running_q_std,
            }

            return (
                policy_params,
                policy_opt_state,
                policy_target_params,
                q1_params,
                q2_params,
                q1_target_params,
                q2_target_params,
                q1_opt_state,
                q2_opt_state,
                running_q_mean,
                running_q_std,
                critic_step,
                step + 1,
                key_out,
                metrics,
            )

        return _update

    def update(self, states, actions, rewards, masks, next_states):
        states = jnp.asarray(states)
        actions = jnp.asarray(actions)
        rewards = jnp.asarray(rewards)
        masks = jnp.asarray(masks)
        next_states = jnp.asarray(next_states)

        (
            self.policy_params,
            self.policy_opt_state,
            self.policy_target_params,
            self.q1_params,
            self.q2_params,
            self.q1_target_params,
            self.q2_target_params,
            self.q1_opt_state,
            self.q2_opt_state,
            self.running_q_mean,
            self.running_q_std,
            self.critic_step,
            self.step,
            self.key,
            metrics,
        ) = self._fused_update(
            self.policy_params,
            self.policy_opt_state,
            self.policy_target_params,
            self.q1_params,
            self.q2_params,
            self.q1_target_params,
            self.q2_target_params,
            self.q1_opt_state,
            self.q2_opt_state,
            self.running_q_mean,
            self.running_q_std,
            self.critic_step,
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
            "policy_params": self.policy_params,
            "policy_target_params": self.policy_target_params,
            "q1_params": self.q1_params,
            "q2_params": self.q2_params,
            "q1_target_params": self.q1_target_params,
            "q2_target_params": self.q2_target_params,
        }
        payload = jax.device_get(payload)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
