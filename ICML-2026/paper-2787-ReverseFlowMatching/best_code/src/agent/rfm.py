import functools
import pickle
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as onp
import optax

from src.components.mlp import ScoreMLP
from src.agent.critic import ReturnFunction, get_q_values, update_q
from src.utils.grad_utils import l2_norm, clip_grads


def anneal_temperature(start, end, step, num_steps):
    step = jnp.asarray(step)
    frac = jnp.clip(step / num_steps, 0.0, 1.0)
    log_temp = start + (end - start) * frac
    return jnp.exp(log_temp)


def safe_arctanh(a, eps=1e-4):
    a = jnp.clip(a, -1.0 + eps, 1.0 - eps)
    return 0.5 * (jnp.log1p(a) - jnp.log1p(-a))


def log_sech2(y):
    log2 = jnp.log(jnp.asarray(2.0, dtype=y.dtype))
    log_cosh = jnp.logaddexp(y, -y) - log2
    return -2.0 * log_cosh


@functools.partial(jax.jit, static_argnames=("module", "optimizer", "max_grad_norm"))
def update_actor_flow_matching(
    module,
    params,
    t_batch,
    y_t_batch,
    states,
    cond_temp_batch,
    y_targets,
    optimizer,
    optimizer_state,
    max_grad_norm,
):
    def loss_fn(network_params):
        preds = module.apply(
            network_params, t_batch, y_t_batch, states, cond_temp_batch
        )
        return jnp.mean((preds - y_targets) ** 2)

    loss, grad = jax.value_and_grad(loss_fn)(params)
    if max_grad_norm is not None:
        grad = clip_grads(grad, max_grad_norm)

    updates, optimizer_state = optimizer.update(grad, optimizer_state, params=params)
    params = optax.apply_updates(params, updates)

    return loss, params, optimizer_state, l2_norm(grad)


@dataclass(frozen=True)
class PosteriorEstimationCfg:
    num_mc_samples: int
    cv_weight: float
    cv_weight_autotune: bool
    t_min: float = 1e-3
    ridge: float = 1e-4
    cv_clip: float = 3.0


def default_source_distribution(key, shape, dtype):
    return jax.random.normal(key, shape, dtype=dtype)


def default_source_score(x):
    # score of N(0, I)
    return -x


def make_posterior_mean_fns_for_q(
    q_module,
    cfg: PosteriorEstimationCfg,
    *,
    source_distribution=default_source_distribution,
    source_score_function=default_source_score,
):

    K = int(cfg.num_mc_samples)
    t_min = float(cfg.t_min)
    ridge = float(cfg.ridge)
    cv_clip = float(cfg.cv_clip)
    cv_weight_autotune = bool(cfg.cv_weight_autotune)
    cv_weight_fixed = float(cfg.cv_weight)

    def per_sample(key, q_params, temperature, t, y_t, state):
        # The noise posterior and the data posterior can be obtained from each other through the linear interpolation constraint; here we estimate the data posterior.
        key_y0, _ = jax.random.split(key)

        t = jnp.clip(t, t_min, 1.0 - t_min)
        alpha_t = t
        beta_t = 1.0 - t

        D = y_t.shape[-1]

        # y0 ~ p0 in latent
        y0 = source_distribution(key_y0, (K, D), dtype=y_t.dtype)

        # bridge: y_t = t y1 + (1-t) y0  ->  y1 = (y_t - (1-t) y0) / t
        y1 = (y_t[None, :] - beta_t * y0) / alpha_t

        # reconstruct y0 from y1 (for score term)
        y0_recon = (y_t[None, :] - alpha_t * y1) / beta_t

        # replicate state for MC samples
        s_rep = jnp.broadcast_to(state[None, :], (K, state.shape[-1]))

        # Q(s, a) with a = tanh(y1)
        def q_fn(latent_y):
            a = jnp.tanh(latent_y)
            q = get_q_values(q_module, q_params, s_rep, a)
            if q.ndim == 2 and q.shape[-1] == 1:
                q = q[:, 0]
            return q[:, None]  # (K, 1)

        q_raw, vjp_fn = jax.vjp(q_fn, y1)  # (K, 1)
        q_vals = q_raw[:, 0]  # (K,)

        temp = jnp.asarray(temperature, dtype=q_vals.dtype)
        temp = jnp.maximum(temp, jnp.asarray(1e-12, dtype=q_vals.dtype))

        # ---- IMPORTANT: Boltzmann in ACTION space => include tanh Jacobian in weights ----
        log_det = jnp.sum(log_sech2(y1), axis=-1)  # (K,)
        logits = q_vals / temp + log_det
        w = jax.nn.softmax(logits, axis=0)
        ess = 1.0 / (jnp.sum(w**2) + 1e-12)

        # grad wrt y1 (includes tanh chain rule automatically)
        grad_q_wrt_y1 = vjp_fn(jnp.ones_like(q_raw))[0]  # (K, D)

        # ---- IMPORTANT: posterior score includes Jacobian score term -2*tanh(y1) ----
        score_posterior = (
            -alpha_t / beta_t * source_score_function(y0_recon)
            + grad_q_wrt_y1 / temp
            - 2.0 * jnp.tanh(y1)
        )  # (K, D)

        # E_w[y1]
        y1_mean = jnp.sum(w[:, None] * y1, axis=0)  # (D,)
        # E_w[score]
        score_mean = jnp.sum(w[:, None] * score_posterior, axis=0)  # (D,)

        # auto-tune CV weight per-dim
        if cv_weight_autotune:
            y1_centered = y1 - y1_mean[None, :]
            score_centered = score_posterior - score_mean[None, :]

            numerator = jnp.sum(
                (w[:, None] ** 2) * y1_centered * score_centered, axis=0
            )
            denominator = jnp.sum((w[:, None] ** 2) * (score_centered**2), axis=0)

            cv_weight = -numerator / (denominator + ridge)
            cv_weight = jnp.clip(cv_weight, -cv_clip, cv_clip)
        else:
            cv_weight = jnp.asarray(cv_weight_fixed, dtype=y_t.dtype)

        # CV endpoint estimator in latent
        y1_post = y1_mean + cv_weight * score_mean  # (D,)

        # implied latent y0 under bridge
        y0_post = (y_t - alpha_t * y1_post) / beta_t  # (D,)

        stats = (
            jnp.mean(cv_weight),
            ess,
            jnp.mean(q_vals),
            jnp.std(q_vals),
        )
        return y0_post, y1_post, stats

    def batch(key, q_params, temperature, t_batch, y_t_batch, states):
        keys = jax.random.split(key, t_batch.shape[0])
        y0_post, y1_post, stats = jax.vmap(
            per_sample, in_axes=(0, None, None, 0, 0, 0)
        )(keys, q_params, temperature, t_batch, y_t_batch, states)
        return y0_post, y1_post, stats

    return per_sample, batch


def _replicate_states(states, num_particles: int):
    if num_particles == 1:
        return states
    b, sd = states.shape
    return jnp.broadcast_to(states[:, None, :], (b, num_particles, sd)).reshape(
        b * num_particles, sd
    )


class RFM:
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
        batch_size,
        cv_weight: float | None = None,
        cv_weight_autotune: bool | None = None,
        cv_ridge: float = 5e-4,
        cv_clip: float = 3.0,
        rfm_on_policy=False,
        init_temperature=0.05,
        final_temperature=0.05,
        temperature_steps=int(1e6),
        num_particles_training: int = 1,
        num_particles_data: int = 1,
        num_particles_eval: int = 1,
        sampler_method: str = "heun",
        t_min: float = 1e-3,
        seed=0,
    ):
        master_key = jax.random.PRNGKey(seed)
        actor_key, critic_key, runtime_key = jax.random.split(master_key, 3)
        self.key = runtime_key

        self.sample_dim = int(sample_dim)
        self.state_dim = int(state_dim)

        # Net acts in latent space (same dimensionality as action)
        self.net = ScoreMLP(
            actor_hidden_size,
            actor_hidden_layers,
            actor_emb_size,
            self.sample_dim,
        )
        self.net_params = self.net.init(
            actor_key,
            jnp.zeros((1,)),  # t
            jnp.zeros((1, self.sample_dim)),  # y_t (latent)
            jnp.zeros((1, self.state_dim)),  # state
            jnp.zeros((1,)),  # temperature
        )
        self.net_optimizer = optax.adam(actor_lr)
        self.net_optimizer_state = self.net_optimizer.init(self.net_params)
        self.actor_max_grad_norm = actor_max_grad_norm

        self.critic = ReturnFunction(**critic_config, rng=critic_key)

        self.init_temperature = float(init_temperature)
        self.final_temperature = float(final_temperature)
        self.temperature_steps = int(temperature_steps)

        self.batch_size = int(batch_size)
        self.num_integration_steps = int(num_integration_steps)
        self.sampler_method = str(sampler_method).lower().strip()
        if self.sampler_method not in ("euler", "heun"):
            raise ValueError("sampler_method must be 'euler' or 'heun'")

        self.num_particles_training = int(num_particles_training)
        self.num_particles_data = int(num_particles_data)
        self.num_particles_eval = int(num_particles_eval)
        if self.num_particles_training < 1:
            raise ValueError("num_particles_training must be >= 1")
        if self.num_particles_data < 1:
            raise ValueError("num_particles_data must be >= 1")
        if self.num_particles_eval < 1:
            raise ValueError("num_particles_eval must be >= 1")

        self.rfm_on_policy = bool(rfm_on_policy)
        self.posterior_cfg = PosteriorEstimationCfg(
            num_mc_samples=int(num_estimator_mc_samples),
            cv_weight=float(cv_weight),
            cv_weight_autotune=bool(cv_weight_autotune),
            t_min=float(t_min),
            ridge=float(cv_ridge),
            cv_clip=float(cv_clip),
        )

        self.step = 0

        self._posterior_one, self._posterior_batch = make_posterior_mean_fns_for_q(
            self.critic.Q_module, self.posterior_cfg
        )
        self._fused_update = self._make_fused_update()
        self._sample_actions_jit = self._make_sample_actions_jit()
        self._det_sample_actions_jit = self._make_deterministic_sample_actions_jit()

    def _sample_latent_actions_flow_fn(
        self,
        params,
        q_params,
        key,
        states,
        temperature,
        *,
        num_particles: int,
        method: str,
    ):
        """
        Integrate flow in latent y-space; return (y_selected, a_selected=tanh(y_selected)).
        For num_particles>1, sample an index i ~ softmax(Q(s, a_i)/T) (Boltzmann in action space).
        """
        b = states.shape[0]
        n = int(num_particles)
        method = str(method).lower()

        t0 = jnp.asarray(self.posterior_cfg.t_min, dtype=jnp.float32)
        t1 = jnp.asarray(1.0 - self.posterior_cfg.t_min, dtype=jnp.float32)
        num_steps = int(self.num_integration_steps)
        dt = (t1 - t0) / jnp.asarray(num_steps, dtype=jnp.float32)

        states_rep = _replicate_states(states, n)
        total = b * n

        key_init, key_cat = jax.random.split(key)
        y = jax.random.normal(key_init, (total, self.sample_dim), dtype=jnp.float32)

        def velocity(t_scalar, y_batch):
            t_batch = jnp.full((y_batch.shape[0],), t_scalar, dtype=y_batch.dtype)
            cond = jnp.full_like(t_batch, temperature)
            return self.net.apply(params, t_batch, y_batch, states_rep, cond)

        def step_euler(y_curr, i):
            t = t0 + dt * i
            v = velocity(t, y_curr)
            y_next = y_curr + dt * v
            return y_next, None

        def step_heun(y_curr, i):
            t = t0 + dt * i
            v0 = velocity(t, y_curr)
            y_pred = y_curr + dt * v0
            v1 = velocity(t + dt, y_pred)
            y_next = y_curr + 0.5 * dt * (v0 + v1)
            return y_next, None

        step_fn = step_euler if method == "euler" else step_heun
        idxs = jnp.arange(num_steps, dtype=jnp.int32)
        yT, _ = jax.lax.scan(step_fn, y, idxs)  # (b*n, D)

        if n == 1:
            aT = jnp.tanh(yT)
            return yT, aT

        # reshape particles
        yT = yT.reshape(b, n, self.sample_dim)
        aT = jnp.tanh(yT).reshape(b * n, self.sample_dim)

        # Q(s, a) for each particle action
        q_vals = get_q_values(self.critic.Q_module, q_params, states_rep, aT)
        if q_vals.ndim == 2 and q_vals.shape[-1] == 1:
            q_vals = q_vals[:, 0]
        q_vals = q_vals.reshape(b, n)

        temp = jnp.asarray(temperature, dtype=q_vals.dtype)
        temp = jnp.maximum(temp, jnp.asarray(1e-12, dtype=q_vals.dtype))

        # logits for categorical sampling (stable shift)
        q_max = jnp.max(q_vals, axis=1, keepdims=True)
        logits = (q_vals - q_max) / temp  # (b, n)

        # sample one particle per batch element
        idx = jax.random.categorical(key_cat, logits, axis=1)  # (b,)
        idx_exp = idx[:, None, None]  # (b,1,1)

        y_sel = jnp.take_along_axis(yT, idx_exp, axis=1)[:, 0, :]  # (b, D)
        a_sel = jnp.tanh(y_sel)
        return y_sel, a_sel

    def _sample_actions_flow_fn(
        self,
        params,
        q_params,
        key,
        states,
        temperature,
        *,
        num_particles: int,
        method: str,
    ):
        _, a = self._sample_latent_actions_flow_fn(
            params,
            q_params,
            key,
            states,
            temperature,
            num_particles=num_particles,
            method=method,
        )
        return a

    def _sample_actions_flow_argmax_fn(
        self,
        params,
        q_params,
        key,
        states,
        temperature,
        *,
        num_particles: int,
        method: str,
    ):
        """
        Deterministic evaluation: integrate flow in latent y-space and return
        the action among particles that maximizes Q(s, a) in ACTION space.
        """
        b = states.shape[0]
        n = int(num_particles)
        method = str(method).lower()

        t0 = jnp.asarray(self.posterior_cfg.t_min, dtype=jnp.float32)
        t1 = jnp.asarray(1.0 - self.posterior_cfg.t_min, dtype=jnp.float32)
        num_steps = int(self.num_integration_steps)
        dt = (t1 - t0) / jnp.asarray(num_steps, dtype=jnp.float32)

        states_rep = _replicate_states(states, n)
        total = b * n

        # deterministic given `key`
        y = jax.random.normal(key, (total, self.sample_dim), dtype=jnp.float32)

        def velocity(t_scalar, y_batch):
            t_batch = jnp.full((y_batch.shape[0],), t_scalar, dtype=y_batch.dtype)
            cond = jnp.full_like(t_batch, temperature)
            return self.net.apply(params, t_batch, y_batch, states_rep, cond)

        def step_euler(y_curr, i):
            t = t0 + dt * i
            v = velocity(t, y_curr)
            return y_curr + dt * v, None

        def step_heun(y_curr, i):
            t = t0 + dt * i
            v0 = velocity(t, y_curr)
            y_pred = y_curr + dt * v0
            v1 = velocity(t + dt, y_pred)
            return y_curr + 0.5 * dt * (v0 + v1), None

        step_fn = step_euler if method == "euler" else step_heun
        idxs = jnp.arange(num_steps, dtype=jnp.int32)
        yT, _ = jax.lax.scan(step_fn, y, idxs)  # (b*n, D)

        aT = jnp.tanh(yT)  # (b*n, D)

        if n == 1:
            return aT

        q_vals = get_q_values(self.critic.Q_module, q_params, states_rep, aT)
        if q_vals.ndim == 2 and q_vals.shape[-1] == 1:
            q_vals = q_vals[:, 0]

        q_vals = q_vals.reshape(b, n)  # (b, n)
        aT = aT.reshape(b, n, self.sample_dim)  # (b, n, D)

        best_idx = jnp.argmax(q_vals, axis=1)  # (b,)
        best_a = aT[jnp.arange(b), best_idx, :]  # (b, D)
        return best_a

    def _make_sample_actions_jit(self):
        n = self.num_particles_data
        method = self.sampler_method

        @jax.jit
        def _sample(params, q_params, key, states, temperature):
            return self._sample_actions_flow_fn(
                params,
                q_params,
                key,
                states,
                temperature,
                num_particles=n,
                method=method,
            )

        return _sample

    def _make_deterministic_sample_actions_jit(self):
        n = self.num_particles_eval
        method = self.sampler_method

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
                s_b = s[None, :]
                a_b = self._sample_actions_flow_argmax_fn(
                    params,
                    q_params,
                    key,
                    s_b,
                    temperature,
                    num_particles=n,
                    method=method,
                )
                return a_b[0]

            return jax.vmap(_one)(keys, states)

        return _sample

    def _make_fused_update(self):
        start_log_temp = jnp.log(self.init_temperature)
        end_log_temp = jnp.log(self.final_temperature)
        temp_steps = self.temperature_steps

        batch_size = self.batch_size
        n_train_particles = self.num_particles_training
        method = self.sampler_method

        net = self.net
        net_opt = self.net_optimizer
        actor_max_grad_norm = self.actor_max_grad_norm

        q_module = self.critic.Q_module
        q_opt = self.critic.optimizer_q
        discount = self.critic.discount
        tau = self.critic.tau
        target_update_period = self.critic.target_update_period
        critic_max_grad_norm = self.critic.max_grad_norm

        t_min = float(self.posterior_cfg.t_min)

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
            actions,  # buffer actions are tanhed actions in [-1,1]
            rewards,
            masks,
            next_states,
        ):
            # rng
            key, key_t, key_noise, key_mc, key_sample = jax.random.split(key, 5)

            # temp
            rl_temp = anneal_temperature(
                start_log_temp, end_log_temp, agent_step, temp_steps
            )

            key_action, key_next_action = jax.random.split(key_sample)

            # choose actions for "on-policy" endpoint (still in action space for critic/env)
            if self.rfm_on_policy:
                y1_on_policy, actions_on_policy = self._sample_latent_actions_flow_fn(
                    net_params,
                    q_online_params,
                    key_action,
                    states,
                    rl_temp,
                    num_particles=n_train_particles,
                    method=method,
                )
            else:
                actions_on_policy = actions
                y1_on_policy = safe_arctanh(actions_on_policy)

            # next actions for critic target (action space)
            next_actions = self._sample_actions_flow_fn(
                net_params,
                q_online_params,
                key_next_action,
                next_states,
                rl_temp,
                num_particles=n_train_particles,
                method=method,
            )

            # critic update uses action-space actions
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

            # ---- flow-matching training batch in LATENT space ----
            t_batch = jax.random.uniform(
                key_t, (batch_size,), minval=t_min, maxval=1.0 - t_min
            )
            y0_noise = jax.random.normal(key_noise, y1_on_policy.shape)

            # y_t = t * y1 + (1-t) * y0  (latent)
            y_t_batch = (
                t_batch[:, None] * y1_on_policy + (1.0 - t_batch)[:, None] * y0_noise
            )

            # posterior endpoints in LATENT, with action-space Boltzmann accounted for
            y0_post, y1_post, stats = self._posterior_batch(
                key_mc, q_online_params, rl_temp, t_batch, y_t_batch, states
            )
            cv_w, ess, q_mean, q_std = stats

            # flow target velocity in LATENT: y1_post - y0_post
            targets = jax.lax.stop_gradient(y1_post - y0_post)

            # actor update (net takes y_t, outputs latent velocity)
            cond_temp_batch = rl_temp * jnp.ones_like(t_batch)
            actor_loss, net_params, net_opt_state, actor_grad_norm = (
                update_actor_flow_matching(
                    net,
                    net_params,
                    t_batch,
                    y_t_batch,  # latent
                    states,
                    cond_temp_batch,
                    targets,  # latent
                    net_opt,
                    net_opt_state,
                    actor_max_grad_norm,
                )
            )
            agent_step += 1

            # target network update
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
                "posterior_cv_weight": jnp.mean(cv_w),
                "posterior_ess": jnp.mean(ess),
                "posterior_q_mean": jnp.mean(q_mean),
                "posterior_q_std": jnp.mean(q_std),
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

        q_params = self.critic.Q_online_params
        if deterministic:
            actions = self._det_sample_actions_jit(
                self.net_params, q_params, states, temperature
            )
        else:
            self.key, subkey = jax.random.split(self.key)
            actions = self._sample_actions_jit(
                self.net_params, q_params, subkey, states, temperature
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

    def save(self, path: str):
        payload = {
            "score_params": self.net_params,
            "critic_online_params": self.critic.Q_online_params,
            "critic_target_params": self.critic.Q_target_params,
        }
        with open(path, "wb") as f:
            pickle.dump(jax.device_get(payload), f)
