import functools
import pickle

import jax
import jax.numpy as jnp
import numpy as onp
import optax

from src.components.mlp import ActorMLP
from src.agent.critic import SACCritic, get_q_values
from src.utils.grad_utils import l2_norm, clip_grads


def _tanh_normal_sample(actor, params, states, key, log_std_min, log_std_max):
    """Sample tanh-squashed Gaussian actions and log-probs."""
    out = actor.apply(params, states)
    means, log_stds = jnp.split(out, 2, axis=-1)
    log_stds = jnp.clip(log_stds, log_std_min, log_std_max)
    stds = jnp.exp(log_stds)

    noise = jax.random.normal(key, means.shape)
    pre_tanh = means + noise * stds
    actions = jnp.tanh(pre_tanh)

    log_probs = -0.5 * (noise**2 + 2.0 * log_stds + jnp.log(2.0 * jnp.pi))
    log_probs = jnp.sum(log_probs, axis=-1)
    log_probs -= jnp.sum(jnp.log(1.0 - actions**2 + 1e-6), axis=-1)
    return actions, log_probs


class SAC:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        critic_config: dict,
        actor_hidden_size: int = 256,
        actor_hidden_layers: int = 2,
        actor_lr: float = 3e-4,
        actor_max_grad_norm=None,
        alpha_lr: float = 3e-4,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        seed: int = 0,
    ):
        master_key = jax.random.PRNGKey(seed)
        actor_key, critic_key, runtime_key = jax.random.split(master_key, 3)
        self.key = runtime_key
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.actor_max_grad_norm = actor_max_grad_norm

        self.actor = ActorMLP(actor_hidden_size, actor_hidden_layers, action_dim * 2)
        self.actor_params = self.actor.init(actor_key, jnp.zeros((1, state_dim)))
        self.actor_optimizer = optax.adam(actor_lr)
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)

        self.critic = SACCritic(**critic_config, rng=critic_key)

        self.target_entropy = -action_dim
        self.log_alpha = jnp.array(0.0, dtype=jnp.float32)
        self.alpha_opt = optax.adam(alpha_lr)
        self.alpha_opt_state = self.alpha_opt.init(self.log_alpha)

        self._fused_update = self._make_fused_update()
        self._sample_actions_jit = self._make_sample_actions_jit()
        self._det_sample_actions_jit = self._make_deterministic_actions_jit()

    def _deterministic_actions(self, states: jnp.ndarray) -> jnp.ndarray:
        out = self.actor.apply(self.actor_params, states)
        means, _ = jnp.split(out, 2, axis=-1)
        return jnp.tanh(means)

    def sample_actions(self, states, deterministic: bool = False):
        single = states.ndim == 1
        if states.ndim == 1:
            states = jnp.expand_dims(states, 0)
        if deterministic:
            actions = self._det_sample_actions_jit(self.actor_params, states)
            log_probs = None
        else:
            self.key, key = jax.random.split(self.key)
            actions, log_probs = self._sample_actions_jit(
                self.actor_params, states, key
            )
        if single:
            actions = actions[0]
            if log_probs is not None:
                log_probs = log_probs[0]
        return (
            onp.asarray(actions)
            if log_probs is None
            else (onp.asarray(actions), log_probs)
        )

    def _update_alpha(self, log_probs):

        def alpha_loss_fn(log_alpha, log_probs):
            alpha = jnp.exp(log_alpha)
            return jnp.mean(alpha * (-log_probs - self.target_entropy))

        loss, grads = jax.value_and_grad(alpha_loss_fn)(
            self.log_alpha, jax.lax.stop_gradient(log_probs)
        )
        updates, self.alpha_opt_state = self.alpha_opt.update(
            grads, self.alpha_opt_state
        )
        self.log_alpha = optax.apply_updates(self.log_alpha, updates)
        return loss, jnp.exp(self.log_alpha)

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=("actor", "q_module", "log_std_min", "log_std_max"),
    )
    def _actor_loss_and_grads(
        actor,
        q_module,
        actor_params,
        q_params,
        states,
        alpha,
        key,
        log_std_min,
        log_std_max,
    ):
        def actor_loss_fn(params, key):
            actions, log_probs = _tanh_normal_sample(
                actor,
                params,
                states,
                key,
                log_std_min,
                log_std_max,
            )
            q_values = get_q_values(q_module, q_params, states, actions)
            return jnp.mean(alpha * log_probs - q_values), log_probs

        return jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_params, key)

    def _update_actor(self, states, alpha, key):
        (loss, log_probs), grads = self._actor_loss_and_grads(
            self.actor,
            self.critic.Q_module,
            self.actor_params,
            self.critic.Q_online_params,
            states,
            alpha,
            key,
            self.log_std_min,
            self.log_std_max,
        )
        if self.actor_max_grad_norm is not None:
            grads = clip_grads(grads, self.actor_max_grad_norm)
        updates, self.actor_opt_state = self.actor_optimizer.update(
            grads, self.actor_opt_state, params=self.actor_params
        )
        self.actor_params = optax.apply_updates(self.actor_params, updates)
        return loss, log_probs, l2_norm(grads)

    def _make_fused_update(self):
        actor = self.actor
        q_module = self.critic.Q_module
        actor_opt = self.actor_optimizer
        q_opt = self.critic.optimizer_q
        alpha_opt = self.alpha_opt
        log_std_min = self.log_std_min
        log_std_max = self.log_std_max
        discount = self.critic.discount
        tau = self.critic.tau
        target_update_period = self.critic.target_update_period
        critic_max_grad_norm = self.critic.max_grad_norm
        actor_max_grad_norm = self.actor_max_grad_norm
        target_entropy = self.target_entropy

        @jax.jit
        def _update(
            actor_params,
            actor_opt_state,
            log_alpha,
            alpha_opt_state,
            q_online_params,
            q_target_params,
            q_opt_state,
            critic_step,
            key,
            states,
            actions,
            rewards,
            masks,
            next_states,
        ):
            key, key_next, key_actor = jax.random.split(key, 3)

            next_actions, next_log_probs = _tanh_normal_sample(
                actor,
                actor_params,
                next_states,
                key_next,
                log_std_min,
                log_std_max,
            )
            alpha_val = jnp.exp(log_alpha)

            def q_target_fn(params):
                inputs = jnp.concatenate([next_states, next_actions], -1)
                return q_module.apply(params, inputs)

            next_q_values = jax.vmap(q_target_fn, in_axes=(0,))(q_target_params)
            next_q = jnp.min(next_q_values, axis=0)
            target_q = rewards + discount * masks * (
                next_q - alpha_val * next_log_probs
            )

            def update_q_online(q_params, opt_state):
                def loss_fn(params):
                    inputs = jnp.concatenate([states, actions], -1)
                    q = q_module.apply(params, inputs)
                    critic_loss = ((q - target_q) ** 2).mean()
                    return critic_loss

                q_loss, grad = jax.value_and_grad(loss_fn)(q_params)
                if critic_max_grad_norm is not None:
                    grad = clip_grads(grad, critic_max_grad_norm)
                updates, opt_state = q_opt.update(grad, opt_state, params=q_params)
                q_params = optax.apply_updates(q_params, updates)
                return q_loss, q_params, opt_state, l2_norm(grad)

            q_loss, q_online_params, q_opt_state, q_grad_norm = jax.vmap(
                update_q_online, in_axes=(0, 0)
            )(q_online_params, q_opt_state)

            q_loss = q_loss.mean()
            q_grad_norm = q_grad_norm.mean()

            q_target_params = jax.lax.cond(
                (critic_step % target_update_period) == 0,
                lambda: jax.tree_util.tree_map(
                    lambda p, t: tau * p + (1.0 - tau) * t,
                    q_online_params,
                    q_target_params,
                ),
                lambda: q_target_params,
            )
            critic_step = critic_step + 1

            (actor_loss, log_probs), actor_grads = self._actor_loss_and_grads(
                actor,
                q_module,
                actor_params,
                q_online_params,
                states,
                alpha_val,
                key_actor,
                log_std_min,
                log_std_max,
            )
            if actor_max_grad_norm is not None:
                actor_grads = clip_grads(actor_grads, actor_max_grad_norm)
            actor_updates, actor_opt_state = actor_opt.update(
                actor_grads, actor_opt_state, params=actor_params
            )
            actor_params = optax.apply_updates(actor_params, actor_updates)
            actor_grad_norm = l2_norm(actor_grads)

            def alpha_loss_fn(log_alpha, log_probs):
                alpha = jnp.exp(log_alpha)
                return jnp.mean(alpha * (-log_probs - target_entropy))

            alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(
                log_alpha, jax.lax.stop_gradient(log_probs)
            )
            alpha_updates, alpha_opt_state = alpha_opt.update(
                alpha_grads, alpha_opt_state
            )
            log_alpha = optax.apply_updates(log_alpha, alpha_updates)
            alpha_val = jnp.exp(log_alpha)

            metrics = {
                "q_loss": q_loss,
                "q_grad_norm": q_grad_norm,
                "actor_loss": actor_loss,
                "actor_grad_norm": actor_grad_norm,
                "temperature": alpha_val,
                "temperature_loss": alpha_loss,
            }

            return (
                actor_params,
                actor_opt_state,
                log_alpha,
                alpha_opt_state,
                q_online_params,
                q_target_params,
                q_opt_state,
                critic_step,
                key,
                metrics,
            )

        return _update

    def _make_sample_actions_jit(self):
        actor = self.actor
        log_std_min = self.log_std_min
        log_std_max = self.log_std_max

        @jax.jit
        def _sample(actor_params, states, key):
            return _tanh_normal_sample(
                actor, actor_params, states, key, log_std_min, log_std_max
            )

        return _sample

    def _make_deterministic_actions_jit(self):
        actor = self.actor

        @jax.jit
        def _det(params, states):
            out = actor.apply(params, states)
            means, _ = jnp.split(out, 2, axis=-1)
            return jnp.tanh(means)

        return _det

    def update(self, states, actions, rewards, masks, next_states):

        states = jnp.asarray(states)
        actions = jnp.asarray(actions)
        rewards = jnp.asarray(rewards)
        masks = jnp.asarray(masks)
        next_states = jnp.asarray(next_states)

        (
            self.actor_params,
            self.actor_opt_state,
            self.log_alpha,
            self.alpha_opt_state,
            self.critic.Q_online_params,
            self.critic.Q_target_params,
            self.critic.optimizer_state_q,
            self.critic.step,
            self.key,
            metrics,
        ) = self._fused_update(
            self.actor_params,
            self.actor_opt_state,
            self.log_alpha,
            self.alpha_opt_state,
            self.critic.Q_online_params,
            self.critic.Q_target_params,
            self.critic.optimizer_state_q,
            self.critic.step,
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
            "actor_params": self.actor_params,
            "critic_online_params": self.critic.Q_online_params,
            "critic_target_params": self.critic.Q_target_params,
            "log_alpha": self.log_alpha,
        }
        payload = jax.device_get(payload)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
