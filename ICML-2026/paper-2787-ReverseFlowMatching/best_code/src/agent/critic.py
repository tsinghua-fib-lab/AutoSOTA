import functools
from copy import deepcopy

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from src.components.mlp import MLP
from src.utils.grad_utils import l2_norm, clip_grads


@functools.partial(jax.jit, static_argnames=("tau"))
def soft_update_target_Q(online_params, target_params, tau):
    return jax.tree_util.tree_map(
        lambda p, p_target: p * tau + p_target * (1.0 - tau),
        online_params,
        target_params,
    )


@functools.partial(
    jax.jit, static_argnames=("q_module", "optimizer", "discount", "max_grad_norm")
)
def update_q(
    q_module,
    q_online_params,
    q_target_params,
    states,
    actions,
    rewards,
    masks,
    next_states,
    next_actions,
    optimizer,
    optimizer_state,
    discount,
    max_grad_norm,
):
    def q_target(params):
        inputs = jnp.concatenate([next_states, next_actions], -1)
        return q_module.apply(params, inputs)

    next_q_values = jax.vmap(q_target, in_axes=(0,))(q_target_params)
    next_q = jnp.min(next_q_values, axis=0)

    target_q = rewards + discount * masks * next_q

    def update_q_online(q_params, opt_state):
        def loss_fn(params):
            inputs = jnp.concatenate([states, actions], -1)
            q = q_module.apply(params, inputs)
            critic_loss = ((q - target_q) ** 2).mean()
            return critic_loss

        q_loss, grad = jax.value_and_grad(loss_fn)(q_params)
        if max_grad_norm is not None:
            grad = clip_grads(grad, max_grad_norm)
        updates, opt_state = optimizer.update(grad, opt_state, params=q_params)
        q_params = optax.apply_updates(q_params, updates)
        return q_loss, q_params, opt_state, l2_norm(grad)

    loss, q_online_params, optimizer_state, q_grad_norm = jax.vmap(
        update_q_online, in_axes=(0, 0)
    )(q_online_params, optimizer_state)

    return loss.mean(), q_online_params, optimizer_state, q_grad_norm.mean()


@functools.partial(
    jax.jit, static_argnames=("q_module", "optimizer", "discount", "max_grad_norm")
)
def update_q_sac(
    q_module,
    q_online_params,
    q_target_params,
    states,
    actions,
    rewards,
    masks,
    next_states,
    next_actions,
    next_log_probs,
    optimizer,
    optimizer_state,
    discount,
    alpha,
    max_grad_norm,
):
    def q_target(params):
        inputs = jnp.concatenate([next_states, next_actions], -1)
        return q_module.apply(params, inputs)

    next_q_values = jax.vmap(q_target, in_axes=(0,))(q_target_params)
    next_q = jnp.min(next_q_values, axis=0)
    target_q = rewards + discount * masks * (next_q - alpha * next_log_probs)

    def update_q_online(q_params, opt_state):
        def loss_fn(params):
            inputs = jnp.concatenate([states, actions], -1)
            q = q_module.apply(params, inputs)
            critic_loss = ((q - target_q) ** 2).mean()
            return critic_loss

        q_loss, grad = jax.value_and_grad(loss_fn)(q_params)
        if max_grad_norm is not None:
            grad = clip_grads(grad, max_grad_norm)
        updates, opt_state = optimizer.update(grad, opt_state, params=q_params)
        q_params = optax.apply_updates(q_params, updates)
        return q_loss, q_params, opt_state, l2_norm(grad)

    loss, q_online_params, optimizer_state, q_grad_norm = jax.vmap(
        update_q_online, in_axes=(0, 0)
    )(q_online_params, optimizer_state)

    return loss.mean(), q_online_params, optimizer_state, q_grad_norm.mean()


@functools.partial(jax.jit, static_argnames=("module", "optimizer", "input_shape"))
def init_fn(module, optimizer, input_shape, rng):
    dummy_input = jnp.ones((1, *input_shape))
    params = module.init(rng, dummy_input)
    opt_state = optimizer.init(params)
    return opt_state, params


@functools.partial(jax.jit, static_argnames=("q_module"))
def get_q_values(q_module, q_online_params, states, actions):
    inputs = jnp.concatenate([states, actions], -1)

    def online_q(params):
        q = q_module.apply(params, inputs)
        return q

    q_values = jax.vmap(online_q, in_axes=(0,))(q_online_params)
    return jnp.min(q_values, axis=0)


class ReturnFunction(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_size,
        hidden_layers,
        num_q,
        discount,
        tau,
        target_update_period=1,
        lr=1e-3,
        max_grad_norm=None,
        seed=0,
        rng=None,
    ):
        self.key = rng if rng is not None else jax.random.PRNGKey(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_q = num_q
        self.target_entropy = -action_dim / 2

        self.discount = discount
        self.tau = tau
        self.target_update_period = target_update_period
        self.max_grad_norm = max_grad_norm

        keys = jax.random.split(self.key, num_q)
        self.Q_module = MLP(hidden_size, hidden_layers)
        self.optimizer_q = optax.adam(lr)
        ensemble_init_fn = jax.vmap(
            init_fn, in_axes=(None, None, None, 0), out_axes=(0, 0)
        )
        self.optimizer_state_q, self.Q_online_params = ensemble_init_fn(
            self.Q_module, self.optimizer_q, (state_dim + action_dim,), keys
        )
        self.Q_target_params = deepcopy(self.Q_online_params)

        self.step = 0

    def get_params(self):
        return {
            "Q_online_params": self.Q_online_params,
        }

    def apply(self, params, states, actions):
        q_params = params["Q_online_params"]
        return get_q_values(self.Q_module, q_params, states, actions)

    def update(self, states, actions, rewards, masks, next_states, next_actions):
        self.step += 1

        q_loss, self.Q_online_params, self.optimizer_state_q, q_grad_norm = update_q(
            self.Q_module,
            self.Q_online_params,
            self.Q_target_params,
            states,
            actions,
            rewards,
            masks,
            next_states,
            next_actions,
            self.optimizer_q,
            self.optimizer_state_q,
            self.discount,
            self.max_grad_norm,
        )

        if self.step % self.target_update_period == 0:
            self.Q_target_params = soft_update_target_Q(
                self.Q_online_params, self.Q_target_params, self.tau
            )

        return {
            "q_loss": q_loss,
            "q_grad_norm": q_grad_norm,
        }


class SACCritic(object):
    """Critic ensemble for SAC with entropy-regularized targets."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_size,
        hidden_layers,
        num_q,
        discount,
        tau,
        target_update_period=1,
        lr=1e-3,
        max_grad_norm=None,
        seed=0,
        rng=None,
    ):
        self.key = rng if rng is not None else jax.random.PRNGKey(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_q = num_q

        self.discount = discount
        self.tau = tau
        self.target_update_period = target_update_period
        self.max_grad_norm = max_grad_norm

        keys = jax.random.split(self.key, num_q)
        self.Q_module = MLP(hidden_size, hidden_layers)
        self.optimizer_q = optax.adam(lr)
        ensemble_init_fn = jax.vmap(
            init_fn, in_axes=(None, None, None, 0), out_axes=(0, 0)
        )
        self.optimizer_state_q, self.Q_online_params = ensemble_init_fn(
            self.Q_module, self.optimizer_q, (state_dim + action_dim,), keys
        )
        self.Q_target_params = deepcopy(self.Q_online_params)

        self.step = 0

    def get_values(self, states, actions):
        return get_q_values(self.Q_module, self.Q_online_params, states, actions)

    def update(
        self,
        states,
        actions,
        rewards,
        masks,
        next_states,
        next_actions,
        next_log_probs,
        alpha,
    ):
        self.step += 1

        q_loss, self.Q_online_params, self.optimizer_state_q, q_grad_norm = (
            update_q_sac(
                self.Q_module,
                self.Q_online_params,
                self.Q_target_params,
                states,
                actions,
                rewards,
                masks,
                next_states,
                next_actions,
                next_log_probs,
                self.optimizer_q,
                self.optimizer_state_q,
                self.discount,
                alpha,
                self.max_grad_norm,
            )
        )

        if self.step % self.target_update_period == 0:
            self.Q_target_params = soft_update_target_Q(
                self.Q_online_params, self.Q_target_params, self.tau
            )

        return {
            "q_loss": q_loss,
            "q_grad_norm": q_grad_norm,
        }
