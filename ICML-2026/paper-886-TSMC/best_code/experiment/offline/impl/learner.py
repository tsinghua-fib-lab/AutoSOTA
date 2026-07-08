from typing import NamedTuple, Any

import jax
import jax.numpy as jnp

import optax
import rlax

import flashbax as fbx

import jit_env

from jaxtyping import PRNGKeyArray, ScalarLike

from experiment import models


class EMLearnerHyperparameters(NamedTuple):
    # Data storage constraints
    prioritized: bool
    priority_exponent: float

    max_age_buffer: int
    min_length_buffer: int

    # Credit assignment hyperparameters
    td_lambda: float
    discount: float
    sarsa: bool

    # Loss hyperparameters
    policy_coeff: float
    value_coeff: float
    sample_entropy_coeff: float
    exact_entropy_coeff: float

    # Stochastic gradient descent hyperparameters
    num_steps: int
    batch_size: int


class EMLearner[Observation, Action]:

    def __init__(
            self,
            optimizer: optax.GradientTransformation,
            model: models.JointModel,
            dummy_input: tuple[Observation, Action],
            hyperparameters: EMLearnerHyperparameters,
            verbose: bool = True
    ):
        self.optimizer = optimizer
        self.model = model
        self.dummy_input = dummy_input

        self.param = hyperparameters

        self.buffer: fbx.trajectory_buffer.TrajectoryBuffer | None = None
        self.verbose = verbose

    def set_buffer(self, buffer: fbx.trajectory_buffer.TrajectoryBuffer):
        self.buffer = buffer

    def preprocess_data(
            self,
            steps: jit_env.TimeStep,
            actions: Action,
            policy_data: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:

        truncated = (steps.step_type != jit_env.StepType.MID) & (
                steps.discount > 0.0
        )
        # Use either V(s) or Q(s, a), both should come from a planner (E-step)
        target_values = policy_data['q_value'] \
            if self.param.sarsa else policy_data['value']

        # Compute target-values for steps 0...T-1.
        returns = jax.vmap(rlax.lambda_returns)(
            *jax.tree_util.tree_map(lambda x: x[:, 1:], (
                steps.reward,
                steps.discount * self.param.discount,
                target_values,
                (1 - truncated) * self.param.td_lambda
            ))
        )
        # Append target-value for step T (no reward or discount available).
        returns = jnp.concatenate(
            [returns, jnp.expand_dims(policy_data['value'][:, -1], 1)], 1
        )

        return {
            'obs': steps.observation,
            'actions': actions,
            'value_target': returns,
            'policy_target': (policy_data['atoms'], policy_data['pmf']),
            'true_pmf': policy_data.get('true_pmf', None)
        }

    def init(self, key: PRNGKeyArray) -> tuple[optax.OptState, optax.Params]:
        params = self.model.init(key, *self.dummy_input)
        opt_state = self.optimizer.init(params)

        if self.verbose:
            num_params = jax.tree.reduce(
                jnp.add, jax.tree_util.tree_map(jnp.size, params)
            )
            print(f'Initialized model parameters of size: {num_params:,}.')

        return opt_state, params

    def update(
            self,
            state: optax.OptState,
            variables: optax.Params,
            key: PRNGKeyArray,
            data: fbx.trajectory_buffer.BufferState
    ) -> tuple[tuple[optax.OptState, optax.Params], dict[str, jax.Array]]:
        """Implements a generalized M-step for RL expectation-maximization

        We include entropy regularization on the SMC-sampled atoms.
        This is done by reweighting the atoms as `w_i (1 - c) + c`
        which is equivalent to a renormalized version of:
            CE(SMC-dist, policy-dist) + H(policy_dist)
        where CE is the cross-entropy and H is the entropy. Our method
        is equivalent since the SMC-dist is originally sampled from
        policy_dist inside the SMC (or MCTS) method.
        """
        if self.buffer is None:
            raise RuntimeError("Learner has no buffer reference!")

        def sample_loss(
                _key: PRNGKeyArray,
                _model_params: optax.Params,
                _inputs: tuple[Any, Any],
                _targets: tuple[tuple[jax.Array, jax.Array], ScalarLike]
        ) -> tuple[ScalarLike, dict[str, ScalarLike]]:
            """Calculate the per-transition loss"""

            out: dict[str, Any] = self.model.apply(
                _model_params, *_inputs, rngs={'default': _key}
            )

            (atoms, pmf), value = _targets

            lp_q = self.model.q_value.logprob(out['q_value'], value)
            lp_v = self.model.value.logprob(out['value'], value)

            lps_pi = jax.vmap(
                self.model.policy.logprob, in_axes=(None, 0)
            )(out['policy'], atoms).squeeze()

            # Entropy regularization of the sampled atoms
            weights = (pmf * (1 - self.param.sample_entropy_coeff) +
                       jnp.ones(pmf.size) / pmf.size *
                       self.param.sample_entropy_coeff)

            # Note for logging that entropy is upper-bounded by the
            # cardinality of the random-variable. This means that the
            # policy_entropy can differ in scale from the target entropy
            # when constructing atoms through prior bootstrapping.
            policy_entropy = self.model.policy.entropy(out['policy'])
            target_entropy = -jnp.sum(pmf * jnp.clip(jnp.log(pmf), -1e3))

            l_pi, l_v, l_q = -weights @ lps_pi, -lp_v, -lp_q
            L = self.param.policy_coeff * l_pi + \
                self.param.value_coeff * (l_v + l_q)

            return L - self.param.exact_entropy_coeff * policy_entropy, {
                'loss': L - self.param.exact_entropy_coeff * policy_entropy,
                'loss_no_entropy': L,
                'policy_entropy': policy_entropy,
                'target_entropy': target_entropy,
                'policy_ece': l_pi, 'value_nll': l_v, 'q_value_nll': l_q,
                'value_targets': value
            }

        def loss(
                _key: PRNGKeyArray,
                _model_params: optax.Params,
                _inputs: tuple[Any, Any],
                _targets: tuple[tuple[jax.Array, jax.Array], ScalarLike]
        ) -> tuple[ScalarLike, dict[str, ScalarLike]]:
            """Calculate the loss over a batch of transitions"""
            key_batch = jax.random.split(_key, self.param.batch_size)
            batch_losses, metrics = jax.vmap(
                sample_loss, in_axes=(0, None, 0, 0)
            )(key_batch, _model_params, _inputs, _targets)

            mean_loss, mean_metrics = jax.tree_util.tree_map(
                jnp.mean, (batch_losses, metrics)
            )

            return mean_loss, mean_metrics

        def update_fun(
                _key: PRNGKeyArray,
                _model_params: optax.Params,
                _opt_state: optax.OptState,
                _data: tuple[
                       tuple[Any, Any],
                       tuple[tuple[jax.Array, jax.Array], ScalarLike]
                ]
        ) -> tuple[dict[str, ScalarLike], optax.Params, optax.OptState]:

            grad, _metrics = jax.grad(loss, argnums=1, has_aux=True)(
                _key, _model_params, *_data
            )

            updates, _opt_state = self.optimizer.update(
                grad, _opt_state, _model_params
            )
            _model_params = optax.apply_updates(_model_params, updates)

            return _metrics, _model_params, _opt_state

        def body(carry, x):
            key_carry, _opt_state, _vars, _bufferdata = carry

            key_carry, key_sample, key_loss = jax.random.split(key_carry, 3)
            sample = self.buffer.sample(_bufferdata, key_sample)

            # We don't need the transition (second) as we precompute targets
            first = sample.experience.first

            # Unpack sampled data for loss functions
            ins = first['obs'], first['actions']
            targets = first['policy_target'], first['value_target']

            _metrics, _vars, _opt_state = update_fun(
                key_loss, _vars, _opt_state, (ins, targets)
            )

            true_target_pmfs = first['true_pmf']  # (batch, dim)
            if true_target_pmfs is not None:
                true_prob_var = jnp.var(true_target_pmfs, axis=0).mean()
                _metrics = _metrics | {
                    'policy_target_pmf_variance': true_prob_var
                }

            _metrics = _metrics | {'l2': optax.tree_utils.tree_l2_norm(_vars)}

            return (key_carry, _opt_state, _vars, _bufferdata), _metrics

        (_, state, variables, _), batch_metrics = jax.lax.scan(
            body, (key, state, variables, data), None,
            length=self.param.num_steps
        )

        aggr_metrics = jax.tree_util.tree_map(jnp.mean, batch_metrics)
        return_metrics = {f'loss/{k}': v for k, v in aggr_metrics.items()}

        return (state, variables), return_metrics


class PPOLearnerHyperparameters(NamedTuple):

    # Credit assignment hyperparameters
    td_lambda: float
    discount: float

    # Loss hyperparameters
    policy_coeff: float
    value_coeff: float
    entropy_coeff: float
    clip_epsilon: float

    # Stochastic gradient descent hyperparameters
    num_steps: int
    batch_size: int

    # Default replay buffer settings. API PARITY; DO NOT TOUCH.
    prioritized: bool = False
    max_age_buffer: int = 1  # Throw away all old policy data
    min_length_buffer: int = 1


class PPOLearner[Observation, Action]:

    def __init__(
            self,
            optimizer: optax.GradientTransformation,
            model: models.JointModel,
            dummy_input: tuple[Observation, Action],
            hyperparameters: PPOLearnerHyperparameters,
            verbose: bool = True
    ):
        self.optimizer = optimizer
        self.model = model
        self.dummy_input = dummy_input

        self.param = hyperparameters

        # PPO still uses a "replay buffer" for minibatching only
        # This is a circular buffer with maximum-policy age = 1.
        self.buffer: fbx.trajectory_buffer.TrajectoryBuffer | None = None
        self.verbose = verbose

    def set_buffer(self, buffer: fbx.trajectory_buffer.TrajectoryBuffer):
        self.buffer = buffer

    def preprocess_data(
            self,
            steps: jit_env.TimeStep,
            actions: Action,
            policy_data: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        # Precompute all necessary learning targets beforehand

        truncated = (steps.step_type != jit_env.StepType.MID) & (
                steps.discount > 0.0
        )

        # Drop 0th step
        r_1t, d_1t, l_1t, v_1t = jax.tree_util.tree_map(lambda x: x[:, 1:], (
            steps.reward,
            steps.discount * self.param.discount,
            (1 - truncated) * self.param.td_lambda,
            policy_data['value']
        ))

        # Compute GAE for the policy update
        gae = jax.vmap(rlax.truncated_generalized_advantage_estimation)(
            r_1t,  # reward: [1, T]
            d_1t,  # discount: [1, T]
            l_1t,  # lambda: [1, T]
            policy_data['value']  # value: [0, T]
        )

        # Compute target-values for steps 0...T-1 for the value update
        returns = jax.vmap(rlax.lambda_returns)(
            r_1t,  # reward: [1, T]
            d_1t,  # discount: [1, T]
            v_1t,  # value: [1, T]
            l_1t  # lambda: [1, T]
        )

        # Truncate last step since we have no transition data
        return jax.tree_util.tree_map(lambda x: x[:, :-1], {
            'obs': steps.observation,
            'actions': actions,
            'log_prob_old': policy_data['log_prob'],
        }) | {
            'value_target': returns,
            'advantages': gae
        }

    def init(self, key: PRNGKeyArray) -> tuple[optax.OptState, optax.Params]:
        params = self.model.init(key, *self.dummy_input)
        opt_state = self.optimizer.init(params)

        if self.verbose:
            num_params = jax.tree.reduce(
                jnp.add, jax.tree_util.tree_map(jnp.size, params)
            )
            print(f'Initialized model parameters of size: {num_params:,}.')

        return opt_state, params

    def update(
            self,
            state: optax.OptState,
            variables: optax.Params,
            key: PRNGKeyArray,
            data: fbx.trajectory_buffer.BufferState
    ) -> tuple[tuple[optax.OptState, optax.Params], dict[str, jax.Array]]:
        """Implements the PPO update

        """
        if self.buffer is None:
            raise RuntimeError("Learner has no buffer reference!")

        def sample_loss(
                _key: PRNGKeyArray,
                _model_params: optax.Params,
                _inputs: tuple[Any, Any],
                _targets: tuple[tuple[jax.Array, jax.Array], ScalarLike]
        ) -> tuple[ScalarLike, dict[str, ScalarLike]]:
            """Calculate the per-transition loss"""

            _, _action = _inputs
            out: dict[str, Any] = self.model.apply(
                _model_params, *_inputs, rngs={'default': _key}
            )

            (log_prob_old, advantage), value = _targets

            # Value loss
            lp_v = self.model.value.logprob(out['value'], value)

            # Clipped policy-gradient loss
            log_prob = self.model.policy.logprob(out['policy'], _action)
            ratio = jnp.exp(log_prob - log_prob_old)

            l_pi = rlax.clipped_surrogate_pg_loss(
                jnp.atleast_1d(ratio), jnp.atleast_1d(advantage),
                epsilon=self.param.clip_epsilon
            ).squeeze()

            # Compute complete PPO loss
            L = self.param.policy_coeff * l_pi - self.param.value_coeff * lp_v
            policy_entropy = self.model.policy.entropy(out['policy'])

            return L - self.param.entropy_coeff * policy_entropy, {
                'loss': L - self.param.entropy_coeff * policy_entropy,
                'loss_no_entropy': L,
                'policy_entropy': policy_entropy,
                'ratio': ratio,
                'policy_loss': l_pi,
                'value_nll': -lp_v
            }

        def loss(
                _key: PRNGKeyArray,
                _model_params: optax.Params,
                _inputs: tuple[Any, Any],
                _targets: tuple[tuple[jax.Array, jax.Array], ScalarLike]
        ) -> tuple[ScalarLike, dict[str, ScalarLike]]:
            """Calculate the loss over a batch of transitions"""
            key_batch = jax.random.split(_key, self.param.batch_size)
            batch_losses, metrics = jax.vmap(
                sample_loss, in_axes=(0, None, 0, 0)
            )(key_batch, _model_params, _inputs, _targets)

            return jax.tree_util.tree_map(jnp.mean, (batch_losses, metrics))

        def update_fun(
                _key: PRNGKeyArray,
                _model_params: optax.Params,
                _opt_state: optax.OptState,
                _data: tuple[
                       tuple[Any, Any],
                       tuple[tuple[jax.Array, jax.Array], ScalarLike]
                ]
        ) -> tuple[dict[str, ScalarLike], optax.Params, optax.OptState]:

            grad, _metrics = jax.grad(loss, argnums=1, has_aux=True)(
                _key, _model_params, *_data
            )

            updates, _opt_state = self.optimizer.update(
                grad, _opt_state, _model_params
            )
            _model_params = optax.apply_updates(_model_params, updates)

            return _metrics, _model_params, _opt_state

        def body(carry, x):
            key_carry, _opt_state, _vars, _bufferdata = carry

            key_carry, key_sample, key_loss = jax.random.split(key_carry, 3)
            sample = self.buffer.sample(_bufferdata, key_sample)

            # We don't need the transition (second) as we precompute targets
            first = sample.experience.first

            # Unpack sampled data for loss functions
            ins = first['obs'], first['actions']
            targets = (
                (first['log_prob_old'], first['advantages']),  # policy-target
                first['value_target']  # value-target
            )

            _metrics, _vars, _opt_state = update_fun(
                key_loss, _vars, _opt_state, (ins, targets)
            )

            _metrics = _metrics | {'l2': optax.tree_utils.tree_l2_norm(_vars)}

            return (key_carry, _opt_state, _vars, _bufferdata), _metrics

        (_, state, variables, _), batch_metrics = jax.lax.scan(
            body, (key, state, variables, data), None,
            length=self.param.num_steps
        )

        aggr_metrics = jax.tree_util.tree_map(jnp.mean, batch_metrics)
        return_metrics = {f'loss/{k}': v for k, v in aggr_metrics.items()}

        return (state, variables), return_metrics

