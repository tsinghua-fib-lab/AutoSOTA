from __future__ import annotations
from typing import Any, Callable

import jax
import jax.numpy as jnp

import jit_env

from jaxtyping import PRNGKeyArray

from .types import Policy, MetricData

from smz.utils import num_unique_atoms


class EnvStepMixin:
    train: bool
    as_pomdp: bool
    policy: Policy
    env: jit_env.Environment

    def _body(
            self,
            carry,
            x=None
    ):
        # Performs an environment step
        key, state, step, policy_state = carry
        key_carry, key_sample = jax.random.split(key)

        show_state = None if self.as_pomdp else state
        policy_state, action, policy_data = self.policy(
            key_sample, policy_state, step.observation,
            state=show_state, train=self.train
        )
        next_state, next_step = self.env.step(state, action)

        return (
            (key_carry, next_state, next_step, policy_state),
            (next_step, action, policy_data)
        )


class EnvironmentDataGenerator[
    PolicyState, PolicyParams, State, Observation, Action
](EnvStepMixin):
    """Generator to sample from an Agent-Environment space.
    """
    train: bool = True

    def __init__(
            self,
            env: jit_env.Environment[State, Action, Observation, float, float],
            policy: Policy[PolicyParams, PolicyState, Observation, Action],
            batch_size: int | None = None,
            length: int | None = None,
            *,
            options: jit_env.EnvOptions | None = None,
            as_pomdp: bool = True
    ):
        self.env = env
        self.policy = policy

        self.batch_size = batch_size
        self.length = length

        self._batch_fun: Callable[..., ...] | None = None

        self.options = options
        self.as_pomdp = as_pomdp

        self.reset()

    def reset(self):
        # Recompile jitted datafunction (e.g., when changing shape attributes)
        self._batch_fun = jax.jit(jax.vmap(
            self._datafun, in_axes=(0, None, 0)
        ), donate_argnums=2)

    def __repr__(self) -> str:
        return (f'{type(self).__name__}('
                f'task={type(self.env).__name__}, '
                f'policy={type(self.policy).__name__}, '
                f'shape={(self.batch_size, self.length)}, '
                f'identifier={hash(self)})')

    def _datafun(
            self,
            key: PRNGKeyArray,
            policy_params: PolicyParams,
            previous_state: tuple[
                                tuple[State, jit_env.TimeStep],
                                PolicyState,
                                PRNGKeyArray
                            ] | None = None
    ) -> tuple[
        tuple[tuple[State, jit_env.TimeStep], PolicyState, PRNGKeyArray],
        tuple[jit_env.TimeStep, jit_env.Action, dict]
    ]:
        if previous_state is None:
            # Initial call to ._datafun()
            key_carry, key_env = jax.random.split(key)
            init_state, init_step = self.env.reset(
                key_env, options=self.options
            )
            policy_state = self.policy.reset(policy_params)
        else:
            # Continue where left of from a previous ._datafun() call.
            (init_state, init_step), policy_state, key_carry = previous_state
            policy_state = self.policy.update(policy_params, policy_state)

        (
            last_key, last_state, last_step, last_policy_state
        ), (
            steps, actions, policy_data
        ) = jax.lax.scan(
            self._body,
            (key_carry, init_state, init_step, policy_state), None,
            length=self.length
        )

        # Prepend initial step, remove trailing step.
        steps = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([jnp.expand_dims(a, 0), b[:-1]]),
            init_step, steps
        )

        # Carry the state to allow continuing where left-off.
        carry_state = (last_state, last_step), last_policy_state, last_key

        # All dims have prefix: [0, T]
        return carry_state, (steps, actions, policy_data)

    def sample_data(
            self,
            key: PRNGKeyArray,
            policy_params: PolicyParams,
            previous_state: tuple[
                                tuple[State, jit_env.TimeStep],
                                PolicyState,
                                PRNGKeyArray
                            ] | None = None
    ) -> tuple[
        tuple[tuple[State, jit_env.TimeStep], PolicyState, PRNGKeyArray],
        tuple[jit_env.TimeStep, Action, Any]
    ]:
        keys = jax.random.split(key, num=self.batch_size)

        # Note, jax.jit will recompile *once* when `previous_state` is carried,
        # since the initial call will receive `None`. This is OK.
        return self._batch_fun(keys, policy_params, previous_state)


class SimpleEvaluator[
    PolicyState, PolicyParams, State, Observation, Action
](EnvStepMixin):
    train: bool = False

    def __init__(
            self,
            env: jit_env.Environment[State, Action, Observation, float, float],
            policy: Policy[PolicyParams, PolicyState, Observation, Action],
            testing_smcts_policy: Policy[PolicyParams, PolicyState, Observation, Action],
            testing_smc_policy: Policy[PolicyParams, PolicyState, Observation, Action],
            batch_size: int,
            *,
            options: jit_env.EnvOptions | None = None,
            as_pomdp: bool = True,
            metric_fun: Callable[
                            [int, dict[str, Any]], MetricData
                        ] | None = None,
            fixed_seed: int | None = None
    ):
        self.env = env
        self.policy = policy
        self.batch_size = batch_size

        self.metric_fun = metric_fun
        self._batch_fun: Callable[..., ...] | None = None

        self.options = options
        self.as_pomdp = as_pomdp

        # Allow fixing the evaluation key to control randomness.
        self.fixed_seed = fixed_seed

        self.training_datasize = 0
        self.reset()

        self.planner_variance_counter = 0
        self.testing_smcts_policy = testing_smcts_policy
        self.testing_smc_policy = testing_smc_policy

    def reset(self):
        # Recompile jitted datafunction (e.g., when changing shape attributes)
        self._batch_fun = jax.jit(jax.vmap(self._datafun, in_axes=(0, None)))

    def _datafun(
            self,
            key: PRNGKeyArray,
            policy_params: PolicyParams
    ) -> tuple[jax.Array, jax.Array]:

        key_carry, key_env = jax.random.split(key)
        init_state, init_step = self.env.reset(
            key_env, options=self.options
        )
        policy_state = self.policy.reset(policy_params)

        def while_body(x):
            _bool, _inputs, (_returns, _counter) = x
            _key, _state, _step, _pi_state = self._body(_inputs)[0]

            _terminate = (_step.step_type == jit_env.StepType.LAST) | (
                jnp.isclose(_step.discount, 0.0))

            _result = (_returns + _step.reward, _counter + 1)

            return ~_terminate, (_key, _state, _step, _pi_state), _result

        carry = key_carry, init_state, init_step, policy_state
        init_result = (jnp.zeros(()), jnp.zeros(()))

        result = jax.lax.while_loop(
            cond_fun=lambda x: x[0],
            body_fun=while_body,
            init_val=(jnp.asarray(True), carry, init_result),
        )[2]

        return result

    def _variance_datafun(
            self,
            key: PRNGKeyArray,
            policy_params: PolicyParams
    ) -> tuple[jax.Array, jax.Array]:

        key_carry, key_env = jax.random.split(key)

        init_state, init_step = self.env.reset(
            key_env, options=self.options
        )
        policy_state = self.policy.reset(policy_params)

        def while_body(x):
            _bool, _inputs, (_previous_result_sum, _counter) = x

            # Number of parallel runs of the planner, across which we will compute variance
            N = 64
            key, state, step, policy_state = _inputs
            key, batch_key = jax.random.split(key)
            tsmcts_batched_keys = jax.random.split(batch_key, N)

            # Call the main planner, N times
            first_out, second_out = jax.vmap(lambda k: self._body((k, state, step, policy_state)))(tsmcts_batched_keys)
            # All the data is in second_out
            first_out = jax.tree_util.tree_map(lambda x: x[0], first_out)
            (_key, _state, _step, _pi_state), (_batched_next_step, _batched_action, batched_policy_data_tsmcts) = first_out, second_out
            _terminate = (_step.step_type == jit_env.StepType.LAST) | (
                jnp.isclose(_step.discount, 0.0))

            # Call the second planner
            key, smct_batch_key = jax.random.split(key)
            smcts_batched_keys = jax.random.split(smct_batch_key, N)
            _policy_state_smcts, _, batched_policy_data_smcts = jax.vmap(
                lambda k: self.testing_smcts_policy(k, policy_state, step.observation, state=state, train=False))(
                smcts_batched_keys)

            # Call the third planner
            key, smc_batch_key = jax.random.split(key)
            smc_batched_keys = jax.random.split(smc_batch_key, N)
            _policy_state_smc, _, batched_policy_data_smc = jax.vmap(
                lambda k: self.testing_smc_policy(k, policy_state, step.observation, state=state, train=False))(
                smc_batched_keys)
            
            # Compute PMF variance
            average_tsmcts_variance = jnp.nanmean(jnp.nanvar(batched_policy_data_tsmcts['true_pmf'], axis=0))
            average_smcts_variance = jnp.nanmean(jnp.nanvar(batched_policy_data_smcts['true_pmf'], axis=0))
            average_smc_variance = jnp.nanmean(jnp.nanvar(batched_policy_data_smc['true_pmf'], axis=0))
            _pmf_variance_result = (average_tsmcts_variance, average_smcts_variance, average_smc_variance)

            # Compute root value variance and average it per planner
            average_tsmcts_variance = jnp.nanmean(jnp.nanvar(batched_policy_data_tsmcts['true_root_values'], axis=0))
            average_smcts_variance = jnp.nanmean(jnp.nanvar(batched_policy_data_smcts['true_root_values'], axis=0))
            average_smc_variance = jnp.nanmean(jnp.nanvar(batched_policy_data_smc['true_root_values'], axis=0))
            _root_value_variance_result = (average_tsmcts_variance, average_smcts_variance, average_smc_variance)

            # Compute average number of unique actions at each root
            all_atoms = self.policy.planner.proposal.policy_model.enumerate_atoms()
            num_unique_atoms_tsmcts = jax.vmap(lambda atoms: num_unique_atoms(all_atoms, atoms))(
                batched_policy_data_tsmcts['final_root_atoms']).mean()
            num_unique_atoms_smcts = jax.vmap(lambda atoms: num_unique_atoms(all_atoms, atoms))(
                batched_policy_data_smcts['final_root_atoms']).mean()
            num_unique_atoms_smc = jax.vmap(lambda atoms: num_unique_atoms(all_atoms, atoms))(
                batched_policy_data_smc['final_root_atoms']).mean()
            _num_actions_result = (num_unique_atoms_tsmcts, num_unique_atoms_smcts, num_unique_atoms_smc)

            # Accumulate outputs, expected shape = [3, 3]
            _result_sum = _previous_result_sum + jnp.array((_pmf_variance_result, _root_value_variance_result, _num_actions_result))

            return ~_terminate, (_key, _state, _step, _pi_state), (_result_sum, _counter + 1)

        carry = key_carry, init_state, init_step, policy_state
        init_result = (jnp.zeros((3, 3)), jnp.zeros(()))

        result = jax.lax.while_loop(
            cond_fun=lambda x: x[0],
            body_fun=while_body,
            init_val=(jnp.asarray(True), carry, init_result),
        )[2]

        return result[0] / result[1], result[1]

    def __call__(
            self,
            step: int,
            data_batch_size: int,
            key: PRNGKeyArray,
            params: PolicyParams
    ) -> MetricData:

        key = key if self.fixed_seed is None else \
            jax.random.key(self.fixed_seed)

        key, key_sample = jax.random.split(key)

        total_interactions = (step + 1) * data_batch_size

        keys = jax.random.split(key, num=self.batch_size)
        returns, episode_lengths = self._batch_fun(keys, params)

        # Setup all the variance params
        average_pmf_variance_tsmcts = None
        average_pmf_variance_smcts = None
        average_pmf_variance_smc = None

        average_root_value_variance_tsmcts = None
        average_root_value_variance_smcts = None
        average_root_value_variance_smc = None
        
        average_action_num_tsmcts = None
        average_action_num_smcts = None
        average_action_num_smc = None

        if (total_interactions > self.planner_variance_counter * 200000
                and self.testing_smcts_policy is not None
                and self.testing_smc_policy is not None
                and not self.policy.planner.proposal.policy_model.continuous):
            self.planner_variance_counter += 1
            results, episode_lengths = jax.jit(jax.vmap(self._variance_datafun, in_axes=(0, None)))(keys, params)
            # results is of shape [3, 3]
            average_pmf_variance_tsmcts = results[:, 0, 0].mean()
            average_pmf_variance_smcts = results[:, 0, 1].mean()
            average_pmf_variance_smc = results[:, 0, 2].mean()
            average_root_value_variance_tsmcts = results[:, 1, 0].mean()
            average_root_value_variance_smcts = results[:, 1, 1].mean()
            average_root_value_variance_smc = results[:, 1, 2].mean()
            average_action_num_tsmcts = results[:, 2, 0].mean()
            average_action_num_smcts = results[:, 2, 1].mean()
            average_action_num_smc = results[:, 2, 2].mean()

        return self.metric_fun(
            step, {
                'returns': returns,
                'episode_lengths': episode_lengths,
                'data_size': self.training_datasize
            }
        ) | {
                'eval_data/average_pmf_variance_tsmcts': average_pmf_variance_tsmcts,
                'eval_data/average_pmf_variance_smcts': average_pmf_variance_smcts,
                'eval_data/average_pmf_variance_smc': average_pmf_variance_smc,
                'eval_data/average_root_value_variance_tsmcts': average_root_value_variance_tsmcts,
                'eval_data/average_root_value_variance_smcts': average_root_value_variance_smcts,
                'eval_data/average_root_value_variance_smc': average_root_value_variance_smc,
                'eval_data/average_action_num_tsmcts': average_action_num_tsmcts,
                'eval_data/average_action_num_smcts': average_action_num_smcts,
                'eval_data/average_action_num_smc': average_action_num_smc,
             }
