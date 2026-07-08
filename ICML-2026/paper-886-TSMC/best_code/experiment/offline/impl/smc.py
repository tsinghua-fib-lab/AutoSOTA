from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp

import optax

import jit_env

import flax.typing as ft
from jaxtyping import PRNGKeyArray, ScalarLike

from smz import SMC, SMCParams, ParticleData, planning, utils


class JitEnvSMCTransition[State, Action]:
    def __init__(self, env: jit_env.Environment):
        self.env = env

    def __call__(
            self,
            params: None,
            key: PRNGKeyArray,
            state: State,
            action: Action
    ) -> tuple[State, ScalarLike, ScalarLike]:
        state, step = self.env.step(state, action)
        return state, step.reward, step.discount


class JitEnvSMCTransitionWithPRNGKey[State, Action]:
    def __init__(self, env: jit_env.Environment):
        self.env = env

    def __call__(
            self,
            params: None,
            key: PRNGKeyArray,
            state: State,
            action: Action
    ) -> tuple[State, ScalarLike, ScalarLike]:
        (state, rank, carry_key) = state

        state, step = self.env.step(state, action)

        return (state, rank, key), step.reward, step.discount


class LinenTRPIProposal[State, Observation, Action](
    planning.impl.TRPIProposal[
        State, Action,
        planning.impl.TRPIParams[ft.VariableDict, ft.VariableDict]
    ]
):
    """Extension to `TRPIProposal` to wrap parametrization of the SMC-proposal.

    This class helps with:
     - initialization with flax modules.
    """

    def format_params(
            self,
            params: ft.VariableDict
    ) -> planning.impl.TRPIParams[ft.VariableDict, ft.VariableDict]:
        # Format/ separate joint-parameter container into expected container

        return planning.impl.TRPIParams(
            policy_params={'params': params['params']['policy']},
            value_params={'params': params['params']['q_value']},
            constraint_param=self.base_constraint_param
        )


class LinenELBOTarget[State, Observation, Action](
    planning.impl.ELBOTarget[
        State, planning.impl.ELBOTargetParams[ft.VariableDict]
    ]
):
    """Extension to `ELBOTarget` to wrap parametrization of the SMC-target.

    This class helps with:
     - initialization with flax modules.
    """

    def format_params(
            self, params: ft.VariableDict
    ) -> planning.impl.ELBOTargetParams[ft.VariableDict]:
        # Format/ separate joint-parameter container into expected container

        return planning.impl.ELBOTargetParams(
            value_params={'params': params['params']['value']},
            temperature=self.temperature,
            root_temperature=self.root_temperature,
            advantage_softmax=self.advantage_softmax,
        )


class SMCPolicy[State, Observation, Action]:

    def __init__(
            self,
            planner: SMC[State, Action, ..., ..., ..., ...],
            discount: ScalarLike,
            td_lambda: ScalarLike,
            option: Literal['dirac', 'value', 'sumprod'],
            value_mixing: float = 0.5,
            stochastic_eval: bool = True
    ):
        if planner.return_logits:
            raise RuntimeError(
                "Ensure that planner returns normalized probabilities! "
                "Set arg `return_logits` in SMC to `False`!"
            )

        legal = {'dirac', 'value', 'sumprod', 'trace', 'dsmc'}
        if option not in legal:
            raise NotImplementedError(
                f"Selected option {option} is not supported. "
                f"Choose from {legal}."
            )

        self.planner = planner
        self.discount = discount
        self.td_lambda = td_lambda

        self.option = option
        self.value_mixing = value_mixing
        self.stochastic_eval = stochastic_eval  # TODO; implement

    def make_behaviour_policy(
            self,
            params: SMCParams,
            atoms: jax.Array,
            pmf: jax.Array,
            smc_data: ParticleData,
            value_tree: jax.Array
    ):
        if self.option == 'dirac':
            # Returns the pmf as given by SMC. Can suffer from degeneracy.
            return pmf

        elif self.option == 'value':
            # Recompute the target-operator for depth = 1 using the improved
            # value estimations from tree TD-lambda.

            # Note, `value_next` is a backed-up value from tree Retrace.
            _, logits = self.planner.target.log_weights(
                params.target,
                log_w=jnp.zeros_like(smc_data.log_prior[0]),
                data=ParticleData(
                    state=None, next_state=None, action=None,
                    log_prior=smc_data.log_prior[0],
                    log_proposal=smc_data.log_proposal[0],
                    reward=jnp.zeros_like(smc_data.reward[0]),
                    discount=jnp.ones_like(smc_data.discount[0]),
                    value_next=value_tree[0],  # V(s) = R + gamma * V(s_next)
                    value_t=value_tree[0] @ pmf  # (logits are shift-invariant)
                )
            )

            return jnp.exp(logits)

        elif self.option == 'sumprod':
            # Recompute the target-operator over the entire trajectory.

            # Compute the per-particle log-weights (without recursive backups)
            log_w, _ = jax.vmap(
                partial(self.planner.target.log_weights, params.target)
            )(
                log_w=jnp.zeros_like(smc_data.log_prior),  # No backup value
                data=smc_data
            )

            # Backup log-weights recursively using (log-)belief propagation.
            logprobs = planning.tree_truncated_sumprod(
                smc_data.state.observation,
                smc_data.next_state.observation,
                log_w
            )

            # Normalize for resampling
            return jax.nn.softmax(logprobs[0])

    def __call__(
            self,
            key: PRNGKeyArray,
            policy_state: SMCParams,
            obs: Observation,
            state: State | None = None,
            train: bool = True
    ) -> tuple[SMCParams, Action, dict[str, jax.Array]]:
        key_credit, key_planner, key_action = jax.random.split(key, 3)

        (atoms, smc_pmf), (trace, stats) = self.planner.run(
            key_planner, policy_state, state
        )

        if self.option == 'dsmc':
            # TODO: Add a proper Q-transform (currently is just a temp).

            # TODO: The below is only a halfway solution. When stitching values back together at the root the problem
            #  needs to be addressed inside the planner, because the updates need to be weighted with respect to each other.
            # The return of dsmc is of shape (num particles), and may have duplicate branches per action at the root
            QT_with_possible_duplicates = trace.data['dsmc']
            final_root_atoms = atoms

            # Aggregate possible duplicate correctly
            if self.planner.proposal.policy_model.continuous:
                # If space is cont., there are no duplicates, and everything is fine.
                QT = QT_with_possible_duplicates
                pmf = jax.nn.softmax(QT / policy_state.target.root_temperature)
            else:
                # If space is discrete, we need to stitch the QT with duplicates back to the original action space
                true_actions = self.planner.proposal.policy_model.enumerate_atoms()
                QT, QT_mask = utils.compute_values_from_branches(
                    true_actions=true_actions, atoms=atoms, values=QT_with_possible_duplicates
                )
                # We mask away all the unvisited actions
                # TODO: Needs to be made safe softmax with mask
                logits = jnp.where(QT_mask, QT / policy_state.target.root_temperature, -jnp.inf)
                pmf = jax.nn.softmax(logits)
                atoms = true_actions

            if self.stochastic_eval or train:
                idx = jax.random.choice(
                    key_action, a=pmf.shape[0], shape=(), p=pmf
                )
            else:
                idx = jnp.argmax(pmf)
            action = atoms.at[idx].get()

            # Only get the values at root weighed by the policy (expected SARSA)
            root_values = QT
            value = QT @ pmf
            q_value = QT.at[idx].get()

            # TODO: I can add completed Q values the way it's implemented in SHSMCTS.
            #  Doesn't seem beneficial from experiments there.

        elif self.option == 'trace':
            pmf = jax.nn.softmax(trace.data['logweight'])
            if self.stochastic_eval or train:
                idx = jax.random.choice(
                    key_action, a=pmf.shape[0], shape=(), p=pmf
                )
            else:
                idx = jnp.argmax(pmf)
            action = atoms.at[idx].get()

            # Retrace operator: RQ = Q + E[sum_t gamma^t (prod_s^t c_s) err_t ]
            root_values = (stats.value_t[0] + trace.data['retrace']) * self.value_mixing + (
                    1.0 - self.value_mixing) * stats.value_t[0]

            # Only get the values at root weighed by the policy (expected SARSA)
            value = root_values @ pmf
            q_value = root_values.at[idx].get()
            final_root_atoms = atoms[trace.branches]

        else:
            if isinstance(stats.next_state, tuple):
                # Reuse cached keys inside state to expand identical 'atoms' for
                # duplicated states from resampling.
                next_state, _rank, _cache = stats.next_state
            else:
                next_state = stats.next_state
            if isinstance(stats.state, tuple):
                # Reuse cached keys inside state to expand identical 'atoms' for
                # duplicated states from resampling.
                state, _rank, _cache = stats.state
            else:
                state = stats.state

            smc_value_tree = planning.tree_truncated_retrace(
                state.observation, next_state.observation,
                stats.reward, stats.discount, stats.value_next,
                log_is=stats.log_prior - stats.log_proposal,
                gamma=self.discount, lambda_=self.td_lambda
            )
            # smc_value_tree = planning.tree_truncated_retrace(
            #     stats.state.observation, stats.next_state.observation,
            #     stats.reward, stats.discount, stats.value_next,
            #     log_is=stats.log_prior - stats.log_proposal,
            #     gamma=self.discount, lambda_=self.td_lambda
            # )

            # Behaviour policy construction
            pmf = self.make_behaviour_policy(
                policy_state, atoms, smc_pmf, stats, smc_value_tree
            )

            if self.stochastic_eval or train:
                idx = jax.random.choice(
                    key_action, a=pmf.shape[0], shape=(), p=pmf
                )
            else:
                idx = jnp.argmax(pmf)

            action = atoms.at[idx].get()

            root_values = smc_value_tree[0] * self.value_mixing + (
                    1.0 - self.value_mixing) * stats.value_t[0]

            # Only get the values at root weighed by the policy (expected SARSA)
            value = root_values @ pmf
            q_value = root_values.at[idx].get()
            final_root_atoms = atoms[trace.branches]

        true_pmf = pmf
        true_root_values = root_values
        if not self.planner.proposal.policy_model.continuous:
            true_actions = self.planner.proposal.policy_model.enumerate_atoms()
            true_pmf = utils.compute_pmf(
                true_actions=true_actions, atoms=atoms, mass=pmf
            )
            true_root_values = utils.compute_true_root_values(
                true_actions=true_actions, atoms=atoms, root_values=root_values
            )

        data = {'atoms': atoms, 'pmf': pmf, 'value': value, 'q_value': q_value,
                'true_pmf': true_pmf, 'true_root_values': true_root_values, 'final_root_atoms': final_root_atoms,
                }

        return policy_state, action, data

    def reset(self, params: optax.Params) -> SMCParams:
        # Format `params` container into a compatible SMC-container

        if hasattr(self.planner.proposal, 'format_params'):
            prop_params = self.planner.proposal.format_params(params)
        else:
            raise NotImplementedError("Could not format proposal params")

        if hasattr(self.planner.target, 'format_params'):
            target_params = self.planner.target.format_params(params)
        else:
            raise NotImplementedError("Could not format target params")

        return self.planner.make_params(
            proposal=prop_params,
            transition=None,
            target=target_params
        )

    def update(self, params: optax.Params, state: SMCParams) -> SMCParams:
        return self.reset(params)  # Simply throw away previous state

    def unpack(self, policy_state: SMCParams) -> optax.Params:
        # Unused
        return {
            k: v for k, v in policy_state.proposal.items()
            if 'constraint_param' != k
        } | {
            k: v for k, v in policy_state.target.items()
            if 'temperature' != k
        }
