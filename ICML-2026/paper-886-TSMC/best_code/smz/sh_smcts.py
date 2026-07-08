"""Extends the SMCTS with Sequential Halving

"""
from typing import Protocol

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray, Integer, ArrayLike, Num, ScalarLike

from smz.smc import SMC, SMCParams, ParticleData

from smz import utils

from copy import copy

import optax

from typing import Literal


class SHSMC[State, Observation, Action]:
    """Implements an iterated version of SMC, with iterations decided by a SH process.

    Performs multiple SMC planning calls to adjust model parameters locally
    based on intermediate samples. Any updated parameters are thrown away.
    """
    def __init__(
            self,
            planner: SMC[State, Action, ..., ..., ..., ...],
            num_actions_to_search: int,
            discount: ScalarLike,
            td_lambda: ScalarLike,
            option: Literal['dirac', 'value', 'sumprod'],
            value_mixing: float = 0.5,
            stochastic_eval: bool = True,
            use_completed_q_values: bool = True,
            use_q_transform: bool = True,
    ):
        """Constructor for IteratedSMC

        Parameters
        ----------
        smc : SMC
            Base sequential Monte-Carlo planner to use
        update_function : ParameterUpdater
            Function that updates model parameters locally
        num_iterations : int
            Number of combined SMC and parameter-update iterations
            If set to `0`, this class is equivalent to `SMC`.
        """
        self.planner = planner

        from experiment.offline.impl.smc import SMCPolicy
        self.policy_for_testing = SMCPolicy(
            planner,
            discount=discount,
            td_lambda=td_lambda,
            option=option,
            value_mixing=value_mixing,
            stochastic_eval=stochastic_eval,
        )

        # Identify whether action space is cont. or not
        self.continuous = self.planner.proposal.policy_model.continuous
        self.num_actions_to_search = int(min(num_actions_to_search, self.planner.num_particles))
        if not self.continuous:
            self.num_actions_to_search = int(min(self.num_actions_to_search, self.planner.proposal.policy_model.output_size))

        self.action_space_size = int(self.planner.proposal.policy_model.output_size)
        self.original_num_actions_to_search = int(num_actions_to_search)

        # Compute sh-depth:
        self.num_iterations = int(jnp.floor(jnp.log2(self.num_actions_to_search).item()))
        self.sh_depth = int(self.planner.depth // self.num_iterations)

        # If sh_depth < 1, then we can do sh_depth = 1, num_iterations = min(total depth, num_iterations)
        if self.sh_depth < 1:
            self.sh_depth = int(1)
            self.num_iterations = int(min(self.num_iterations, self.planner.depth))

        self.stochastic_eval = stochastic_eval
        self.use_completed_q_values = use_completed_q_values
        self.use_q_transform = use_q_transform

        self.sh_smcs = []
        num_particles_per_iteration = self.planner.num_particles // self.num_actions_to_search

        if num_particles_per_iteration < 1:
            raise ValueError(f"The number of iterations of SH was less than one, means that depth budget was too small")

        # This is used to complete the depth budget in case it doesn't divide evenly
        total_depth = 0
        for x in range(self.num_iterations):
            smc_copy = copy(self.planner)
            # If we're at the last iteration, use all the remaining depth
            if x == self.num_iterations - 1 and total_depth < self.planner.depth:
                current_depth = self.planner.depth - total_depth
            else:
                current_depth = self.sh_depth
            smc_copy.depth = current_depth
            smc_copy.num_particles = num_particles_per_iteration
            smc_policy_reinitialized = SMCPolicy(
                smc_copy,
                discount=discount,
                td_lambda=td_lambda,
                option=option,
                value_mixing=value_mixing,
                stochastic_eval=stochastic_eval,
            )
            self.sh_smcs.append(smc_policy_reinitialized)
            num_particles_per_iteration = num_particles_per_iteration * 2
            total_depth += current_depth

    def __str__(self) -> str:
        return (f"{type(self).__name__}(N={self.num_iterations}, "
                f"SMC={str(self.planner)})")

    def _complete_qvalues(self, qvalues, visit_counts, value):
        """Returns completed Q-values, with the `value` for unvisited actions."""

        # The missing qvalues are replaced by the value.
        completed_qvalues = jnp.where(
          visit_counts > 0,
          qvalues,
          value)
        return completed_qvalues

    def _compute_mixed_value(self, raw_value, qvalues, visit_counts, prior_probs):
        """Interpolates the raw_value and weighted qvalues.

        Args:
          raw_value: an approximate value of the state. Shape `[]`.
          qvalues: Q-values for all actions. Shape `[num_actions]`. The unvisited
            actions have undefined Q-value.
          visit_counts: the visit counts for all actions. Shape `[num_actions]`.
          prior_probs: the action probabilities, produced by the policy network for
            each action. Shape `[num_actions]`.

        Returns:
          An estimator of the state value. Shape `[]`.
        """
        sum_visit_counts = jnp.sum(visit_counts, axis=-1)
        # Ensuring non-nan weighted_q, even if the visited actions have zero
        # prior probability.
        prior_probs = jnp.maximum(jnp.finfo(prior_probs.dtype).tiny, prior_probs)
        # Summing the probabilities of the visited actions.
        sum_probs = jnp.sum(jnp.where(visit_counts > 0, prior_probs, 0.0),
                            axis=-1)
        weighted_q = jnp.sum(jnp.where(
            visit_counts > 0,
            prior_probs * qvalues / jnp.where(visit_counts > 0, sum_probs, 1.0),
            0.0), axis=-1)
        return (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1)

    def _rescale_qvalues(self, qvalues, epsilon):
        """Rescales the given completed Q-values to be from the [0, 1] interval."""
        min_value = jnp.min(qvalues, axis=-1, keepdims=True)
        max_value = jnp.max(qvalues, axis=-1, keepdims=True)
        return (qvalues - min_value) / jnp.maximum(max_value - min_value, epsilon)

    def qtransform_completed_by_mix_value(
            self,
            qvalues,
            visit_counts,
            raw_value,
            prior_logits,
            *,
            value_scale = 0.1,
            maxvisit_init = 50.0,
            rescale_values = True,
            use_mixed_value = True,
            epsilon = 1e-8,
    ):
        """Returns completed qvalues.

        The missing Q-values of the unvisited actions are replaced by the
        mixed value, defined in Appendix D of
        "Policy improvement by planning with Gumbel":
        https://openreview.net/forum?id=bERaNdoegnO

        The Q-values are transformed by a linear transformation:
          `(maxvisit_init + max(visit_counts)) * value_scale * qvalues`.

        Args:
          tree: _unbatched_ MCTS tree state.
          node_index: scalar index of the parent node.
          value_scale: scale for the Q-values.
          maxvisit_init: offset to the `max(visit_counts)` in the scaling factor.
          rescale_values: if True, scale the qvalues by `1 / (max_q - min_q)`.
          use_mixed_value: if True, complete the Q-values with mixed value,
            otherwise complete the Q-values with the raw value.
          epsilon: the minimum denominator when using `rescale_values`.

        Returns:
          Completed Q-values. Shape `[num_actions]`.
        """
        # Computing the mixed value and producing completed_qvalues.
        prior_probs = jax.nn.softmax(prior_logits)
        if use_mixed_value:
            value = self._compute_mixed_value(
                raw_value,
                qvalues=qvalues,
                visit_counts=visit_counts,
                prior_probs=prior_probs)
        else:
            value = raw_value
        completed_qvalues = self._complete_qvalues(
            qvalues, visit_counts=visit_counts, value=value)

        # Scaling the Q-values.
        rescaled_qvalues = completed_qvalues
        if rescale_values:
            rescaled_qvalues = self._rescale_qvalues(completed_qvalues, epsilon)
        maxvisit = jnp.max(visit_counts, axis=-1)
        visit_scale = maxvisit_init + maxvisit
        return visit_scale * value_scale * rescaled_qvalues, completed_qvalues

    def __call__(
            self,
            key: PRNGKeyArray,
            policy_state: SMCParams,
            obs: Observation,
            state: State | None = None,
            train: bool = True
    ) -> tuple[SMCParams, Action, dict[str, jax.Array]]:
        """
            Executed SH on m actions at the root
        """
        ## Instantiate keys
        key, key_model, key_sampling = jax.random.split(key, 3)

        ## Get all logits
        policy_out = self.planner.proposal.policy_model.apply(
            policy_state.proposal.policy_params,  # Is this the correct params?
            state.observation,
            rngs={'default': key_model}
        )

        if self.continuous:
            starting_actions_to_search = self.planner.proposal.policy_model.sample(
                key_sampling, (self.num_actions_to_search,), policy_out
            )
            # When sampling from prior, logits are uniform
            logits_mapped = jnp.zeros(self.num_actions_to_search)
            indexes_of_starting_actions_w_respect_to_all_actions = jnp.arange(self.num_actions_to_search)
            all_true_actions = starting_actions_to_search
        else:
            # Add gumbel noise
            gumbel_noise = jax.random.gumbel(key_sampling, (policy_out.size,))
            logits = policy_out + gumbel_noise
            _, indexes_of_starting_actions_w_respect_to_all_actions = jax.lax.top_k(logits,
                                                                     self.num_actions_to_search
                                                                     )
            # Get all actions
            all_true_actions = self.planner.proposal.policy_model.enumerate_atoms()
            indexes_of_all_true_actions = jnp.arange(self.action_space_size)
            starting_actions_to_search = all_true_actions[indexes_of_starting_actions_w_respect_to_all_actions]
            logits_mapped = logits[indexes_of_starting_actions_w_respect_to_all_actions]

        # Get next states for all actions
        # Split the key for sampling next states
        key, key_batch = jax.random.split(key)
        key_batch = jax.random.split(key_batch, self.num_actions_to_search)

        # States are wrapped with key and rank for deterministic resampling, need to do and undo
        cache = rank = None
        state = (state, rank, cache)

        # Sorted jumbled, according to starting_actions_to_search
        m_next_states, m_rewards, m_gammas = jax.vmap(self.planner.transition, in_axes=(None, 0, None, 0))(
            policy_state.transition, key_batch, state, starting_actions_to_search
        )
        (m_next_states, _rank, _cache) = m_next_states
        indexes_of_current_actions_to_search = jnp.arange(self.num_actions_to_search)
        current_num_actions_to_search = self.num_actions_to_search
        mapped_values_of_children = jnp.zeros(self.num_actions_to_search)
        Q_sum = jnp.zeros(self.num_actions_to_search)
        Q_normalizer = jnp.zeros(self.num_actions_to_search)

        for index, smc_policy in enumerate(self.sh_smcs):
            indexed_states = jax.tree_util.tree_map(lambda x: x[indexes_of_current_actions_to_search], m_next_states)

            # Call the planner individually for each next state
            key, key_batch = jax.random.split(key)
            key_batch = jax.random.split(key_batch, current_num_actions_to_search)
            _batched_policy_state, _batched_action, batched_data = jax.vmap(smc_policy, in_axes=
            (0, None, None, 0, None))(
                key_batch, policy_state, obs, indexed_states, train
            )

            # Extract values, stitch back to root
            mapped_values_of_children = mapped_values_of_children.at[indexes_of_current_actions_to_search].set(batched_data["value"])

            # This is of size starting_actions_to_search, indexed like starting_actions_to_search ([0, ..., N])
            # This is still organized in the jumbled order of indexes_of_starting_actions_to_search
            Q_values_of_searched_root_actions = (m_rewards
                                                 + m_gammas
                                                 * mapped_values_of_children)  # Let's ignore discount for now
            Q_sum = Q_sum.at[indexes_of_current_actions_to_search].add(
                Q_values_of_searched_root_actions[indexes_of_current_actions_to_search] * smc_policy.planner.num_particles
            )
            Q_normalizer = Q_normalizer.at[indexes_of_current_actions_to_search].add(smc_policy.planner.num_particles)
            Q_sh = Q_sum / Q_normalizer.clip(min=1)

            # Compute improved policy
            current_logits = Q_sh / policy_state.target.root_temperature + logits_mapped

            # Update the number of actions to search
            current_num_actions_to_search = current_num_actions_to_search // 2
            _, indexes_of_current_actions_to_search = jax.lax.top_k(current_logits, current_num_actions_to_search)

        # Compute target policy to act from
        pmf = jax.nn.softmax(Q_sh / policy_state.target.root_temperature + logits_mapped)

        # Return improved policy value, action
        key, key_choice = jax.random.split(key)
        if self.stochastic_eval or train:
            idx = jax.random.choice(
                key_choice,
                a=pmf.shape[0],
                shape=(),
                p=pmf,
            )
        else:
            idx = jnp.argmax(pmf)

        # Get the true action to act with in env.:
        true_idx = indexes_of_starting_actions_w_respect_to_all_actions[idx]
        action = all_true_actions[true_idx]

        # Get the Q value and value for targets
        q_value = Q_sh[idx]
        value = pmf @ Q_sh
        true_pmf = pmf
        true_root_values = Q_sh
        atoms = starting_actions_to_search

        if not self.continuous:
            # Use the full GMZ q transform, which uses the completed q values, to compute the policy target:
            if self.use_q_transform:
                key, value_key = jax.random.split(key)
                raw_root_value = self.planner.target(policy_state.target, value_key, state)
                all_qs = jnp.zeros(self.action_space_size).at[indexes_of_starting_actions_w_respect_to_all_actions].set(Q_sh)
                all_visit_counts = jnp.zeros(self.action_space_size).at[indexes_of_starting_actions_w_respect_to_all_actions].set(Q_normalizer)
                transformed_qs, completed_qs = self.qtransform_completed_by_mix_value(all_qs, all_visit_counts, raw_root_value,
                                                                           logits,
                                                                           value_scale=1/policy_state.target.root_temperature)
                pmf = jax.nn.softmax(transformed_qs + logits)
                true_pmf = pmf
                atoms =  all_true_actions
                true_root_values = completed_qs
            # Use the GMZs completed q values to compute the policy target:
            elif self.use_completed_q_values:
                key, value_key = jax.random.split(key)
                raw_root_value = self.planner.target(policy_state.target, value_key, state)
                all_qs = jnp.zeros(self.action_space_size).at[indexes_of_starting_actions_w_respect_to_all_actions].set(
                    Q_sh)
                all_visit_counts = jnp.zeros(self.action_space_size).at[
                    indexes_of_starting_actions_w_respect_to_all_actions].set(Q_normalizer)
                _, completed_qs = self.qtransform_completed_by_mix_value(all_qs, all_visit_counts, raw_root_value,
                                                                         logits,
                                                                         value_scale=1/policy_state.target.root_temperature)
                pmf = jax.nn.softmax(completed_qs / policy_state.target.root_temperature + logits)
                true_pmf = pmf
                atoms = all_true_actions
                true_root_values = completed_qs
            # Return the mapped true pmf for pmf variance logging
            else:
                true_pmf = utils.compute_pmf(
                    true_actions=all_true_actions, atoms=starting_actions_to_search, mass=pmf
                )
                true_root_values = utils.compute_true_root_values(
                    true_actions=all_true_actions, atoms=starting_actions_to_search, root_values=Q_sh
                )

        # Compute statistics
        data = {'atoms': atoms, 'pmf': pmf, 'value': value, 'q_value': q_value,
                'true_pmf': true_pmf, 'true_root_values': true_root_values, 'final_root_atoms': atoms,
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
