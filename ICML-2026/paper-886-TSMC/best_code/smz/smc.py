"""Implements a base sequential Monte-Carlo planner for reinforcement learning
"""
from __future__ import annotations
from typing import Protocol, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray, Num, Integer, Float, ArrayLike, ScalarLike

ROOT_DEPTH: int = 1


class ParticleData[State, Action](NamedTuple):
    """Defines all data carried by our SMC particles"""
    state: State
    next_state: State
    action: Action
    reward: ScalarLike
    discount: ScalarLike
    log_prior: ScalarLike
    log_proposal: ScalarLike
    value_next: ScalarLike
    value_t: ScalarLike


class Trace[E, T](NamedTuple):
    branches: Integer[jax.Array, ' num_particles']
    log_weights: Float[jax.Array, ' num_particles']

    memory: E | None
    data: T | None


class WrappedState[State](NamedTuple):
    """Namespace to carry a state and a reference state for resampling"""
    state: State
    reference: State


class SMCParams[ProposalParams, TransitionParams, TargetParams](NamedTuple):
    """Namespace to organize model parameters for each SMC-component"""
    proposal: ProposalParams | None = None
    transition: TransitionParams | None = None
    target: TargetParams | None = None


class Proposal[State, Action, Params](Protocol):
    """Bound on the proposal distribution"""

    def __call__(
            self,
            params: Params,
            key: PRNGKeyArray,
            state: State,
            level: Integer[ArrayLike, ''],
    ) -> tuple[Action, ScalarLike, ScalarLike]:
        """Sample an action at the current state and return its likelihood

        Parameters
        ----------
        params : Params
            Parameters of the proposal distribution
        key : PRNGKeyArray
            Jax based random key
        state : State
            Current state of the hidden Markov model
        level : Integer
            Current depth (level) of the planner

        Returns
        -------
        Action
            The action to be performed in Transition given state
        float
            The log probability of action under the prior (base model)
        float
            The log probability of action under the implemented proposal
        """
        ...


class Transition[State, Action, Params](Protocol):
    """Bound on the implementation of state transitions"""

    def __call__(
            self,
            params: Params,
            key: PRNGKeyArray,
            state: State,
            action: Action
    ) -> tuple[State, ScalarLike, ScalarLike]:
        """Sample a state transition

        Parameters
        ----------
        params : Params
            Parameters of the transition distribution
        key : PRNGKeyArray
            Jax based random key
        state : State
            Current state of the hidden Markov model
        action : Action
            Input to the transition function to update `state`

        Returns
        -------
        State
            The model's state after the transition for (state, action).
        float
            The log-potential of the transition (i.e., reward)
        float
            Continuation probability (i.e., discount)

        Notes
        -----
        For terminal states in the Markov chain, ensure that the model doesn't
        automatically restart but infinitely stays in the same absorbing state
        with zero rewards and zero continuation probability. If this is not
        correctly implemented, this can artificially lower or increase the
        likelihood of trajectories inside the SMC-planner.
        """
        ...


class Target[State, Params](Protocol):
    """Bound on the implementation of a target policy improvement operator"""

    def __call__(
            self,
            params: Params,
            key: PRNGKeyArray,
            state: State
    ) -> ScalarLike:
        """Compute the log of the smoothing distribution (future potential)

        Parameters
        ----------
        params : Params
            Parameters of the smoothing distribution
        key : PRNGKeyArray
            Jax based random key
        state : State
            Current state of the hidden Markov model

        Returns
        -------
        float
            (Estimated) log-prob. of the smoothing distribution at `state`
        """
        ...

    def log_weights(
            self,
            params: Params,
            log_w: ScalarLike,
            data: ParticleData
    ) -> tuple[ScalarLike, ScalarLike]:
        """Compute the (recursive) log likelihood of the target operator.


        Returns
        -------
        float
            Log-probability of the target to recursively accumulate
        float
            Transformed log-probability (to be used for resampling)
        """
        ...


class StatisticAccumulator[T, E, Statistic](Protocol):

    def __init__(self, *args, **kwargs):
        pass

    def init(self, num_particles: int, depth: int) -> Trace[T, E]:
        pass

    def update(
            self,
            trace: Trace[T, E],
            data: ParticleData,
            new_branches: Integer[jax.Array, ' num_particles'],
            old_log_weights: Float[jax.Array, ' num_particles'],
            logits: Float[jax.Array, ' num_particles'],
            new_log_weights: Float[jax.Array, ' num_particles']
    ) -> tuple[Trace[T, E], Statistic]:
        pass


class NoAccumulation[T, E]:

    def __init__(self, return_data: bool = False):
        self.return_data = return_data

    def init(self, num_particles: int, depth: int) -> Trace[T, E]:
        branches = jnp.arange(0, num_particles)
        log_weights = jnp.zeros(num_particles)

        return Trace(branches, log_weights, None, None)

    def update(
            self,
            trace: Trace[T, E],
            data: ParticleData,
            new_branches: Integer[jax.Array, ' num_particles'],
            old_log_weights: Float[jax.Array, ' num_particles'],
            logits: Float[jax.Array, ' num_particles'],
            new_log_weights: Float[jax.Array, ' num_particles']
    ) -> tuple[Trace[T, E], ParticleData | None]:
        data = data if self.return_data else None
        return Trace(new_branches, new_log_weights, None, None), data


class Resampler(Protocol):
    """Defines function signature of a categorical resampling method.
    """

    def __call__(
            self,
            key: PRNGKeyArray,
            logits: Float[jax.Array, ' num_particles']
    ) -> Integer[jax.Array, ' num_particles']:
        """Bootstrap array indices from `logits` to keep in the next SMC step.

        I.e., sampling should be done with replacement to duplicate indices
        with high probability under the target operator.

        Parameters
        ----------
        key : PRNGKeyArray
            Jax based random key
        logits : jax.Array of floats of length `num_particles`
            Array of log probabilities to resample indices from

        Returns
        -------
        jax.Array of integers of length `num_particles`
            Array of indices of the same shape as `logits` that indicate which
            particles are kept (duplicated/ dropped) for the SMC-algortithm.
        """
        ...


def multinomial_resampling(
        key: PRNGKeyArray,
        logits: Float[jax.Array, ' num_particles']
) -> Integer[jax.Array, ' num_particles']:
    """Default multinomial resampling for sequential Monte-Carlo"""
    return jax.random.categorical(key, logits, shape=(logits.size,))


def deterministic_resampling(
        key: PRNGKeyArray,
        logits: Float[jax.Array, ' num_particles']
) -> Integer[jax.Array, ' num_particles']:
    """Default multinomial resampling for sequential Monte-Carlo"""
    pmf = jnp.exp(logits)
    cmf = jnp.cumsum(pmf)

    bins = jnp.linspace(0, 1, logits.size)
    indices = jnp.argmax(bins[:, None] <= cmf, axis=1)

    return indices


class SMC[
    State, Action,
    TargetStatistic,
    ProposalParams, TransitionParams, TargetParams
]:
    """Implements a sequential Monte-Carlo algorithm for control planning.

    Partial observability and learned models are supported through functional
    implementation of `State` and properly adapting the transition model.

    Adaptation of the SMC-algorithm by:
     - Piche et al., Probabilistic planning with sequential Monte-Carlo
       methods. 2018. https://openreview.net/forum?id=ByetGn0cYX

    Includes a heuristic extension to prevent terminal (absorbing) states
    from being dropped by the resampling method: we do this by resampling
    the last observed non-terminal state in a sampled sequence.
    """

    def __init__(
            self,
            proposal: Proposal[State, Action, ProposalParams],
            transition: Transition[State, Action, TransitionParams],
            target: Target[State, TargetParams],
            *,
            depth: int,
            num_particles: int,
            resampling_period: int | None = 1,
            resampling_method: Resampler = multinomial_resampling,
            statistic_fun: StatisticAccumulator = NoAccumulation(),
            return_logits: bool = False,
            prevent_particle_death: bool = True
    ):
        """Constructor for SMC

        Parameters
        ----------
        proposal : object
            Bounded by Proposal for importance-sampling of actions
        transition : object
            Bounded by Transition for state-transitions
        target : object
            Bounded by Target for computing the SMC target
        depth : int
            The exact depth that the search method reaches
        num_particles : int
            The number of samples to search with
        resampling_period : int or None (default = 1)
            Period (1/frequency) parameter to indicate resampling at every
            `resampling_period` steps. If None, no resampling is done.
        resampling_method : Callable (default = jax.random.categorical)
            (Sampling) Function that returns the indices of particles to keep
        statistic_fun : Statistic
        return_logits : bool (default = False)
            Whether to return logits when calling SMC.run()
        """
        self.proposal = proposal
        self.transition = transition
        self.target = target

        self.depth = depth
        self.num_particles = num_particles

        self.resampling_period = (
            (depth + 1) if resampling_period is None else resampling_period
        )
        self.resampling_method = resampling_method

        # Assumes temporal separability over the trajectory
        self.statistic_fun = statistic_fun

        self.return_logits = return_logits
        self.prevent_particle_death = prevent_particle_death

    def __str__(self) -> str:
        return f"{type(self).__name__}(h={self.depth}, N={self.num_particles})"

    @property
    def root_depth(self) -> int:
        """Base index (level) as a reference"""
        return ROOT_DEPTH

    @property
    def budget(self) -> int:
        """Number of transitions computed by SMC at each call to `run`.

        """
        return self.depth * self.num_particles

    @staticmethod
    def make_params(*args, **kwargs) -> SMCParams[
        ProposalParams, TransitionParams, TargetParams
    ]:
        """Helper method inside SMC namespace for constructing param-container
        """
        return SMCParams(*args, **kwargs)

    @partial(jax.vmap, in_axes=(None, 0, None, 0, None))
    def _step_particle(
            self,
            key: PRNGKeyArray,
            params: SMCParams[ProposalParams, TransitionParams, TargetParams],
            state: State,
            level: Integer[ArrayLike, '']
    ) -> tuple[ParticleData, ScalarLike]:
        """Particle update: Samples an environment transition"""
        key_act, key_env, key_prop = jax.random.split(key, 3)

        # Query the proposal model (policy)
        key_action_model, key_action_sample = jax.random.split(key_act)
        action, log_prior, log_proposal = self.proposal(
            params.proposal, key_action_model, state, level
        )

        # Sample from the transition model (environment)
        next_state, reward, gamma = self.transition(
            params.transition, key_env, state, action
        )

        # Do 1-step message passing with a target model
        key_t, key_next = jax.random.split(key_prop)
        value_t = self.target(params.target, key_t, state)
        value_next = self.target(params.target, key_next, next_state)

        # Bookkeeping
        data = ParticleData(
            state=state, next_state=next_state, action=action,
            reward=reward, discount=gamma,
            log_prior=log_prior, log_proposal=log_proposal,
            value_next=value_next, value_t=value_t
        )

        return data, jnp.isclose(gamma, 0.0)  # gamma == 0 -> termination

    def _resample(
            self,
            key: PRNGKeyArray,
            states: WrappedState[State],
            logits: Float[jax.Array, ' num_particles'],
            branches: Integer[jax.Array, ' num_particles'],
    ) -> tuple[WrappedState[State], Integer[jax.Array, ' num_particles']]:
        """Resample particles according to the target logits

        States are resampled to their `reference` in `WrappedState` to prevent
        dropping trajectories in absorbing states but with high rewards.
        """
        idx = self.resampling_method(key, logits)

        # Keep track of which root-branch particle `i` belongs to
        states_resampled = jax.tree_util.tree_map(
            lambda x: x.at[idx].get(), states.reference
        )
        root_branches_resampled = branches.at[idx].get()

        def compute_rank(carry, xs):
            _counter, _rank = carry
            idx_root, idx_particle = xs

            _rank = _rank.at[idx_particle].set(_counter.at[idx_root].get())
            _counter = _counter.at[idx_root].add(1)

            return (_counter, _rank), (_counter, _rank)

        counter_init = jnp.zeros(root_branches_resampled.size, int)
        rank_init = jnp.zeros(root_branches_resampled.size, int)
        pos = jnp.arange(root_branches_resampled.size)

        ranks = jax.lax.scan(compute_rank, init=(counter_init, rank_init),
                             xs=(root_branches_resampled, pos))[0][1]

        real_states, old_ranks, old_cache = states_resampled
        states_resampled = (real_states, ranks, old_cache)

        wrapped_resampled = WrappedState(states_resampled, states_resampled)

        return wrapped_resampled, root_branches_resampled

    def _bodyfun(
            self,
            carry: tuple[
                PRNGKeyArray,
                SMCParams[ProposalParams, TransitionParams, TargetParams],
                WrappedState[State],
                Trace
            ],
            xs: Integer[jax.Array, '']
    ) -> tuple[
        tuple[
            PRNGKeyArray,
            SMCParams[ProposalParams, TransitionParams, TargetParams],
            WrappedState[State],
            Trace
        ],
        tuple[Action, TargetStatistic]
    ]:
        """Main SMC body, implements planner step for jax.lax.scan"""

        key, params, wrapped, trace = carry
        key_carry, key_update, key_resample = jax.random.split(key, 3)

        # Batch state transition over all particles
        key_batch = jax.random.split(key_update, self.num_particles)

        data, done = self._step_particle(key_batch, params, wrapped.state, xs)

        # Carry the last non-terminal state previous steps separately
        reference_state = data.next_state
        if self.prevent_particle_death:
            reference_state = jax.vmap(
                lambda flag, *args: jax.lax.cond(
                    flag,
                    lambda x, y: x,
                    lambda x, y: y,
                    *args
                )
            )(done, wrapped.reference, data.next_state)
        new_wrapped_state = WrappedState(data.next_state, reference_state)

        # Update weights using the target operator and get resampling logits
        log_w, logits = self.target.log_weights(
            params=params.target, log_w=trace.log_weights, data=data
        )

        # Resample periodically according to the weights
        do_resample = xs % self.resampling_period == 0
        wrapped_next, branches = jax.lax.cond(
            do_resample,
            self._resample,
            lambda *args: (new_wrapped_state, trace.branches),
            key_resample,
            new_wrapped_state,
            logits,
            trace.branches,
        )
        # Bootstrap the weights if resampled
        new_log_w = jax.lax.select(do_resample, jnp.zeros_like(log_w), log_w)

        # Update bookkeeping and statistic accumulation
        new_trace, out = self.statistic_fun.update(
            trace, data, branches, log_w, logits, new_log_w
        )

        return (key_carry, params, wrapped_next, new_trace), (data.action, out)

    def run(
            self,
            key: PRNGKeyArray,
            params: SMCParams[ProposalParams, TransitionParams, TargetParams],
            state: State
    ) -> tuple[
        tuple[
            Num[jax.Array, ' num_particles'],
            Float[jax.Array, ' num_particles']
        ], tuple[Trace, TargetStatistic]
    ]:
        """Run the SMC planner at a given state.

        Returns a sample (bootstrapped) distribution with normalized weights,
        along with sampled statistics as given by self.statistic_fun.

        Parameters
        ----------
        key : PRNGKeyArray
            Jax based random key
        params : SMCParams[ProposalParams, TransitionParams, TargetParams]
            Parameter container for the proposal, transition, and target
        state : State
            Current state of the hidden Markov model to start planning at

        Returns
        -------
        tuple of jax.Array
            The first array are the atoms of the bootstrapped/ sampled
            distribution over the actions. The second array are the normalized
            weights over these atoms (in log-space if `return_logits=True`).
        TargetStatistic object
            Some generic value, statistic, or data collected during planning.
        """
        key_planner, key_cache = jax.random.split(key)

        # Initialize all particles with the same state
        state = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (self.num_particles, *jnp.shape(x))),
            state
        )

        # Assume (state, rank, key) format for state;
        # Label the new state-batch with their duplicate-state counter
        state_batch = (state, jnp.arange(self.num_particles),
                       jax.random.split(key_cache, num=self.num_particles))

        wrapped = WrappedState(state_batch, state_batch)

        # Keep track/ bookkeeping of particle statistics
        trace = self.statistic_fun.init(self.num_particles, self.depth)

        # Run the SMC planner from t = 1 to T = depth + 1.
        (*_, final_trace), (actions, stats) = jax.lax.scan(
            self._bodyfun,
            (key_planner, params, wrapped, trace),
            xs=jnp.arange(1, self.depth + 1, dtype=jnp.int16),
        )

        # Note, we only need the actions at index zero.
        root_actions = jax.tree_util.tree_map(lambda x: x.at[0].get(), actions)

        # Normalize and correct (for resampling) the IS-weights
        log_weights = jax.nn.log_softmax(final_trace.log_weights)
        if self.resampling_period > self.depth:
            # No resampling; -> sequential importance sampling
            return_weights = log_weights if self.return_logits \
                else jnp.exp(log_weights)
            return (root_actions, return_weights), (final_trace, stats)

        # Count the root-branch occupancy of the particles
        one_hot = jax.nn.one_hot(final_trace.branches, self.num_particles)
        bincount = one_hot.T @ jnp.ones_like(log_weights)  # == jnp.bincount

        # Combine log-weights only for root-actions with non-zero occupancy
        aggregated = jax.vmap(jax.nn.logsumexp, in_axes=(None, None, 1))(
            log_weights, None, one_hot
        )
        logits = jnp.where(
            bincount > 0, aggregated, jnp.full_like(aggregated, -jnp.inf)
        )

        # The actions present at the root and their final particle mass
        return_weights = logits if self.return_logits else jnp.exp(logits)
        return (root_actions, return_weights), (final_trace, stats)
