"""This module implements the key components needed for the SMC planners"""
from typing import Sequence, Callable, Protocol, NamedTuple
from typing_extensions import Self
from functools import partial

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray, ScalarLike, PyTree, Bool

from smz.smc import ROOT_DEPTH, Trace, ParticleData, StatisticAccumulator
from smz.utils import normalizers as norm
from smz.utils import tree_compare

from . import proposals


def tree_truncated_sumprod[State](
        s_t: State,  # States ([0, T - 1], B, ...)
        s_next: State,  # True next States: ([1, T], B, ...)
        log_weights: jax.Array,  # Target log-likelihoods ([1, T], B, D)
        *,
        stop_target_gradients: bool = False,
        _equality_fun: Callable[[State, State], bool] = tree_compare
) -> jax.Array:
    """Compute from a partially ordered trajectory-batch the soft-values.

    All non-keyword arguments to this function should be prefixed with
    (time, batch, ...) where the `time` axis is assumed to be ordered.

    This function stitches together the paths that get dropped/ duplicated
    in SMC. Dropped paths bootstrap from the canonical next-state (i.e., the
    next state, if the particle had not been dropped), duplicated paths
    average over the future particle lambda-returns.

    To construct the adjacency matrix, the `s_t` and `s_next` states must
    be comparable through the provided `_equality_fun` argument.

    Assuming the `log_weights` are generated using the SMC planner, this
    function computes the Monte-Carlo estimation of the posterior soft-value.
    This is similar to the `tree_truncated_td_lambda(...)` estimator, except
    that the soft-value is treated as a log-probability. This means that the
    particle log-likelihoods are averaged in terms of their PDF and not in
    terms of their log-PDFs.

    Example aggregation 2 particles,
     - tree_truncated_td_lambda: V(particle_1) / 2 + V(particle_2) / 2
     - tree_truncated_sumprod: ln [ 1/2 * exp log_weight(particle_1) +
                                        1/2 * exp log_weight(particle_2) ]

    I.e., the `tree_truncated_sumprod` uses a `logsumexp` aggregation for
    combining particle log-likelihoods. Hence, the name soft-value, since
    this resulting distribution belongs to the Boltzmann distributions (also
    often referred to as `softmax` distributions).

    Parameters
    ----------
    s_t : State
        Some state object to compare to
    s_next : State
        Some next_state object after `s_t` to compare to
    log_weights: jax.Array
        A per-timestep log-likelihood of some target distribution.
    stop_target_gradients : bool (default = False)
        Whether to cancel gradients on the returned estimates.
        This function argument is meant for API parity with `rlax`.
    _equality_fun : Callable[[State, State], bool] (default = jnp.array_equal)
        Function to compare `s_t` with `s_next` in order to correctly
        stitch together shuffled (or dropped/ duplicated) paths along the
         batch-dimension

    Returns
    -------
    jax.Array
        A jax array of floats of shape: (T, batch) containing the recursively
        backed up log-likelihood of the given (non-backed up) individual
        log-likelihoods.

    See Also
    --------
    Levine, Sergey. Reinforcement learning and control as probabilistic
        inference: Tutorial and review. arXiv preprint arXiv:1805.00909 (2018).
    """

    def _body(carry, x):
        step, (log_weights_carry, state_reference) = carry
        state, true_next_state, log_weights_t = x

        # 1) STITCHING
        # Connect particles from `s_next` to `s_reference`
        adjacency = jnp.asarray(jax.vmap(
            jax.vmap(
                tree_compare,
                in_axes=(None, 0)
            ), in_axes=(0, None)
        )(true_next_state, state_reference))

        # 2) SUM; Connect paths as a uniform mixture over particles
        # Computes: p(x) = 1/n \sum_{i \in connected} \delta_particle_i(x)
        lse = jax.vmap(
            lambda z: jax.nn.logsumexp(
                log_weights_carry, b=z * (step > 0) / jnp.clip(z.sum(), 1)
            )
        )(adjacency)
        lse = jnp.nan_to_num(lse, neginf=0.0)

        # 2) (log) PRODUCT; BACKUP
        num_edges_t = adjacency.sum(axis=1)
        mask = num_edges_t > 0  # Correct for dropped particles
        backup = log_weights_t + mask * lse

        return (step + 1, (backup, state)), backup

    _, backup_logweights = jax.lax.scan(
        _body,
        (0, jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x[-1]), (log_weights, s_t)
        )),
        (s_t, s_next, log_weights),
        reverse=True
    )

    return jax.lax.select(
        stop_target_gradients,
        jax.lax.stop_gradient(backup_logweights),
        backup_logweights
    )


def tree_truncated_td_lambda[State](
        s_t: State,  # States ([0, T - 1], B, ...)
        s_next: State,  # True next States: ([1, T], B, ...)
        r_t: jax.Array,  # Env-rewards: ([1, T], B, D)
        discount_t: jax.Array,  # Env-discounts: ([1, T], B, D)
        v_next: jax.Array,  # State Values: ([1, T], B, D)
        *,
        gamma: ScalarLike = 1.0,  # Credit-assignment algorithm base discount
        lambda_: ScalarLike = 1.0,
        stop_target_gradients: bool = False,
        _equality_fun: Callable[[State, State], bool] = tree_compare
) -> jax.Array:
    """Compute from a partially ordered trajectory-batch the lambda returns

    This function is a proxy for `tree_truncated_retrace(...)`.
    """
    return tree_truncated_retrace(
        s_t, s_next, r_t, discount_t, v_next,
        gamma=gamma, lambda_=lambda_, log_is=jnp.zeros(()),
        stop_target_gradients=stop_target_gradients,
        _equality_fun=_equality_fun
    )


def tree_truncated_retrace[State](
        s_t: State,  # States ([0, T - 1], B, ...)
        s_next: State,  # True next States: ([1, T], B, ...)
        r_t: jax.Array,  # Env-rewards: ([1, T], B, D)
        discount_t: jax.Array,  # Env-discounts: ([1, T], B, D)
        v_next: jax.Array,  # State Values: ([1, T], B, D)
        *,
        gamma: ScalarLike = 1.0,  # Credit-assignment algorithm base discount
        lambda_: ScalarLike = 1.0,
        log_is: jax.Array = 0.0,
        stop_target_gradients: bool = False,
        _equality_fun: Callable[[State, State], bool] = tree_compare
) -> jax.Array:
    """Compute from a partially ordered trajectory-batch the Retrace(lambda)

    All non-keyword arguments to this function should be prefixed with
    (time, batch, ...) where the `time` axis is assumed to be ordered.

    This function stitches together the paths that get dropped/ duplicated
    in SMC. Dropped paths bootstrap from the canonical next-state (i.e., the
    next state, if the particle had not been dropped), duplicated paths
    average over the future particle lambda-returns.

    To construct the adjacency matrix, the `s_t` and `s_next` states must
    be comparable through the provided `_equality_fun` argument.

    The retrace correction ensures that off-policy data is adjusted by their
    importance sampling ratio 'pi_target / pi_sample', which is clipped to
    min(1, ratio), to prevent exploding values for pi_target > pi_sample.
    This makes the estimator conservative when pi_target < pi_sample.

    For on-policy data, pi_target = pi_sample, this estimator coincides with
    standard TD-lambda.

    Parameters
    ----------
    s_t : State
        Some state object to compare to
    s_next : State
        Some next_state object after `s_t` to compare to
    r_t : jax.Array
        An array of rewards
    discount_t : jax.Array
        An array of discounts (continuation probabilities in [0, 1])
    v_next : jax.Array
        An array of values of `s_next` to recursively backup
    gamma : float (default = 1.0)
        Base discount (continuation probability in [0, 1]) to include
    lambda_ : float (default = 1.0)
        TD-lambda bias-variance parameter in [0, 1].
    log_is: float (default = 0.0)
        Log of the importance sampling ratio: `ln(pi_target) - ln(pi_sample)`
    stop_target_gradients : bool (default = False)
        Whether to cancel gradients on the returned estimates.
        This function argument is meant for API parity with `rlax`.
    _equality_fun : Callable[[State, State], bool] (default = jnp.array_equal)
        Function to compare `s_t` with `s_next` in order to correctly
        stitch together shuffled (or dropped/ duplicated) paths along the
         batch-dimension

    Returns
    -------
    jax.Array
        A jax array of floats of shape: (T, batch) containing the value
        estimate for each state `s_t`.

        The TD-error can be computed with this function as,
            td-error = tree_truncated_retrace(s_t, ...) - V(s_t) OR
            td-error = tree_truncated_retrace(s_t, ...) - Q(s_t, a_t)
        batched over all `s_t` and `a_t` in the (time, batch) dimension.

    See Also
    --------
    TD-Lambda algorithm by R. Sutton,
        http://incompleteideas.net/sutton/book/ebook/node74.html
    Implementation for common RL estimators in rlax
        https://github.com/google-deepmind/rlax
    Munos, Rémi, et al. Safe and efficient off-policy reinforcement learning.
        Advances in neural information processing systems 29 (2016).
        https://papers.nips.cc/paper_files/paper/2016/hash/c3992e9a68c5ae12bd18488bc579b30d-Abstract.html
    """

    def _body(carry, x):
        step, (return_next, state_reference) = carry
        state, true_next_state, reward, discount, value, cor = x

        # 1) STITCHING
        # Connect particles from `s_next` to `s_reference`
        adjacency = jnp.asarray(jax.vmap(
            jax.vmap(
                _equality_fun,
                in_axes=(None, 0)
            ), in_axes=(0, None)
        )(true_next_state, state_reference))

        # Backup connected paths using mean-operator
        num_edges_t = adjacency.sum(axis=1)
        avg_return_t = adjacency @ return_next / jnp.clip(num_edges_t, 1)
        avg_return_t = avg_return_t * (step > 0)  # No valid adjacency at leaf

        # 2) BACKUP
        # Mask to correct for lambda
        lambda_t = cor * lambda_ * (num_edges_t > 0)

        # Update rule: r + gamma * ((1 - lambda) V + lambda G)
        lambda_returns = reward + gamma * discount * (
                (1 - lambda_t) * value + lambda_t * avg_return_t)

        return (step + 1, (lambda_returns, state)), lambda_returns

    corrections = jnp.clip(jnp.exp(log_is), max=1.0)
    corrections = jnp.broadcast_to(corrections, r_t.shape)

    _, returns = jax.lax.scan(
        _body,
        (0, jax.tree_util.tree_map(lambda x: jnp.zeros_like(x[-1]), (r_t, s_t))),
        (s_t, s_next, r_t, discount_t, v_next, corrections),
        reverse=True
    )

    return jax.lax.select(
        stop_target_gradients,
        jax.lax.stop_gradient(returns),
        returns
    )


class PolicyProtocol[State, Action, Params, DistributionParams](Protocol):
    """General policy compatibility protocol for implemented SMC components

    We use this component in the `Proposal` for SMC.
    """

    def apply(
            self,
            params: Params,
            state: State,
            rngs: dict[str, PRNGKeyArray]
    ) -> DistributionParams:
        ...

    def sample(
            self,
            key: PRNGKeyArray,
            shape: Sequence[int],
            dist_params: DistributionParams
    ) -> Action:
        ...

    def logprob(
            self,
            dist_params: DistributionParams,
            action: Action
    ) -> ScalarLike:
        ...


class ActionValueProtocol[State, Action, Params, DistributionParams](Protocol):
    """General value compatibility protocol for implemented SMC components

    We use this component in the `Proposal` for SMC.
    """

    def apply(
            self,
            params: Params,
            state: State,
            action: Action,
            rngs: dict[str, PRNGKeyArray]
    ) -> DistributionParams:
        ...

    def sample(
            self,
            key: PRNGKeyArray,
            shape: Sequence[int],
            dist_params: DistributionParams
    ) -> jax.Array:
        ...

    def logprob(
            self,
            dist_params: DistributionParams,
            value: jax.Array
    ) -> ScalarLike:
        ...


class ValueProtocol[State, Params, DistributionParams](Protocol):
    """General value compatibility protocol for implemented SMC components

    We use this component in the `Target` for SMC.
    """

    def apply(
            self,
            params: Params,
            state: State,
            rngs: dict[str, PRNGKeyArray]
    ) -> DistributionParams:
        ...

    def sample(
            self,
            key: PRNGKeyArray,
            shape: Sequence[int],
            dist_params: DistributionParams
    ) -> jax.Array:
        ...

    def logprob(
            self,
            dist_params: DistributionParams,
            value: jax.Array
    ) -> ScalarLike:
        ...


class TRPIParams[PolicyParams, ValueParams](NamedTuple):
    policy_params: PolicyParams
    value_params: ValueParams
    constraint_param: ScalarLike


class TRPIProposal[
    State,
    Action,
    ProposalParams: TRPIParams[PyTree[jax.Array], PyTree[jax.Array]]
]:
    """Implements the local TRPI proposal for SMC action generation.

    This class offers two main convenient features:
     - Different proposals at root and non-root nodes
     - 3 Different parametrizations for the trust-region constraint

    This class is another layer of abstraction for the `Proposal` needed by
    SMC. It shifts additional required implementation from `Proposal` to the
    `ActionValueProtocol` and `PolicyProtocol`.
    """

    def __init__(
            self,
            value_model: ActionValueProtocol[  # Q(s, a)
                State, Action, PyTree[jax.Array], PyTree[jax.Array]
            ],
            policy_model: PolicyProtocol[  # \pi(a | s)
                State, Action, PyTree[jax.Array], PyTree[jax.Array]
            ],
            bootstrap: int | None,
            constraint_param: float,
            constraint_type: str,
            method: str,
            root_method: str | None = None,
            mode: str = '',
            k: int = None
    ):
        prop = getattr(proposals, method, None)
        root_prop = prop
        if root_method is not None:
            root_prop = getattr(proposals, root_method, None)

        if (prop is None) or (root_prop is None):
            raise ValueError(
                f'{method} or {root_method} is not supported! Choose from: '
                f'{proposals.supported}'
            )

        self.proposal = prop(logits=True)
        self.root_proposal = root_prop(logits=True)

        supported_types = {'soft', 'hard', 'sandwich'}
        if constraint_type not in supported_types:
            raise ValueError(
                f'{constraint_type} is not supported! '
                f'Choose from: {supported_types}'
            )

        self.value_model = value_model
        self.policy_model = policy_model

        self.bootstrap = bootstrap

        self.base_constraint_param = constraint_param
        self.constraint_type = constraint_type

        self.mode = mode
        self.k = k

    def sandwich_epsilon(
            self,
            qs: jax.Array,  # vector
            log_pi: jax.Array,  # vector
            factor: ScalarLike,  # float-like
            level: jax.typing.ArrayLike  # int-like
    ) -> ScalarLike:  # float-like

        high0 = self.root_proposal.trust_region_upperbound(qs, jnp.exp(log_pi))
        high_ = self.proposal.trust_region_upperbound(qs, jnp.exp(log_pi))

        high = jax.lax.select(level == ROOT_DEPTH, high0, high_)
        low = jax.lax.select(
            level == ROOT_DEPTH,
            self.root_proposal._epsilon_ltol,
            self.proposal._epsilon_ltol
        )

        high = jnp.clip(high, low)

        factor = jnp.clip(factor, 0, 1)
        epsilon = factor * (high - low) + low
        epsilon = jnp.clip(epsilon, 0, high)  # numerical stability

        return epsilon

    def __call__(
            self,
            params: ProposalParams,
            key: PRNGKeyArray,
            state: State,
            level: jax.Array,  # int-like
    ) -> tuple[Action, ScalarLike, ScalarLike]:
        key_policy, key_value, key_boot, key_sample = jax.random.split(key, 4)

        cache = rank = None
        if isinstance(state, tuple):
            # Reuse cached keys inside state to expand identical 'atoms' for
            # duplicated states from resampling.
            state, rank, cache = state

        # Sample atoms (or take them directly) from the policy model
        policy_out = self.policy_model.apply(
            params.policy_params,
            state.observation,
            rngs={'default': key_policy}
        )
        if self.bootstrap is not None:
            # Use a uniform prior when bootstrapping from prior.

            log_pi = jnp.zeros(self.bootstrap)
            sampled_atoms = self.policy_model.sample(
                key_boot, (self.bootstrap,), policy_out
            )

            atoms = sampled_atoms

        else:
            log_pi = policy_out
            atoms = jnp.arange(log_pi.size)

        log_pi = jax.nn.log_softmax(log_pi.squeeze())  # ensures a proper PMF

        # Compute (estimates of) the value for all sampled atoms
        key_value_model, key_value_sample = jax.random.split(key_value)
        qs_params = jax.vmap(
            partial(self.value_model.apply, rngs={'default': key_value_model}),
            in_axes=(None, None, 0)
        )(params.value_params, state.observation, atoms)
        qs = self.value_model.sample(key_value_sample, (), qs_params)

        if log_pi.shape != qs.shape:
            raise RuntimeError(
                "Proposal dimensionality error: "
                "Log-Prior does not match shape of Qs!"
            )

        # Extract trust-region parameter from Params container
        value = params.constraint_param
        inv_beta, epsilon = {
            'soft': (value, None),
            'hard': (None, value),
            'sandwich': (None, self.sandwich_epsilon(qs, log_pi, value, level))
        }[self.constraint_type]

        # Run the proposal algorithm given the unpacked and computed statistics
        logits = jax.lax.cond(
            level == ROOT_DEPTH,
            partial(self.root_proposal, inv_beta=inv_beta, epsilon=epsilon),
            partial(self.proposal, inv_beta=inv_beta, epsilon=epsilon),
            qs,
            jnp.exp(log_pi),
        )

        # Sample an action and return its likelihoods.
        if self.mode == 'argmax':
            # Take the argmax action (naive det. sampling)
            idx = logits.argmax()

        elif self.mode == 'argtopk' and rank is not None:
            # Take the top k (num. particles associated with this state) actions (proper det. sampling)
            k = min(max(self.k, 0), int(logits.shape[0]))
            _, indices = jax.lax.top_k(logits, k)
            idx = indices.at[rank % indices.size].get()

        elif self.mode == 'sampled-argtopk' and cache is not None and rank is not None:
            # Sample without replacement (multiple particles associated with the state will take different actions
            gumbel_noise = jax.random.gumbel(cache, (log_pi.size,))
            logits = logits + gumbel_noise
            k = min(max(self.k, 0), int(logits.shape[0]))
            _, indices = jax.lax.top_k(logits, k)
            idx = indices.at[rank % indices.size].get()
        else:
            idx = jax.random.categorical(
                key_sample, logits=logits, shape=()
            )

        return (
            jax.tree_util.tree_map(lambda x: x.at[idx].get(), atoms),
            log_pi.at[idx].get(),
            logits.at[idx].get()
        )


class ELBOTargetParams[ValueParams](NamedTuple):
    value_params: ValueParams
    temperature: ScalarLike
    root_temperature: ScalarLike
    advantage_softmax: Bool


class ELBOTarget[State, TargetParams: ELBOTargetParams[PyTree[jax.Array]]]:
    """Implements the Evidence Lower BOund target operator for RL-SMC.

    Provides functionality for log-weight normalization as suggested by:
     - Macfarlane et al. SMX: Sequential Monte Carlo Planning for
       Expert Iteration. 2024. https://arxiv.org/abs/2402.07963

    This class is another layer of abstraction for the `Target` needed by SMC.
    It still requires additional implementation for the `ValueProtocol`.
    """

    def __init__(
            self,
            value_model: ValueProtocol[
                State, PyTree[jax.Array], PyTree[jax.Array]
            ],
            normalizer: str | None,
            temperature: float = 1.0,
            root_temperature: float = 1.0,
            advantage_softmax: bool = True,
    ):
        self.value_model = value_model

        norm_method = norm.unnormed
        norm_options = {
            'softmax_minmax': norm.logsoftmax_minmax,
            'softmax_standardize': norm.logsoftmax_standardize,
            'softmax': norm.logsoftmax
        }
        if normalizer in norm_options:
            norm_method = norm_options[normalizer]

        self.logweight_normalizer = norm_method
        self.temperature = temperature
        self.root_temperature = root_temperature
        self.advantage_softmax = advantage_softmax


    def __call__(
            self,
            params: TargetParams,
            key: PRNGKeyArray,
            state: State
    ):
        if isinstance(state, tuple):
            # Reuse cached keys inside state to expand identical 'atoms' for
            # duplicated states from resampling. See Proposal.
            state, *_ = state

        # Return a value of the given state (compute message).
        key_model, key_sample = jax.random.split(key)
        value_params = self.value_model.apply(
            params.value_params, state.observation, rngs={'default': key_model}
        )
        value = self.value_model.sample(key_sample, (), value_params)

        return value

    def log_weights(
            self,
            params: TargetParams,
            log_w: ScalarLike,
            data: ParticleData
    ) -> tuple[ScalarLike, ScalarLike]:
        # Compute target importance-sampling weights

        log_is = data.log_prior - data.log_proposal
        value_next = data.value_next * data.discount

        if self.advantage_softmax:
            new_weights = log_w + log_is + (
                    data.reward + value_next - data.value_t
            ) / params.temperature
        else:
            new_weights = log_w + log_is + (
                    data.reward + value_next
            ) / params.temperature

        logits = self.logweight_normalizer(new_weights, 1.0)

        return new_weights, logits


class CompositeAccumulator:

    def __init__(
            self,
            *args: tuple[str, StatisticAccumulator],
            return_data: bool = False
    ):
        self.accumulators = list(args)

        if not self.accumulators:
            raise RuntimeError(
                "CompositeAccumulator was given empty arguments"
            )

        self.names = [x[0] for x in self.accumulators]
        self.return_data = return_data

    def init(self, num_particles: int, depth: int) -> Trace:
        branches = jnp.arange(0, num_particles)
        log_weights = jnp.zeros(num_particles)

        traces = [x[1].init(num_particles, depth) for x in self.accumulators]

        return Trace(
            branches, log_weights,
            memory={k: v.memory for k, v in zip(self.names, traces)},
            data={k: v.data for k, v in zip(self.names, traces)}
        )

    def update(
            self,
            trace: Trace,
            data: ParticleData,
            new_branches,
            old_log_weights,
            logits,
            new_log_weights
    ):
        return_data = data if self.return_data else None

        traces = [x[1].update(
            Trace(trace.branches, trace.log_weights,
                  trace.memory[x[0]], trace.data[x[0]]),
            data, new_branches,
            old_log_weights, logits, new_log_weights
        )[0]
            for x in self.accumulators
                  ]

        return Trace(
            traces[0].branches,
            traces[0].log_weights,
            memory={k: v.memory for k, v in zip(self.names, traces)},
            data={k: v.data for k, v in zip(self.names, traces)}
        ), return_data


class LogweightAccumulator:

    def __init__(self, return_data: bool = False):
        self.return_data = return_data

    def init(self, num_particles: int, depth: int) -> Trace:
        branches = jnp.arange(0, num_particles)
        log_weights = jnp.zeros(num_particles)

        eligiblity = jnp.ones(num_particles)  # Memory
        values = jnp.zeros(num_particles)  # Q(s, a) values

        return Trace(branches, log_weights, eligiblity, values)

    def update(
            self,
            trace: Trace,
            data: ParticleData,
            new_branches,
            old_log_weights,
            logits,
            new_log_weights
    ):
        return_data = data if self.return_data else None

        logprobs = jax.nn.log_softmax(old_log_weights - trace.memory)

        # Calculate average occupancy per branch
        one_hot = jax.nn.one_hot(trace.branches, trace.branches.size)
        bincount = one_hot.T @ jnp.ones_like(logprobs)  # == jnp.bincount

        # Combine log-weights only for root-actions with non-zero occupancy
        aggregated = jax.vmap(
            lambda z: jax.nn.logsumexp(logprobs, b=z / jnp.clip(z.sum(), 1)),
        )(
            one_hot.T
        )

        updates = jnp.nan_to_num(
            (bincount > 0) * aggregated, neginf=0.0, posinf=0.0
        )

        # Backup non-normalized values if not resampled to reduce variance
        resampled = (trace.branches == new_branches).all()
        track = jax.lax.select(resampled, updates, logprobs)

        new_data = trace.data + track

        return Trace(
            new_branches, new_log_weights,
            memory=logprobs, data=new_data
        ), return_data


class RetraceAccumulator:
    """Accumulate the Retrace(lambda) estimator for Q(s, a)."""

    def __init__(
            self,
            gamma: float, td_lambda: float,
            return_data: bool = False
    ):
        self.gamma = gamma
        self.td_lambda = td_lambda

        self.return_data = return_data

    def init(self, num_particles: int, depth: int) -> Trace:
        branches = jnp.arange(0, num_particles)
        log_weights = jnp.zeros(num_particles)

        eligiblity = jnp.ones(num_particles)
        values = jnp.zeros(num_particles)  # Q(s, a) values

        return Trace(branches, log_weights, eligiblity, values)

    def update(
            self,
            trace: Trace,
            data: ParticleData,
            new_branches,
            old_log_weights,
            logits,
            new_log_weights
    ):
        return_data = data if self.return_data else None

        # Calculates the Retrace(lambda) correction (cf. arXiv:1606.02647)
        log_is = data.log_prior - data.log_proposal

        c = jnp.clip(jnp.exp(log_is), max=1.0) * self.td_lambda

        discount = data.discount * self.gamma
        value_next = discount * data.value_next
        err = data.reward + value_next - data.value_t

        # Calculate preliminary updates to be allocated to the new roots
        update = trace.memory * err
        eligibility = trace.memory * c * discount

        # Calculate average occupancy per branch
        one_hot = jax.nn.one_hot(trace.branches, trace.branches.size)
        bincount = one_hot.T @ jnp.ones_like(err)  # == jnp.bincount

        # Average backup along trajectories.
        avg_trace = (update @ one_hot) / jnp.clip(bincount, 1)

        new_data = trace.data + avg_trace
        new_eligibility = eligibility.at[new_branches].get()

        return Trace(
            new_branches, new_log_weights,
            memory=new_eligibility, data=new_data
        ), return_data


class DSMCAccumulator:
    def __init__(
        self,
        gamma: float,
        td_lambda: None,
        return_data: bool = False
    ):
        self.gamma = gamma
        self.return_data = return_data

    def init(self, num_particles: int, depth: int) -> Trace:
        branches = jnp.arange(0, num_particles)
        log_weights = jnp.zeros(num_particles)

        memory = (
            jnp.zeros(num_particles),                   # cumulative rewards
            jnp.zeros(num_particles, dtype=jnp.int32),  # depth
            1                                           # iteration count
        )

        # TODO: In discrete envs, this needs to be size logits, bootstrap, or num_actions_to_search, not num_particles
        values = jnp.zeros(num_particles)  # Q-values per root

        return Trace(branches, log_weights, memory=memory, data=values)

    def update(
        self,
        trace: Trace,
        data: ParticleData,
        new_branches,
        old_log_weights,
        logits,
        new_log_weights
    ):
        return_data = data if self.return_data else None

        cumreward, depth, it = trace.memory
        discount = data.discount * self.gamma

        reward_update = jnp.power(discount, depth) * data.reward
        value_next = jnp.power(discount, depth + 1) * data.value_next

        cumreward = cumreward + reward_update
        returns = cumreward + value_next  # shape: (N,)

        # --- CORRECTION 1: derive actual number of branches ---
        num_branches = trace.branches.size

        # --- CORRECTION 2: one-hot shape = (N, num_branches) ---
        one_hot = jax.nn.one_hot(trace.branches, num_branches)  # shape (N, num_branches)
        bincount = one_hot.T @ jnp.ones_like(data.reward)       # shape (num_branches,)

        # --- Numerically stable softmax with masking ---
        def safe_softmax_masked(log_w, z):
            masked = jnp.where(z, log_w, -jnp.inf)
            return jax.lax.cond(
                jnp.any(z),
                lambda _: jnp.exp(masked - jax.nn.logsumexp(masked)),
                lambda _: jnp.zeros_like(log_w),
                operand=None
            )

        # --- CORRECTION 3: shape (num_branches, N) ---
        root_normed_weight = jax.vmap(lambda z: safe_softmax_masked(old_log_weights, z))(one_hot.T)

        # --- CORRECTION 4: returns @ weights = (num_branches,) ---
        values = root_normed_weight @ returns  # shape: (num_branches,)

        # --- Handle branches that were dropped this iteration ---
        mask = bincount > 0                    # shape: (num_branches,)
        # TODO: Currently, this is not implemented correctly for discrete spaces. If there are duplicate actions at the
        #  root, the action values will be averaged as if all duplicate action-values at the root have the same weight.
        #  But this is not correct, because each update should be weighted by the particles' probabilities with respect
        #  to each other. This is currently computed this way outside the planner (in SMCPolicy or SHSMCTS), but really
        #  must be addressed here.
        # An "easy" way to do this is to cast values and mask to num_actions_to_search size first
        prev_Q = trace.data
        new_Q = jnp.where(mask, values, prev_Q)#values * mask + prev_Q * (1 - mask)

        # --- Moving average ---
        updated_data = ((it - 1) * prev_Q + new_Q) / it

        # Get new cumulative rewards at branch positions
        new_cumrewards = cumreward.at[new_branches].get()

        return Trace(
            new_branches,
            new_log_weights,
            memory=(new_cumrewards, depth + 1, it + 1),
            data=updated_data
        ), return_data
