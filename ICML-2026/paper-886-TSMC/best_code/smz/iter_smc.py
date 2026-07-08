"""Extends the base sequential Monte-Carlo with iterative parameter updates

"""
from typing import Protocol

import jax
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray, Integer, ArrayLike, Num

from smz.smc import SMC, SMCParams, ParticleData


class ParameterUpdater[
    State, Action,
    ProposalParams, TransitionParams, TargetParams
](Protocol):
    """Defines function signature for a SMC-parameter updater
    """

    def __call__(
            self,
            key: PRNGKeyArray,
            step: Integer[ArrayLike, ''],
            params: SMCParams[ProposalParams, TransitionParams, TargetParams],
            data: ParticleData[State, Action]
    ) -> SMCParams[ProposalParams, TransitionParams, TargetParams]:
        """Update `param` given `data` to improve subsequent target estimation

        Parameters
        ----------
        key : PRNGKeyArray
            Jax based random key
        step : int
            Current iteration/ step of IteratedSMC
        params : SMCParams
            Parameter container for the SMC components
        data : ParticleData
            Particle-data container collected during the current SMC iteration

        Returns
        -------
        SMCParams
            An updated version of the given SMCParams. For example, one can
            use MLE/ MAP estimation to fit/ improve a model to `data` so that
            the next iteration of `SMC` has an improved target or proposal.
        """
        ...


class IteratedSMC[
    State, Action,
    ProposalParams, TransitionParams, TargetParams
]:
    """Implements an iterated version of SMC

    Performs multiple SMC planning calls to adjust model parameters locally
    based on intermediate samples. Any updated parameters are thrown away.
    """

    def __init__(
            self,
            smc: SMC[
                State, Action,
                ParticleData[State, Action],
                ProposalParams, TransitionParams, TargetParams
            ],
            *,
            update_function: ParameterUpdater[
                State, Action,
                ProposalParams, TransitionParams, TargetParams
            ],
            num_iterations: int
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
        self.smc = smc
        self.smc.statistic_fun = self.statistic_fun  # Warning: mutability.

        self.update_function = update_function
        self.num_iterations = num_iterations

    def __str__(self) -> str:
        return (f"{type(self).__name__}(N={self.num_iterations}, "
                f"SMC={str(self.smc)})")

    @staticmethod
    def statistic_fun(
            data: ParticleData[State, Action]
    ) -> ParticleData[State, Action]:
        """Collect all transitions within the SMC planner"""
        return data

    @property
    def root_depth(self) -> int:
        """Base index (level) as a reference"""
        return self.smc.root_depth

    @property
    def budget(self) -> int:
        """Number of transitions computed by IteratedSMC at each call to `run`.

        """
        return (
                self.smc.depth *
                self.smc.num_particles *
                (self.num_iterations + 1)
        )

    @staticmethod
    def make_params(*args, **kwargs) -> SMCParams[
        ProposalParams, TransitionParams, TargetParams
    ]:
        """Helper method inside SMC namespace for constructing param-container
        """
        return SMCParams(*args, **kwargs)

    def iterate_params(
            self,
            key: PRNGKeyArray,
            params: SMCParams[ProposalParams, TransitionParams, TargetParams],
            state: State
    ) -> tuple[
        tuple[
            SMCParams[ProposalParams, TransitionParams, TargetParams],
            PRNGKeyArray
        ],
        tuple[
            tuple[
                Num[jax.Array, ' num_particles'],
                Integer[jax.Array, ' num_particles']
            ],
            ParticleData[State, Action],
            SMCParams[ProposalParams, TransitionParams, TargetParams]
        ] | None
        ]:
        """Repeat the SMC planner at a given state to improve `params`.

        Parameters
        ----------
        key : PRNGKeyArray
            Jax based random key
        params : SMCParams[PropParam, TrParam, TarParam]
            Parameter container for the proposal, transition, and target
        state : State
            Current state of the hidden Markov model to start planning at

        Returns
        -------
        SMCParams[PropParam, TrParam, TarParam] optional
            The locally updated parameters for SMC.
        """

        def _body(
                _carry: tuple[SMCParams, PRNGKeyArray],
                i: Integer[ArrayLike, '']
        ) -> tuple[
            tuple[SMCParams, PRNGKeyArray],
            tuple[
                tuple[
                    Num[jax.Array, ' num_particles'],
                    Integer[jax.Array, ' num_particles']
                ],
                ParticleData[State, Action],
                SMCParams[ProposalParams, TransitionParams, TargetParams]
            ]
        ]:
            carry_params, carry_rng = _carry
            carry_rng, step_key, update_key = jax.random.split(carry_rng, 3)

            # Sample data using SMC
            action_out, stats = self.smc.run(step_key, carry_params, state)

            # Update params using sampled data
            new_params = self.update_function(
                update_key, i, carry_params, stats
            )

            return (new_params, carry_rng), (action_out, stats, new_params)

        # Do multiple iterations of SMC to locally improve `params`.
        carry, out = (params, key), None
        if self.num_iterations > 1:
            steps = jnp.arange(0, self.num_iterations - 1)
            carry, out = jax.lax.scan(_body, carry, steps)

        return carry, out

    def run(
            self,
            key: PRNGKeyArray,
            params: SMCParams[ProposalParams, TransitionParams, TargetParams],
            state: State
    ) -> tuple[
        tuple[
            Num[jax.Array, ' num_particles'],
            Integer[jax.Array, ' num_particles']
        ], ParticleData[State, Action]
    ]:
        """Run the IteratedSMC planner at a given state.

        Returns a sample (bootstrapped) distribution with normalized weights,
        along with sampled statistics as given by self.statistic_fun.

        Parameters
        ----------
        key : PRNGKeyArray
            Jax based random key
        params : SMCParams[PropParam, TrParam, TarParam]
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
        SMCParams[PropParam, TrParam, TarParam] optional
            If `return_params=True`, then the final parameters are returned
        """

        # Cache key so that output of SMC == I-SMC when num_iterations = 0
        cached = key
        key_use, _ = jax.random.split(key)

        final_params, _ = self.iterate_params(key_use, params, state)[0]

        # Use the improved parameters to do a final run of SMC.
        outputs = self.smc.run(cached, final_params, state)

        return outputs
