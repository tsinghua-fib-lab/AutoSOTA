"""This module implements update methods for the iterated SMC planner"""
from typing import Callable, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp

import optax

from jaxtyping import PRNGKeyArray, ScalarLike, PyTree

from smz.smc import SMC, SMCParams, ParticleData
from . import impl


type _DataSet[State, Action] = tuple[[State, Action], jax.Array]


class FinetuningHyperparams(NamedTuple):

    # Credit assignment parameters
    td_lambda: float
    discount: float

    # Flag to include or exclude updating of `proposal` or `target`
    update_proposal: bool
    update_target: bool

    # Maximum size of the buffer in terms of iterations (tabular only)
    max_buffer_size: int | None = None

    # Gradient descent parameters (network finetuning only)
    num_steps: int | None = None
    batch_size: int | None = None


class MakeSMCDatasetMixin[
    State, Action,
    ProposalParams, TransitionParams, TargetParams
]:
    smc: SMC
    param: FinetuningHyperparams

    def _make_dataset(
            self,
            key: PRNGKeyArray,
            params: SMCParams[ProposalParams, TransitionParams, TargetParams],
            transitions: ParticleData[State, Action]
    ) -> _DataSet[State, Action]:

        # Estimate the value for the target operator.
        key_batch_td = jax.random.split(
            key, (self.smc.depth, self.smc.num_particles)
        )
        next_state_values = jax.vmap(jax.vmap(
            partial(self.smc.target, params.target)
        ))(key_batch_td, transitions.next_state)

        value_tree = impl.tree_truncated_retrace(
            transitions.state.observation, transitions.next_state.observation,
            transitions.reward, transitions.discount, next_state_values,
            log_is=transitions.log_prior - transitions.log_proposal,
            gamma=self.param.discount, lambda_=self.param.td_lambda
        )

        dataset = jax.tree_util.tree_map(
            lambda x: x.reshape(self.smc.budget, *x.shape[2:]),
            ((transitions.state, transitions.action), value_tree)
        )

        return dataset


class ValueFinetuning[
    State, Action,
    X, A,
    ProposalParams: impl.TRPIParams[PyTree[jax.Array], PyTree[jax.Array]],
    TransitionParams,
    TargetParams: impl.ELBOTargetParams[PyTree[jax.Array]]
](MakeSMCDatasetMixin[
    State, Action, ProposalParams, TransitionParams, TargetParams
]):
    """Use SMC-collected data to update value models to improve SMC-estimation

    This class only works with `impl.TRPIProposal` and `impl.ELBOTarget`.
    """

    def __init__(
            self,
            smc: SMC[
                State, Action,
                ParticleData[State, Action],
                ProposalParams, TransitionParams, TargetParams
            ],
            optimizer: optax.GradientTransformation,
            hyperparams: FinetuningHyperparams,
            formatter: Callable[
                           [_DataSet[State, Action]], _DataSet[X, A]
                       ] | None = None,
    ):
        # Argument validation
        _prop = smc.proposal
        if not isinstance(_prop, impl.TRPIProposal):
            raise NotImplementedError(
                f"The given `SMC` planner does not have a supported proposal! "
                f"Expected: {impl.TRPIProposal}. "
                f"Received: {smc.proposal.__class__}"
            )

        _tar = smc.target
        if not isinstance(_tar, impl.ELBOTarget):
            raise NotImplementedError(
                f"The given `SMC` planner does not have a supported target! "
                f"Expected: {impl.ELBOTarget}. "
                f"Received: {smc.target.__class__}"
            )

        if (not hyperparams.update_proposal) and \
                (not hyperparams.update_target):
            raise ValueError(
                "Hyperparameters: `update_proposal` and `update_target` "
                "cannot both be `False`!"
            )

        self.planner = smc

        self.target_value = _tar.value_model
        self.prop_value = _prop.value_model

        self.optimizer = optimizer
        self.param = hyperparams

        self.formatter = (lambda x: x) if formatter is None else formatter

    def __call__(
            self,
            key: PRNGKeyArray,
            iteration: jax.Array,
            params: SMCParams[ProposalParams, TransitionParams, TargetParams],
            smc_data: ParticleData[State, Action]
    ) -> SMCParams[ProposalParams, TransitionParams, TargetParams]:

        def sample_loss(
                _key: PRNGKeyArray,
                _model_params: optax.Params,
                _inputs: tuple[X, A],
                _targets: jax.Array
        ) -> ScalarLike:
            state, actions = _inputs

            _loss = 0.
            if self.param.update_proposal:
                qs_params = self.prop_value.apply(
                    _model_params['Q'], state, actions, rngs={'default': _key}
                )

                _loss += -self.prop_value.logprob(qs_params, _targets)

            if self.param.update_target:
                v_params = self.target_value.apply(
                    _model_params['V'], state, rngs={'default': _key}
                )

                _loss += -self.target_value.logprob(v_params, _targets)

            return _loss

        def loss(
                _key: PRNGKeyArray,
                _model_params: optax.Params,
                _inputs: tuple[X, A],
                _targets: jax.Array
        ) -> ScalarLike:
            key_batch = jax.random.split(_key, self.param.batch_size)

            batch_losses = jax.vmap(
                sample_loss, in_axes=(0, None, 0, 0)
            )(key_batch, _model_params, _inputs, _targets)

            return batch_losses.mean()

        def update_fun(
                _key: PRNGKeyArray,
                _model_params: optax.Params,
                _opt_state: optax.OptState,
                _data: _DataSet[X, A]
        ) -> tuple[optax.Params, optax.OptState]:

            grad = jax.grad(loss, argnums=1)(_key, _model_params, *_data)

            updates, _opt_state = self.optimizer.update(
                grad, _opt_state, _model_params
            )
            _model_params = optax.apply_updates(_model_params, updates)

            return _model_params, _opt_state

        def body(
                carry: tuple[
                    PRNGKeyArray, optax.OptState, optax.Params, _DataSet[X, A]
                ],
                xs: None = None
        ) -> tuple[
            tuple[PRNGKeyArray, optax.OptState, optax.Params, _DataSet[X, A]],
            None
        ]:
            # Unpack scan data
            key_carry, _opt_state, _vars, _dataset = carry
            key_carry, key_sample, key_loss = jax.random.split(key_carry, 3)

            # Uniformly randomly construct a minibatch of data
            batch_indices = jax.random.randint(
                key_sample, (self.param.batch_size, ), 0, self.planner.budget
            )
            ins, targets = jax.tree_util.tree_map(
                lambda x: x.at[batch_indices].get(), _dataset
            )

            # Do a SGD update
            _vars, _opt_state = update_fun(
                key_loss, _vars, _opt_state, (ins, targets)
            )

            return (key_carry, _opt_state, _vars, _dataset), None

        key_sgd, key_make_data = jax.random.split(key)

        # Create a dataset of (s, a) -> Value pairs from `smc_data`.
        dataset = self._make_dataset(key_make_data, params, smc_data)
        formatted_dataset = self.formatter(dataset)

        old_params = {
            'Q': params.proposal.value_params,
            'V': params.target.value_params
        }
        opt_state = self.optimizer.init(old_params)

        *_, new_params, _ = jax.lax.scan(
            body, (key, opt_state, old_params, formatted_dataset), None,
            length=self.param.num_steps
        )[0]

        new_prop_params = params.proposal._replace(
            value_params=new_params['Q']
        )
        new_target_params = params.target._replace(
            value_params=new_params['V']
        )

        updated_params = SMCParams(
            proposal=new_prop_params,
            transition=params.transition,
            target=new_target_params
        )

        return updated_params


class ValueBufferUpdate[
    State, Action,
    X, A,
    ProposalParams: impl.TRPIParams[PyTree[jax.Array], PyTree[jax.Array]],
    TransitionParams,
    TargetParams: impl.ELBOTargetParams[PyTree[jax.Array]]
](MakeSMCDatasetMixin[
    State, Action, ProposalParams, TransitionParams, TargetParams
]):
    """Use SMC-collected data to update value models to improve SMC-estimation

    This class only works with `impl.TRPIProposal` and `impl.ELBOTarget`.
    """

    def __init__(
            self,
            smc: SMC[
                State, Action,
                ParticleData[State, Action],
                ProposalParams, TransitionParams, TargetParams
            ],
            hyperparams: FinetuningHyperparams,
            formatter: Callable[
                           [_DataSet[State, Action]], _DataSet[X, A]
                       ] | None = None,
    ):
        # Argument validation
        _prop = smc.proposal
        if not isinstance(_prop, impl.TRPIProposal):
            raise NotImplementedError(
                f"The given `SMC` planner does not have a supported proposal! "
                f"Expected: {impl.TRPIProposal}. "
                f"Received: {smc.proposal.__class__}"
            )

        _tar = smc.target
        if not isinstance(_tar, impl.ELBOTarget):
            raise NotImplementedError(
                f"The given `SMC` planner does not have a supported target! "
                f"Expected: {impl.ELBOTarget}. "
                f"Received: {smc.target.__class__}"
            )

        if (not hyperparams.update_proposal) and \
                (not hyperparams.update_target):
            raise ValueError(
                "Hyperparameters: `update_proposal` and `update_target` "
                "cannot both be `False`!"
            )

        self.planner = smc

        self.target_value = _tar.value_model
        self.prop_value = _prop.value_model

        self.param = hyperparams

        self.formatter = (lambda x: x) if formatter is None else formatter

    def __call__(
            self,
            key: PRNGKeyArray,
            iteration: jax.Array,
            params: SMCParams[ProposalParams, TransitionParams, TargetParams],
            smc_data: ParticleData[State, Action]
    ) -> SMCParams[ProposalParams, TransitionParams, TargetParams]:

        # Checks for configuration
        none_target = params.target.buffer is None
        none_prop = params.proposal.buffer is None
        if none_target and self.param.update_target:
            print('Warning: requested target-updating, but buffer is None')
        if none_prop and self.param.update_proposal:
            print('Warning: requested proposal-updating, but buffer is None')

        if none_prop and none_target:
            raise RuntimeError("Do not use ValueBufferUpdate without buffers!")

        # Create a dataset of (s, a) -> Value pairs from `smc_data`.
        dataset = self._make_dataset(key, params, smc_data)
        data_in, targets = self.formatter(dataset)

        prop_params = params.proposal
        if (not none_prop) and self.param.update_proposal:
            new_buffer = params.proposal.buffer.update(data_in, targets)
            prop_params = params.proposal._replace(buffer=new_buffer)

        target_params = params.target
        if (not none_target) and self.param.update_target:
            new_buffer = params.target.buffer.update(data_in, targets)
            target_params = params.target._replace(buffer=new_buffer)

        updated_params = SMCParams(
            proposal=prop_params,
            transition=params.transition,
            target=target_params
        )

        return updated_params


class TRPIScheduleHyperparams(NamedTuple):
    min: float
    max: float
    shift: float
    scale: float
    target_scale: float | None = 2.0


class TRPISchedule[
    State, Action,
    ProposalParams: impl.TRPIParams,
    TransitionParams,
    TargetParams: impl.ELBOTargetParams
]:

    def __init__(
            self,
            smc: SMC[
                State, Action,
                ParticleData[State, Action],
                ProposalParams, TransitionParams, TargetParams
            ],
            hyperparams: TRPIScheduleHyperparams
    ):
        # Argument validation
        _prop = smc.proposal
        if not isinstance(_prop, impl.TRPIProposal):
            raise NotImplementedError(
                f"The given `SMC` planner does not have a supported proposal! "
                f"Expected: {impl.TRPIProposal}. "
                f"Received: {smc.proposal.__class__}"
            )

        self.planner = smc
        self.method = _prop.constraint_type
        self.param = hyperparams

    def __call__(
            self,
            key: PRNGKeyArray,
            iteration: jax.Array,
            params: SMCParams[ProposalParams, TransitionParams, TargetParams],
            smc_data: ParticleData[State, Action]
    ) -> SMCParams[ProposalParams, TransitionParams, TargetParams]:

        # TODO: extend/ test beyond linear schedules
        i = iteration + 1
        options = {
            'soft': jnp.clip(
                self.param.scale / i, self.param.min, self.param.max
            ),
            'hard': jnp.clip(
                self.param.scale * i, self.param.min, self.param.max
            ),
            'sandwich': jnp.clip(self.param.scale * i, 0, 1),
        }
        new_param = self.param.shift + options[self.method]

        new_trpi_params = params.proposal._replace(constraint_param=new_param)
        updated_params = params._replace(proposal=new_trpi_params)

        if self.param.target_scale is not None:
            # Update target-temperature using soft-schedule that is slightly
            # more greedy than the proposal (according to target_scale).

            target_temp = jnp.clip(
                self.param.scale / (i * max(1.0, self.param.target_scale)),
                self.param.min, self.param.max
            )
            new_target_params = params.target._replace(temperature=target_temp)
            updated_params = updated_params._replace(target=new_target_params)

        return updated_params


class ProposalRootBootstrapCache[
    State, Action,
    ProposalParams: impl.TRPIParams,
    TransitionParams,
    TargetParams
]:

    def __call__(
            self,
            key: PRNGKeyArray,
            iteration: jax.Array,
            params: SMCParams[ProposalParams, TransitionParams, TargetParams],
            smc_data: ParticleData[State, Action]
    ) -> SMCParams[ProposalParams, TransitionParams, TargetParams]:

        if params.proposal.root_cache is not None:
            prop_params = params.proposal._replace(
                root_cache=(smc_data.action[0], jnp.asarray(True))
            )
            params = params._replace(proposal=prop_params)

        return params


def chain[State, Action, ProposalParams, TransitionParams, TargetParams](
        *args: Callable[[
            PRNGKeyArray,
            jax.Array,
            SMCParams[ProposalParams, TransitionParams, TargetParams],
            ParticleData[State, Action]
        ], SMCParams[ProposalParams, TransitionParams, TargetParams]]
) -> Callable[[
    PRNGKeyArray,
    jax.Array,
    SMCParams[ProposalParams, TransitionParams, TargetParams],
    ParticleData[State, Action]
], SMCParams[ProposalParams, TransitionParams, TargetParams]]:
    # Combines multiple params-transformations (updates) into one callable.

    if len(args) == 0:
        raise ValueError("Specify at least 1 function to *args")

    def f(
            key: PRNGKeyArray,
            iteration: jax.Array,
            params: SMCParams[ProposalParams, TransitionParams, TargetParams],
            smc_data: ParticleData[State, Action]
    ) -> SMCParams[ProposalParams, TransitionParams, TargetParams]:

        result = params
        carry = key
        for fun in args:
            carry, key = jax.random.split(carry)
            result = fun(key, iteration, result, smc_data)

        return result

    return f
