from typing import Literal, Sequence

import jax

import optax

from jaxtyping import PRNGKeyArray

from smz.planning import impl


class PPOPolicy[State, Observation, Action, Param]:

    def __init__(
            self,
            value_model: impl.ValueProtocol[
                State, Param, jax.Array | Sequence[jax.Array]
            ],
            policy_model: impl.PolicyProtocol[  # \pi(a | s)
                State, Action, Param, jax.Array | Sequence[jax.Array]
            ],
            stochastic_eval: bool = True
    ):
        self.value_model = value_model
        self.policy_model = policy_model

        self.stochastic_eval = stochastic_eval  # TODO: Implement. PPO defaults to stoch. action selection, so det. currently unimplemented.

    def __call__(
            self,
            key: PRNGKeyArray,
            policy_state: dict[Literal['policy', 'value'], optax.Params],
            obs: Observation,
            state: State | None = None,
            train: bool = True
    ) -> tuple[..., Action, dict[str, jax.Array]]:
        key_policy, key_value = jax.random.split(key)

        # Get action and logprob at current state
        key_model, key_sample = jax.random.split(key_policy)
        policy_out = self.policy_model.apply(
            policy_state['policy'], state.observation,
            rngs={'default': key_model}
        )
        action = self.policy_model.sample(key_sample, (), policy_out)
        logprob = self.policy_model.logprob(policy_out, action)

        # Get value at the current state (cache for TD-estimation later).
        key_value_out, key_value_sample = jax.random.split(key_value)
        value_out = self.value_model.apply(
            policy_state['value'], state.observation,
            rngs={'default': key_value_out},
        )
        prior_value = self.value_model.sample(key_value_sample, (), value_out)

        data = {
            'action': action,
            'log_prob': logprob,
            'value': prior_value
        }

        return policy_state, action, data

    def reset(
            self,
            params: optax.Params
    ) -> dict[Literal['policy', 'value'], optax.Params]:
        return {
            'policy': {'params': params['params']['policy']},
            'value': {'params': params['params']['value']},
        }

    def update(
            self,
            params: optax.Params,
            state: dict[Literal['policy', 'value'], optax.Params]
    ) -> dict[Literal['policy', 'value'], optax.Params]:
        return self.reset(params)  # Simply throw away previous state

    def unpack(
            self,
            policy_state:  dict[Literal['policy', 'value'], optax.Params]
    ) -> optax.Params:
        return {
            'params': {
                'policy': policy_state['policy'],
                'value': policy_state['value']
            }
        }
