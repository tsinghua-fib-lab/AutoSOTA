from typing import Literal, Sequence

import jax
import jax.numpy as jnp

import optax

import mctx

from jaxtyping import PRNGKeyArray

from smz import Transition
from smz.planning import impl


class MCTXComponents[State, Action, Param]:

    def __init__(
            self,
            value_model: impl.ValueProtocol[
                State, Param, jax.Array | Sequence[jax.Array]
            ],
            policy_model: impl.PolicyProtocol[  # \pi(a | s)
                State, Action, Param, jax.Array | Sequence[jax.Array]
            ],
            transition: Transition[State, Action, None],
            bootstrap: int | None
    ):
        self.value_model = value_model
        self.policy_model = policy_model

        self.transition = transition

        # (Optionally) Bootstrap from prior for continuous/ large action spaces
        # Note that in MCTS it is common to make the breadth for continous
        # spaces adaptive based on the visitation count. In jax this must be
        # predetermined (fixed). Thus, the value for `bootstrap` should be
        # tuned based on the `budget` to the MCTS planner.
        self.bootstrap = bootstrap

    def recurrent_fn(
            self,
            params: dict[Literal['policy', 'value'], Param],
            key: PRNGKeyArray,
            action: Action,
            recurrent_state: tuple[State, jax.Array]
    ) -> tuple[mctx.RecurrentFnOutput, tuple[State, jax.Array]]:
        key_transition, key_expand = jax.random.split(key)
        state, atoms = recurrent_state

        # Get the sampled atom from the discrete action
        true_action = atoms.at[action].get()
        new_state, reward, discount = self.transition(
            None, key_transition, state, true_action
        )

        state_output = self.expand_node(params, key_expand, new_state)

        return mctx.RecurrentFnOutput(
            reward=reward, discount=discount,   # type: ignore
            prior_logits=state_output.prior_logits,  # type: ignore
            value=state_output.value  # type: ignore
        ), state_output.embedding

    def expand_node(
        self,
        params: dict[Literal['policy', 'value'], Param],
        key: PRNGKeyArray,
        state: State
    ) -> mctx.RootFnOutput:
        key_policy, key_value, key_boot = jax.random.split(key, 3)

        # Sample atoms (or take them directly) from the policy model
        policy_out = self.policy_model.apply(
            params['policy'], state.observation, rngs={'default': key_policy}
        )

        if self.bootstrap is not None:  # Sampled AlphaZero/ MuZero
            # Uniform prior to prevent double-sampling
            log_pi = jnp.zeros(self.bootstrap)
            atoms = self.policy_model.sample(
                key_boot, (self.bootstrap,), policy_out
            )
        else:
            log_pi = policy_out
            atoms = jnp.arange(log_pi.size)

        # Ensure a proper PMF over the atoms
        log_pi = jax.nn.log_softmax(log_pi).squeeze()

        # Compute (estimate of) the value at the current state
        key_value_out, key_value_sample = jax.random.split(key_value)
        value_out = self.value_model.apply(
            params['value'], state.observation,
            rngs={'default': key_value_out},
        )
        prior_value = self.value_model.sample(key_value_sample, (), value_out)

        return mctx.RootFnOutput(
            prior_logits=log_pi,  # type: ignore
            value=prior_value.squeeze(),  # type: ignore
            embedding=(state, atoms)  # type: ignore
        )


class MCTXPolicy[State, Observation, Action]:

    def __init__(
            self,
            components: MCTXComponents,
            budget: int,
            max_depth: int,
            max_breadth_root: int = 16,
            stochastic_eval: bool = True
    ):
        self.components = components

        # Search constraints
        self.budget = budget
        self.max_depth = min(max_depth, budget)  # max depth <= budget
        self.max_breadth_root = max_breadth_root

        self.stochastic_eval = stochastic_eval  # TODO: implement. Currently uses the default of GMZ, which is det. enough.

    def __call__(
            self,
            key: PRNGKeyArray,
            policy_state: dict[Literal['policy', 'value'], optax.Params],
            obs: Observation,
            state: State | None = None,
            train: bool = True
    ) -> tuple[..., Action, dict[str, jax.Array]]:
        key_search, key_root = jax.random.split(key)

        root = self.components.expand_node(policy_state, key_root, state)
        _, atoms = root.embedding  # Extract (sampled) action atoms

        # Planner expects (batch, ...) prefixed data and outputs (batch, ...)
        policy_output = mctx.gumbel_muzero_policy(
            params=policy_state,
            rng_key=key_search,
            root=jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), root),
            recurrent_fn=jax.vmap(  # type: ignore
                self.components.recurrent_fn, in_axes=(None, None, 0, 0)
            ),
            num_simulations=self.budget,
            invalid_actions=None,
            max_depth=self.max_depth,
            max_num_considered_actions=self.max_breadth_root,
            gumbel_scale=1.0  # Do not change; adjust logit temperature instead
        )
        # Remove batch-index
        # This action maximizes `gumbel_a + logits(s, a) + f(Q(s, a))`. Due to
        # the gumbel noise (gumbel-max trick) this action is a sample from the
        # softmax distribution: a ~ softmax(logits(s, a) + f(Q(s, a)))`.
        # I.e., the MPO posterior: pi(a | s) exp(f(Q(s, a))) / Z
        index = policy_output.action[0]
        true_action = atoms.at[index].get()

        summary = policy_output.search_tree.summary()
        root_value = summary.value[0]
        q_value = summary.qvalues[0].at[index].get()

        data = {
            'atoms': atoms,
            'pmf': policy_output.action_weights[0],
            'value': root_value,
            'q_value': q_value
        }

        return policy_state, true_action, data

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
