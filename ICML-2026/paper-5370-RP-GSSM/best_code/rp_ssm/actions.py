from typing import Protocol

import flax.linen as nn

import jax.numpy as np
from jax import Array, vmap
from jax.nn import softmax

from rp_ssm.types import TransitionMatrixType, TransitionBiasType


class ActionMapper(Protocol):

    def init(
        self,
        key: Array,
        dim: int,
        transition_matrix: TransitionMatrixType,
        transition_bias: TransitionBiasType,
        actions: Array,  # of shape BxTxA
    ) -> dict[str, Array]:
        ...

    def apply(self, actions: Array, params: dict[str, Array]) -> Array:
        """
        Map a series of actions {a_t} to transition
        matrices {B(a_t)} and biases {b(a_t)}.
        """
        ...


class ActionNetwork(nn.Module):
    """
    Class that concatenates a neural network
    with a linear projection layer to ensure
    the output is compatible with the shape
    of the latent transition matrix.
    """

    network: nn.Module
    latent_dim: int

    @nn.compact
    def __call__(self, x: Array) -> tuple[Array, Array]:
        x = self.network(x) * 1e-2
        C = nn.Dense(self.latent_dim**2)(x).reshape((self.latent_dim, self.latent_dim)) * 1e-2
        c = nn.Dense(self.latent_dim)(x)
        return C, c


class DiscreteChoiceActionMapper(ActionMapper):

    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def init(
        self,
        key: Array,
        dim: int,
        transition_matrix: TransitionMatrixType,
        transition_bias: TransitionBiasType,
        actions: Array,
    ) -> dict[str, Array]:
        assert self.num_actions >= len(
            np.unique(actions)
        ), "Number of model actions must be at least as large as number of actions in the data"

        if transition_matrix == "constant":
            C = None
        else:
            C = np.tile(
                np.eye(dim), (self.num_actions, 1, 1)
            )
        if transition_bias in ["constant", "none"]:
            c = None
        elif transition_bias == "action_dependent":
            c = np.zeros(
                (self.num_actions, dim)
            )

        all_params = {"C": C, "c": c}

        return {k: v for k, v in all_params.items() if v is not None}

    def apply(self, actions: Array, params: dict[str, Array]) -> tuple[Array, Array]:
        if "C" in params:
            Cs = np.take(params["C"], actions.squeeze(), axis=0)
        else:
            Cs = 0.0
        if "c" in params:
            cs = np.take(params["c"], actions.squeeze(), axis=0)
        else:
            cs = 0.0

        return Cs, cs


class DiscreteNetworkActionMapper(ActionMapper):

    def __init__(self, network: nn.Module):
        self.network = network

    def init(self, key: Array, dim: int) -> dict[str, Array]:
        raise NotImplementedError

    def apply(self, actions: Array, params: dict[str, Array]) -> Array:
        logits = self.network.apply(params, actions)
        return softmax(logits)


class ContinuousActionMapper(ActionMapper):

    def __init__(self, action_network: ActionNetwork):
        self.action_network = action_network

    def init(
        self,
        key: Array,
        actions: Array,  # of shape BxTxA
    ) -> dict[str, Array]:
        params = self.action_network.init(key, actions[0, 0])
        return params

    def apply(self, actions: Array, params: dict[str, Array]) -> Array:
        action_params = {"params": params["params"]}
        out = vmap(lambda act: self.action_network.apply(action_params, act))(actions)
        return out
