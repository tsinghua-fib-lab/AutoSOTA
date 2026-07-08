from typing import Protocol, Any

import flax.linen as nn

from smz.planning import impl
from . import networks, blocks


class ModelNameSpace(Protocol):

    QValueModel: impl.ActionValueProtocol
    ValueModel: impl.ValueProtocol

    class PolicyModel[
        State, Action, Params, DistributionParams
    ](impl.PolicyProtocol[State, Action, Params, DistributionParams]):

        def entropy(self, dist_params: DistributionParams) -> float:
            ...


class JointModel(nn.Module):
    """Helper class to combine all model components.

    Combining this is useful to reduce manual parameter/ state management.
    """
    policy: nn.Module | impl.PolicyProtocol | None = None
    value: nn.Module | impl.ValueProtocol | None = None
    q_value: nn.Module | impl.ActionValueProtocol | None = None

    @nn.compact
    def __call__(self, obs, act) -> dict[str, Any]:
        """Initializes network parameters in one joint container."""
        return {
            'policy': None if self.policy is None else self.policy(obs),
            'value': None if self.value is None else self.value(obs),
            'q_value': None if self.q_value is None else self.q_value(obs, act)
        }
