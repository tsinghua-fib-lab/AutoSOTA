from typing import Union, Any, Literal, Callable

from flax.core.frozen_dict import FrozenDict

from jax import Array

PriorParams = dict[str, Array]
NetworkParams = Union[FrozenDict, dict[str, Any]]
AllParams = tuple[PriorParams, NetworkParams]
TransitionMatrixType = Literal["constant", "additive", "multiplicative"]
TransitionBiasType = Literal["constant", "action_dependent", "none"]
LearningRate = Union[float, Callable[[int], float]]