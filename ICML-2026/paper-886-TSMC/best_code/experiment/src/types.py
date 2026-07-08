from typing import Protocol, NamedTuple, Any
from typing_extensions import Self

import os

from jaxtyping import PRNGKeyArray

type MetricData = dict[str, Any]


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs) -> Self:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def __hash__(self):
        return 0


class Identity(metaclass=Singleton):
    """Global setup identity per run."""

    def __init__(
            self,
            path: str | None = None,
            name: str | None = None,
            token: str | None = None,
            debug: bool | None = None,
    ):
        self._path = path
        self._name = name
        self._token = token
        self._debug = debug

    @property
    def debug(self):
        return self._debug

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name

    @property
    def token(self):
        return self._token

    def make_path(self):
        return os.path.join(self.path, self.name, self.token)


class Experimenter(Protocol):

    def run(self) -> None:
        ...


class HasConfigLoader(Protocol):

    def from_config(
            self,
            config_dict: dict,
            service: None  # Missing typing stubs by Wandb
    ) -> Experimenter:
        ...


class DataGen[EnvData, Params, Action, GeneratorState](Protocol):

    def reset(self) -> None:
        ...

    def sample_data(
            self,
            key: PRNGKeyArray,
            params: Params,
            previous_state: GeneratorState | None
    ) -> tuple[GeneratorState, tuple[EnvData, Action, Any]]:
        ...


class Policy[PolicyParams, PolicyState, State, Observation, Action](Protocol):

    def __call__(
            self,
            key: PRNGKeyArray,
            policy_state: PolicyState,
            obs: Observation,
            state: State | None = None,
            train: bool = True
    ) -> tuple[PolicyState, Action, Any]:
        ...

    def reset(self, params: PolicyParams) -> PolicyState:
        ...

    def update(self, params: PolicyParams, state: PolicyState) -> PolicyState:
        ...

    def unpack(self, policy_state: PolicyState) -> PolicyParams:
        ...


class Learner[
    LearnerState, LearnerVariables, LearnerHyperparams: NamedTuple
](Protocol):
    # Separate the namespace for hyperparameters (which can be very large)!
    param: LearnerHyperparams

    def init(self, key: PRNGKeyArray) -> tuple[LearnerState, LearnerVariables]:
        ...

    def update(
            self,
            state: LearnerState,
            variables: LearnerVariables,
            key: PRNGKeyArray,
            data: Any
    ) -> tuple[tuple[LearnerState, LearnerVariables], dict[str, Any]]:
        ...


class Logger(Protocol):

    def log(self, data: dict[str, Any]) -> None:
        ...
