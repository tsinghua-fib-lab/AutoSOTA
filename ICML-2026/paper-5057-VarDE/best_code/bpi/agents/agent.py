# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

from __future__ import annotations
import numpy.typing as npt
import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple, Callable

# Define a named tuple to store experience data
class Experience(NamedTuple):
    s_t: int     # State at time t
    a_t: int     # Action at time t
    r_t: float   # Reward at time t
    s_tp1: int   # State at time t+1

# Define a named tuple for agent parameters
class AgentParameters(NamedTuple):
    dim_state_space: int     # Dimension of state space
    dim_action_space: int    # Dimension of action space
    discount_factor: float   # Discount factor for future rewards
    horizon: int             # Horizon (time steps) to plan ahead

# Define an abstract agent class
class Agent(ABC):
    # Class attributes
    dim_state_space: int
    dim_action_space: int
    discount_factor: float
    exp_visits: npt.NDArray[np.float64]
    total_state_visits: npt.NDArray[np.float64]
    last_visit: npt.NDArray[np.float64]
    greedy_policy: npt.NDArray[np.int64]
    omega: npt.NDArray[np.float64]
    horizon: int

    # Initialize the agent with agent parameters
    def __init__(self, agent_parameters: AgentParameters):
        self.dim_state_space = agent_parameters.dim_state_space
        self.dim_action_space = agent_parameters.dim_action_space
        self.discount_factor = agent_parameters.discount_factor
        self.exp_visits = np.zeros((self.ns, self.na, self.ns), order='C')
        self.state_action_visits = np.zeros((self.ns, self.na), order='C')
        self.total_state_visits = np.zeros((self.ns), order='C')
        self.last_visit = np.zeros((self.ns), order='C')
        self.greedy_policy = np.zeros((self.ns), dtype=np.int64, order='C')
        self.omega = np.ones((self.ns, self.na), order='C')
        self.exploration_parameter = self.suggested_exploration_parameter(self.ns, self.na)
        self.horizon = agent_parameters.horizon

    # Property getter for state space dimension
    @property
    def ns(self) -> int:
        return self.dim_state_space

    # Property getter for action space dimension
    @property
    def na(self) -> int:
        return self.dim_action_space

    # Abstract static method to return the suggested exploration parameter
    @staticmethod
    @abstractmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        return 1.

    # Method to compute forced exploration probability
    def forced_exploration_callable(self, state: int, step: int, minimum_exploration: float = 0.1) -> float:
        return max(minimum_exploration, (1 / max(1, self.total_state_visits[state])) ** self.exploration_parameter)

    # Abstract method for forward pass
    @abstractmethod
    def forward(self, state: int, step: int) -> int:
        raise NotImplementedError

    # Abstract method for processing experience
    @abstractmethod
    def process_experience(self, experience: Experience, step: int) -> None:
        raise NotImplementedError

    # Method for backward pass (update agent)
    def backward(self, experience: Experience, step: int) -> None:
        # Increment visit count for the current state-action pair
        self.exp_visits[experience.s_t, experience.a_t, experience.s_tp1] += 1
        self.state_action_visits[experience.s_t, experience.a_t] += 1
        
        # Update last visit time and total state visits count for the next state
        self.last_visit[experience.s_tp1] = step + 1
        self.total_state_visits[experience.s_tp1] += 1
        
        # If this is the first time step, update last visit time and total state visits count for the current state
        if step == 0:
            self.last_visit[experience.s_t] = step
            self.total_state_visits[experience.s_t] += 1

        # Process the experience to update the agent's internal model
        self.process_experience(experience, step)
