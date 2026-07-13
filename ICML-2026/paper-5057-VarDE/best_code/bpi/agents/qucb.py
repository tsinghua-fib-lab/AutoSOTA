# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

import numpy as np
from numpy.typing import NDArray
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple

# Define QUCBParameters as a named tuple with a single attribute 'confidence'.
class QUCBParameters(NamedTuple):
    confidence: float

# Define QUCB class which inherits from Agent class.
class QUCB(Agent):
    def __init__(self, parameters: QUCBParameters, agent_parameters: AgentParameters):
        super().__init__(agent_parameters)
        # Initialize Q-table with initial values.
        self.Q = np.ones((self.ns, self.na)) / (1 - self.discount_factor)
        self.parameters = parameters

    # Static method that suggests an exploration parameter (1 in this case).
    @staticmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        return 1

    # Given a state and the current step, return the action to take.
    def forward(self, state: int, step: int) -> int:
        # Choose a random action with probability determined by the exploration strategy.
        if np.random.uniform() < self.forced_exploration_callable(state, step, minimum_exploration=1e-3):
            return np.random.choice(self.na)
        
        # Otherwise, choose the action with the highest Q-value for the given state.
        return self.Q[state].argmax()

    # Update the Q-table and policy based on the experience (s, a, r, sp) and current step.
    def process_experience(self, experience: Experience, step: int) -> None:
        s, a, r, sp = experience.s_t, experience.a_t, experience.r_t, experience.s_tp1
        k = self.exp_visits[s, a].sum()
        
        H = 1 / (1-self.discount_factor)
        alpha_t = (H + 1) / (H + k)
        
        T = step + 1
        iota = np.log(self.ns * self.na * T / self.parameters.confidence)
        b_t = 1e-3 * H * np.sqrt(iota / k)
        
        # Update Q-value for the given state-action pair.
        target = r + self.discount_factor * self.V[sp] + b_t
        self.Q[s,a] = (1 - alpha_t) * self.Q[s,a] + alpha_t * target
        
        # Update the greedy policy.
        self.greedy_policy = (np.random.random(self.Q.shape) * (self.Q==self.Q.max(1, keepdims=True))).argmax(1)
      
    # Return the value function V(s) for all states based on the maximum Q-value.
    @property
    def V(self) -> NDArray[np.float64]:
        return np.minimum(1/(1-self.discount_factor), self.Q.max(1))
