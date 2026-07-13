# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

# Import necessary libraries
import numpy as np
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple

# Define an empty class for QLearningParameters as a NamedTuple
class QLearningParameters(NamedTuple):
    pass

# Define the QLearning class which inherits from Agent base class
class QLearning(Agent):
    """ Classical Qlearning agent """

    # Initialize the QLearning agent with given parameters and agent_parameters
    def __init__(self, parameters: QLearningParameters, agent_parameters: AgentParameters):
        # Call the Agent base class's constructor
        super().__init__(agent_parameters)

        # Initialize Q-table with initial values
        self.Q = np.ones((self.ns, self.na)) / (1 - self.discount_factor)
        self.parameters = parameters

    # Define a method to suggest exploration parameter based on state and action dimensions
    @staticmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        return 1

    # Define the forward method to select an action given the current state and step
    def forward(self, state: int, step: int) -> int:
        # Perform exploration with probability determined by the exploration callable
        if np.random.uniform() < self.forced_exploration_callable(state, step, minimum_exploration=1e-3):
            return np.random.choice(self.na)

        # Otherwise, select action with the highest Q-value
        return self.Q[state].argmax()

    # Define the method to process the experience and update the Q-table
    def process_experience(self, experience: Experience, step: int) -> None:
        # Calculate the learning rate (alpha) based on the number of visits to the current state-action pair
        k = self.exp_visits[experience.s_t, experience.a_t].sum()
        H = 1 / (1 - self.discount_factor)
        alpha_t = (H + 1) / (H + k)

        # Calculate the target Q-value using the reward and the max Q-value of the next state
        target = experience.r_t + self.discount_factor * self.Q[experience.s_tp1].max()

        # Update the Q-value for the current state-action pair using the learning rate and target Q-value
        self.Q[experience.s_t, experience.a_t] = (1 - alpha_t) * self.Q[experience.s_t, experience.a_t] + alpha_t * target

        # Update the greedy_policy with the current optimal actions
        self.greedy_policy = (np.random.random(self.Q.shape) * (self.Q == self.Q.max(1, keepdims=True))).argmax(1)