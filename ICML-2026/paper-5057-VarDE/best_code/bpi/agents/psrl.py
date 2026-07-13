# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

import numpy as np
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple, Optional
import sys
sys.path.append("../..")

from utils.mdp_description import MDPDescription
from utils.new_mdp_description import NewMDPDescription
from utils.simplified_new_mdp_description import SimplifiedNewMDPDescription
from utils.posterior import PosteriorProbabilisties
from utils.utils import policy_iteration
from enum import Enum

class PSRL(Agent):
    """ PSRL Algorithm """

    def __init__(self,  agent_parameters: AgentParameters):
        super().__init__(agent_parameters)
        self.posterior = PosteriorProbabilisties(self.ns, self.na, prior_p=1, prior_r=1)
        self.greedy_policy = np.zeros(self.ns, dtype=int)
        self.V = np.zeros(self.ns)
        self.state_action_visits_copy = self.state_action_visits.copy()

    @staticmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        return 1
    def forward(self, state: int, step: int) -> int:
        if np.random.uniform() <= self.forced_exploration_callable(state, step, minimum_exploration=1e-3):
            return np.random.choice(self.na)
        return self.greedy_policy[state]
    
    def process_experience(self, experience: Experience, step: int) -> None:
        s, a, r, sp = experience.s_t, experience.a_t, experience.r_t, experience.s_tp1
        self.posterior.update(s, a, sp, r)

        if step % (np.ceil(1 / (1 - self.discount_factor))) == 0 or self.state_action_visits[s,a] >= 2 * self.state_action_visits_copy[s,a]:
            for _ in range(30):
                P, R = self.posterior.sample_posterior()
                V, pi, _ = policy_iteration(self.discount_factor, P, R, self.greedy_policy, self.V)
                if not np.array_equal(pi, self.greedy_policy):
                    break
            
            if self.state_action_visits[s,a] >= 2 * self.state_action_visits_copy[s,a]:
                self.state_action_visits_copy[s,a] = self.state_action_visits[s,a]

            self.V=V
            self.greedy_policy = pi

            


    
