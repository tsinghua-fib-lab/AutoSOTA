# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

import numpy as np
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple
import sys
sys.path.append("../..")

from utils.simplified_new_mdp_description import SimplifiedNewMDPDescription

golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2

class OBPIParameters(NamedTuple):
    frequency_computation: int
    kbar: int

class OBPI(Agent):
    """ O-BPI Algorithm """

    def __init__(self, parameters: OBPIParameters, agent_parameters: AgentParameters):
        super().__init__(agent_parameters)
        self.parameters = parameters
        self.uniform_policy = np.ones((self.ns, self.na)) / (self.ns * self.na)
        self.Q = np.zeros((self.ns, self.na), dtype=np.float64, order='C')
        self.M = np.zeros((self.ns, self.na), dtype=np.float64, order='C')
        self.frequency_computation = self.parameters.frequency_computation
        self.state_action_visits_copy = self.state_action_visits.copy()
        self.max_steps = 2000 * self.horizon / 50000

    @staticmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        return 1

    def forward(self, state: int, step: int) -> int:
        epsilon = self.forced_exploration_callable(state, step, minimum_exploration=1e-3)
        omega = (1-epsilon) * self.omega + epsilon * self.uniform_policy
        omega = omega[state] / omega[state].sum()
        return np.random.choice(self.na, p=omega)
    
    def process_experience(self, experience: Experience, step: int) -> None:
        s, a, r, sp = experience.s_t, experience.a_t, experience.r_t, experience.s_tp1
        
        k = self.exp_visits[s, a].sum()
        H = 1 / (1-self.discount_factor)
        alpha_t = (H + 1) / (H + k)
        
        beta_t = alpha_t ** 1.1
                
        ## Update Q
        target = r + self.discount_factor * self.Q[sp].max()
        self.Q[s,a] = (1 - alpha_t) * self.Q[s,a] + alpha_t * target
        
        ## Update V
        delta = (r + self.discount_factor * self.Q[sp].max()- self.Q[s,a]) / self.discount_factor
        self.M[s,a] = self.M[s,a] + beta_t * (delta ** (2 ** self.parameters.kbar)  - self.M[s,a])


        if step % self.frequency_computation == 0 or self.state_action_visits[s,a] >= 2 * self.state_action_visits_copy[s,a]:   
            self.prev_omega = self.omega.copy() 
            
            self.omega = SimplifiedNewMDPDescription.compute_mf_allocation(
                self.discount_factor, self.Q, self.M ** (2 ** (1 - self.parameters.kbar)), self.exp_visits, navigation_constraints=True
            )

            if self.state_action_visits[s,a] >= 2 * self.state_action_visits_copy[s,a]:
                self.state_action_visits_copy = self.state_action_visits.copy()

            slope = max(self.parameters.frequency_computation, self.max_steps * (step) / (self.horizon * 0.5))
            self.frequency_computation = min(self.max_steps, int(slope))
            

        self.greedy_policy = (np.random.random(self.Q.shape) * (self.Q==self.Q.max(1)[:,None])).argmax(1)
        
        
        
        
        


    
