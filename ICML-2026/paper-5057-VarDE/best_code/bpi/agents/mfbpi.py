# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

import numpy as np
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple
import sys
import cvxpy as cp
sys.path.append("../..")
from scipy.stats import t
from utils.simplified_new_mdp_description import SimplifiedNewMDPDescription

golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2

class MFBPIParameters(NamedTuple):
    kbar: int
    ensemble_size: int

class MFBPI(Agent):
    """ Model Free BPI Algorithm """

    def __init__(self, parameters: MFBPIParameters, agent_parameters: AgentParameters):
        # Initialize parent class with agent parameters
        super().__init__(agent_parameters)
        self.parameters = parameters
        self.ensemble_size = parameters.ensemble_size

        # Create a uniform policy matrix
        self.uniform_policy = np.ones((self.ns, self.na)) / (self.ns * self.na)

        # Initialize Q_greedy matrix
        self.Q_greedy = np.ones((self.ns, self.na)) / (self.ns * self.na) / (1 - self.discount_factor)

        # Initialize Q-table and M-table for each ensemble member
        if self.ensemble_size > 1:
            self.Q = np.tile(np.linspace(0, 1, self.ensemble_size)[:, None, None], (1, self.ns, self.na)) / (1 - self.discount_factor)
            self.M = np.tile(np.linspace(0, 1, self.ensemble_size)[:, None, None], (1, self.ns, self.na)) / ((1 - self.discount_factor) ** (2 ** self.parameters.kbar))

            self.Q = self.Q.flatten()
            self.M = self.M.flatten()

            # Shuffle the Q and M matrices
            np.random.shuffle(self.Q)
            np.random.shuffle(self.M)
            self.Q = self.Q.reshape(self.ensemble_size, self.ns, self.na)
            self.M = self.M.reshape(self.ensemble_size, self.ns, self.na)
        else:
            self.Q = np.ones((1, self.ns, self.na)) / (1 - self.discount_factor)
            self.M = np.ones((1, self.ns, self.na)) / ((1 - self.discount_factor) ** (2 ** self.parameters.kbar))

        # Initialize omega and policy matrices
        self.omega = np.ones(shape=(self.ns, self.na)) / (self.ns * self.na)
        self.policy = np.ones(shape=(self.ns, self.na)) / (self.na)

        # Initialize a visits matrix for each ensemble member
        self._visits = np.zeros((self.ensemble_size, self.ns, self.na, self.ns))

    @staticmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        return 1

    def forward(self, state: int, step: int) -> int:
        epsilon = self.forced_exploration_callable(state, step, minimum_exploration=1e-3)
        omega = (1-epsilon) * self.policy[state] + epsilon * np.ones(( self.na)) / ( self.na)
        return np.random.choice(self.na, p=omega)

    def compute_omega(self):
        # If there's only one ensemble member, use its Q and M values
        if self.ensemble_size == 1:
            q_values = self.Q[0]
            m_values = self.M[0]
        else:
            # If there are multiple ensemble members, sample a random value from the uniform distribution
            x = np.random.uniform()
            q_values = np.quantile(self.Q,x  ,axis=0)
            m_values = np.quantile(self.M, x,  axis=0)

        # Compute the greedy policy
        greedy_policy = q_values.argmax(1)
        
        # Identify the suboptimal actions
        idxs_subopt_actions = np.array([
            [False if greedy_policy[s] == a else True for a in range(self.na)] for s in range(self.ns)]).astype(np.bool_)

        # Compute Delta
        delta = np.clip((q_values.max(-1, keepdims=True) - q_values) , a_min=1e-8, a_max=None)
        delta_subopt = delta[idxs_subopt_actions]
        delta_min = delta_subopt.min()
    
        # Update Delta for optimal actions
        delta[~idxs_subopt_actions] = delta_min * (1 - self.discount_factor) / (1 + self.discount_factor)
       
        # Compute Hsa
        Hsa = (2 + 8 * golden_ratio_sq * m_values) / (delta ** 2)

        C = np.max(np.maximum(4, 16 * (self.discount_factor ** 2) * golden_ratio_sq * m_values[~idxs_subopt_actions]))

        Hopt = C / (delta[~idxs_subopt_actions] ** 2)
        
        # Update Hsa for optimal actions
        Hsa[~idxs_subopt_actions] = np.sqrt(Hopt * Hsa[idxs_subopt_actions].sum() / self.ns )
  
        # Compute omega and update policy
        self.omega = Hsa/Hsa.sum()
        self.policy = self.omega / self.omega.sum(-1, keepdims=True)

    
    def process_experience(self, experience: Experience, step: int) -> None:
        # Unpack the experience tuple
        s, a, r, sp = experience.s_t, experience.a_t, experience.r_t, experience.s_tp1

        # Randomly select a subset of the ensemble
        idxs = np.random.choice(self.ensemble_size, size=int(.7 * self.ensemble_size), replace=False)

        # Update visit counts
        self._visits[idxs, s, a, sp] += 1
        k = self._visits[idxs, s, a].sum(-1)
        H = 1 / (1 - self.discount_factor)
        alpha_t = (H + 1) / (H + k)

        # Compute beta_t
        beta_t = alpha_t ** 1.1

        # Calculate target Q value
        target = r + self.discount_factor * self.Q[idxs, sp].max(-1)
        self.Q[idxs, s, a] = (1 - alpha_t) * self.Q[idxs, s, a] + alpha_t * target

        # Update Q_greedy
        k = self.exp_visits[s, a].sum()
        alpha_t = (H + 1) / (H + k)
        target = r + self.discount_factor * self.Q_greedy[sp].max(-1)
        self.Q_greedy[s, a] = (1 - alpha_t) * self.Q_greedy[s, a] + alpha_t * target

        # Update M values
        delta = (r + self.discount_factor * self.Q[idxs, sp].max(-1) - self.Q[idxs, s, a]) / self.discount_factor
        self.M[idxs, s, a] = (1 - beta_t) * self.M[idxs, s, a] + beta_t * (delta ** (2 ** self.parameters.kbar))

        # Update the greedy policy
        self.greedy_policy = (np.random.random(self.Q_greedy.shape) * (self.Q_greedy == self.Q_greedy.max(-1, keepdims=True))).argmax(-1)

        # Update the ensemble head
        self._head = np.random.choice(self.ensemble_size)

        # Recompute omega values and update the policy
        self.compute_omega()
