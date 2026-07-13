# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

import numpy as np
import numpy.typing as npt
from typing import Tuple

class RiverSwim(object):
    """RiverSwim environment
    @See also https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1374179
    """
    ns: int                                 # Number of states
    na: int                                 # Number of actions
    min_reward: float                       # Minimum reward
    max_reward: float                       # Maximum reward
    transitions: npt.NDArray[np.float64]    # Transition function P(s'|s,a)
    rewards: npt.NDArray[np.float64]        # Rewards r(s,a,s')
    current_state: int                      # CUrrent state
    
    def __init__(self, 
                 num_states: int = 5,
                 min_reward: float = 0.05,
                 max_reward: float = 1):
        """Initialize a river swim environment

        Parameters
        ----------
        num_states : int, optional
            Maximum number of states, by default 5
        min_reward : float, optional
            Minimum reward obtainable in state 0, by default 0.05
        max_reward : float, optional
            Maximum reward obtainable in the last state, by default 1
        """        
        self.ns = num_states
        self.na = 2
        self.min_reward = min_reward
        self.max_reward = max_reward
        
        self.rewards = np.zeros((self.ns, self.na))
        self.transitions = np.zeros((self.ns, self.na, self.ns))
        
        # Create rewards
        self.rewards[0, 0] = min_reward
        self.rewards[-1, 1] = max_reward
        
        # Create transitions
        for s in range(1, self.ns-1):
            self.transitions[s, 1, s] = 0.6
            self.transitions[s, 1, s-1] = 0.1
            self.transitions[s, 1, s+1] = 0.3
        self.transitions[1:-1, 0, 0:-2] = np.eye(num_states-2)

        self.transitions[0, 0, 0] = 1
        self.transitions[0, 1, 0] = 0.7
        self.transitions[0, 1, 1] = 0.3
        self.transitions[-1, 1, -1] = 0.3
        self.transitions[-1, 1, -2] = 0.7
        self.transitions[-1, 0, -2] = 1
        
        # Reset environment
        self.reset()
    
    def reset(self) -> int:
        """Reset the current state

        Returns
        -------
        int
            initial state 0
        """        
        self.current_state = 0
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float]:
        """Take a step in the environment

        Parameters
        ----------
        action : int
            An action (0 or 1)

        Returns
        -------
        Tuple[int, float]
            Next state and reward
        """        
        assert action == 0 or action == 1, 'Action needs to either 0 or 1'
        
        next_state = np.random.choice(self.ns, p=self.transitions[self.current_state, action])
        reward = 1. if np.random.uniform()  < self.rewards[self.current_state, action] else 0.
        self.current_state = next_state
        return next_state, reward
    

if __name__ == '__main__':
    from RiverSwimExperiments.utils.new_mdp_description import MDPDescription2
    env = RiverSwim()
    gamma = 0.95
    mdp = MDPDescription2(env.transitions, env.rewards, gamma, 1)
    print(mdp.compute_allocation(navigation_constraints=True))