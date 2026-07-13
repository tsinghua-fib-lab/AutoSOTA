# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

import numpy as np
import numpy.typing as npt
from typing import Tuple

class ForkedRiverSwim(object):
    """Forked RiverSwim environment
    Like the RiverSwim environment but with 2 rivers
    
    0 1 2 3 4 <- 1st branch
    | | | |
    - 5 6 7 8 <- 2nd branch
    """
    ns: int                                 # Number of states
    na: int                                 # Number of actions
    min_reward: float                       # Minimum reward
    max_reward: float                       # Maximum reward
    transitions: npt.NDArray[np.float64]    # Transition function P(s'|s,a)
    rewards: npt.NDArray[np.float64]        # Rewards r(s,a,s')
    current_state: int                      # CUrrent state
    
    _LEFT: int = 0
    _RIGHT: int = 1
    _SWITCH: int = 2
    
    def __init__(self, 
                 river_length: int = 5,
                 min_reward: float = 0.05,
                 max_reward_river1: float = 1,
                 max_reward_river2: float = 0.95):
        """Initialize a forked river swim environment

        Parameters
        ----------
        river_length : int, optional
            Length of each river branch
        min_reward : float, optional
            Minimum reward obtainable in state 0, by default 0.05
        max_reward_river1 : float, optional
            Maximum reward obtainable in the last state of river 1, by default 1
        max_reward_river2 : float, optional
            Maximum reward obtainable in the last state of river 2, by default 0.95
        """        
        
        self.ns = 1 + (river_length - 1) * 2
        self.na = 3
        self.min_reward = min_reward
        self.max_reward_river1 = max_reward_river1
        self.max_reward_river2 = max_reward_river2
        
        self._end_river_1 = river_length - 1
        self._end_river_2 = self.ns - 1
        self._start = 0
        
        self.rewards = np.zeros((self.ns, self.na))
        self.transitions = np.zeros((self.ns, self.na, self.ns))
        
        # Create rewards
        self.rewards[self._start, self._LEFT] = min_reward
        self.rewards[self._end_river_1, self._RIGHT] = max_reward_river1
        self.rewards[self._end_river_2, self._RIGHT] = max_reward_river2
        
        # Create transitions
        for start, end in [(1, self._end_river_1), (self._end_river_1 + 1, self._end_river_2)]:
            for s in range(start, end):
                self.transitions[s, self._RIGHT, s] = 0.6
                self.transitions[s, self._RIGHT, s-1] = 0.1
                self.transitions[s, self._RIGHT, s+1] = 0.3
                
                other_side = s + river_length - 1 if s < self._end_river_1 else s - river_length + 1
                self.transitions[s, self._SWITCH, other_side] = 1
            
        self.transitions[1:self._end_river_1, self._LEFT, 0:self._end_river_1-1] = np.eye(river_length - 2)
        self.transitions[self._end_river_1+2:self._end_river_2, self._LEFT, self._end_river_1+1:self._end_river_2-1] = np.eye(river_length - 3)
        self.transitions[self._end_river_1+1, self._LEFT, 0] = 1

        self.transitions[0, self._LEFT, 0] = 1
        self.transitions[0, self._RIGHT, 0] = 0.7
        self.transitions[0, self._RIGHT, 1] = 0.3
        for end in [self._end_river_1, self._end_river_2]:
            self.transitions[end, self._RIGHT, end] = 0.3
            self.transitions[end, self._RIGHT, end-1] = 0.7
            self.transitions[end, self._LEFT, end-1] = 1
            
        self.transitions[self._end_river_1, self._SWITCH, self._end_river_1] = 1
        self.transitions[self._end_river_2, self._SWITCH, self._end_river_2] = 1
        self.transitions[self._start, self._SWITCH, self._start] = 1
 
        
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
        assert action == 0 or action == 1 or action == 2, 'Action needs to either 0 or 1'
        
        next_state = np.random.choice(self.ns, p=self.transitions[self.current_state, action])
        reward = 1 if np.random.uniform()  < self.rewards[self.current_state, action] else 0
        self.current_state = next_state
        return next_state, reward
    

if __name__ == '__main__':
    from RiverSwimExperiments.utils.new_mdp_description import MDPDescription2
    env = ForkedRiverSwim()
    gamma = 0.95
    mdp = MDPDescription2(env.transitions, env.rewards, gamma, 1)
    print(mdp.compute_allocation(navigation_constraints=True))
    import pdb
    pdb.set_trace()