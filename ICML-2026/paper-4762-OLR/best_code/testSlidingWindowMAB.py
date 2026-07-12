"""
Multi-Armed Bandit (MAB) Sliding Window Simulation

This module implements a multi-armed bandit framework with sliding window functionality
for baseline synthetic data experiments. It provides:
- Bernoulli arm implementation for bandit problems
- Arm reading buffer for streaming simulation
- Regret computation for epoch and everlasting scenarios
"""

import os
import pickle
import numpy as np
import random
import copy
import tqdm
from scipy.stats import truncnorm


def truncated_normal(mean=0.25, sd=0.1, low=0.0, upp=0.5):
    """
    Generate a truncated normal distribution with specified parameters.
    
    This function creates a truncated normal distribution, which is useful for
    generating bounded random values (e.g., probabilities) that follow a normal
    distribution within specified limits.
    
    Args:
        mean (float): Mean of the underlying normal distribution (default: 0.25)
        sd (float): Standard deviation of the underlying normal distribution (default: 0.1)
        low (float): Lower bound for truncation (default: 0.0)
        upp (float): Upper bound for truncation (default: 0.5)
    
    Returns:
        scipy.stats.truncnorm: A truncated normal distribution object
    """
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)



# Define a class for a Bernoulli arm (coin) in multi-armed bandit problems
class Bernoulli_Arm:
    '''
    A Bernoulli arm (bandit) that generates binary rewards (0 or 1) with a fixed probability.
    
    This class represents a single arm in a multi-armed bandit problem where each arm
    follows a Bernoulli distribution. The arm can be pulled to get a random reward
    based on its underlying probability parameter.
    
    Attributes:
        p (float): The probability of receiving a reward of 1 (success probability)
        t (int): The current time step for this arm
    
    Methods:
        pull(): Execute a single arm pull and return the reward
        batch_pull(num_pull): Execute multiple pulls and return the average reward
        time(): Return the current time step
    '''

    def __init__(self, p=None, random_mean = 'uniform', t = 0):
        '''
        Initialize a Bernoulli arm with specified or randomly generated probability.
        
        Args:
            p (float, optional): The success probability for the arm. If None, 
                                will be randomly generated based on random_mean parameter
            random_mean (str): Distribution type for random probability generation.
                              'uniform' - uniform distribution [0,1]
                              'gaussian' - truncated normal distribution
            t (int): Initial time step for this arm (default: 0)
        
        Raises:
            ValueError: If random_mean is not 'uniform' or 'gaussian'
        '''
        # Set the success probability either from parameter or random generation
        if p is not None:
            self.p = p
        elif random_mean=='uniform':
            # Generate probability from uniform distribution [0,1]
            self.p = np.random.uniform()
        elif random_mean=='gaussian':
            # Generate probability from truncated normal distribution [0,0.5]
            truc_gaussian_sampler = truncated_normal()
            self.p = truc_gaussian_sampler.rvs()
        else:
            raise ValueError('Distribution of the mean not recognized!')
        
        # Initialize time step
        self.t = t

    def pull(self):
        '''
        Execute a single arm pull using rejection sampling.
        
        Generates a random sample and compares it to the arm's success probability
        to determine the reward (1 for success, 0 for failure).
        
        Returns:
            float: 1.0 if successful (reward), 0.0 if failed (no reward)
        '''
        # Generate a random sample from uniform distribution [0,1]
        this_sample = np.random.uniform()
        
        # Return reward based on comparison with success probability
        if this_sample <= self.p:
            return 1.0  # Success - give reward
        else:
            return 0.0  # Failure - no reward
        
    def batch_pull(self, num_pull):
        '''
        Execute multiple arm pulls and return the average reward.
        
        This method efficiently handles large numbers of pulls using either
        vectorized operations (for smaller numbers) or Gaussian approximation
        (for very large numbers) to improve performance.
        
        Args:
            num_pull (int): Number of pulls to execute
            
        Returns:
            float: Average reward over all pulls (between 0 and 1)
        '''
        
        if num_pull <= 1e5:
            # For smaller numbers of pulls, use vectorized operations for speed
            vec_sample = np.random.uniform(size=[(int)(num_pull)])
            
            # Vectorized implementation of rejection sampling:
            # (p - vec_sample) >= 0 means success, < 0 means failure
            # np.ceil() converts positive values to 1, non-positive to 0
            # Add small epsilon (1e-20) to avoid numerical issues
            # Return the average of all rewards
            return np.average(np.ceil(self.p - vec_sample + 1e-20))
        else:
            # For very large numbers, approximate using Central Limit Theorem
            # Add Gaussian noise with variance proportional to 1/sqrt(n)
            mean = self.p + np.random.normal(0, 2/np.sqrt(num_pull))
            # Ensure the result doesn't exceed 1 (maximum possible reward)
            return min(1, mean)



# Arm Reading Buffer: Simulates a 'bank' that feeds arms to the algorithm
# This class manages the streaming of arms and maintains local memory constraints
class Arm_Reading_Buffer:
    '''
    A buffer class that manages sequential arm reading for sliding window multi-armed bandit algorithms.
    
    This class simulates a 'bank' or stream that feeds arms to the bandit algorithm one at a time,
    maintaining the sliding window constraint and computing regret metrics for evaluation.
    
    The buffer handles:
    - Sequential arm delivery to simulate streaming scenarios
    - Sliding window best reward computation
    - Regret tracking for both epoch-based and everlasting scenarios
    - Stream reset functionality for repeated experiments
    
    Attributes:
        arm_set (np.ndarray): Array of Bernoulli arms in the stream
        n (int): Total number of arms in the set
        window_size (int): Size of the sliding window for local memory constraint
        t (int): Current time step (position in the stream)
        rewards (np.ndarray): Array of true reward probabilities for each arm
        best_rewards (np.ndarray): Best possible reward at each time step given window constraint
        everlasting_reward (float): Global best reward across all arms
        regret_counter (float): Accumulated regret over time
    
    Methods:
        read_next_arm(): Get the next arm in the sequence
        reset(): Reset the buffer to initial state
        regret_update_epoch(pull_award, time_now, num_pulls): Update regret using epoch-based comparison
        regret_update_everlasting(pull_award, num_pulls): Update regret using global best comparison
    '''
    
    def __init__(self, arm_set, window_size):
        '''
        Initialize the arm reading buffer with a set of arms and window size.
        
        Args:
            arm_set: Collection of Bernoulli_Arm objects (will be converted to numpy array)
            window_size (int): Size of the sliding window for local memory constraint
            
        Raises:
            ValueError: If arm_set cannot be converted to numpy array
        '''
        
        # Validate and store the arm set as a flattened numpy array
        if type(arm_set) is not np.ndarray:
            raise ValueError('Expecting numpy array of arms. Please convert')
        # Flatten the array to ensure 1D structure
        self.arm_set = np.reshape(arm_set, [-1])

        # Store the total number of arms
        self.n = np.shape(self.arm_set)[0]
        
        # Assign time steps to each arm (arm index = time step when it appears)
        for i in range(self.n):
            self.arm_set[i].t = i

        # Store the sliding window size
        self.window_size = window_size
        
        # Initialize current time step (starts at -1, increments to 0 on first read)
        self.t = -1

        # Extract and store the true reward probabilities from all arms
        list_rewards = []
        for arm in self.arm_set:
            list_rewards.append(arm.p)
        self.rewards = np.array(list_rewards)

        # Pre-compute the best possible rewards at each time step given window constraint
        list_best_rewards = []
        
        # For the initial window_size time steps, best reward is max among seen arms so far
        for time_step in range(self.window_size):
            # At time_step t, we've seen arms 0 through t (inclusive)
            list_best_rewards.append(np.max(self.rewards[:time_step+1]))
            
        # For subsequent time steps, best reward is max within the sliding window
        for time_step in range(self.window_size, self.n):
            # Window contains arms from (t-window_size+1) to t (inclusive)
            list_best_rewards.append(np.max(self.rewards[time_step-self.window_size+1:time_step+1]))
            
        # Convert to numpy array for efficient access
        self.best_rewards = np.array(list_best_rewards)

        # Store the global best reward (for everlasting regret computation)
        self.everlasting_reward = np.max(self.rewards)

        # Initialize regret counter
        self.regret_counter = 0

        # Internal state management
        self._terminate_flag = False  # Flag to indicate when stream is exhausted        

    def read_next_arm(self):
        '''
        Read the next arm from the stream and advance the time step.
        
        This method simulates the sequential arrival of arms in a streaming bandit
        scenario. Each call returns the next arm in the sequence and increments
        the internal time counter.
        
        Returns:
            Bernoulli_Arm or None: The next arm in the sequence, or None if the 
                                  stream has been exhausted
        '''
        if not self._terminate_flag:
            # Advance to next time step
            self.t = self.t + 1
            # Retrieve the arm at current time step
            return_arm = self.arm_set[self.t]
            
            # Check if we've reached the end of the stream
            if (self.t == self.n):
                self._terminate_flag = True
            
            return return_arm
        else:
            # Stream has been exhausted - no more arms available
            return None
    
    def reset(self):
        '''
        Reset the buffer to its initial state for reuse in multiple experiments.
        
        This method allows the same arm sequence to be used multiple times
        by resetting the internal state variables to their initial values.
        Useful for running repeated experiments or testing different algorithms
        on the same arm sequence.
        '''
        self.t = -1                    # Reset time step to before first arm
        self._terminate_flag = False   # Reset termination flag
        self.regret_counter = 0        # Reset accumulated regret
        
    def regret_update_epoch(self, pull_award, time_now, num_pulls):
        '''
        Update the regret counter using epoch-based (sliding window) comparison.
        
        In epoch-based regret, we compare the algorithm's performance against the
        best arm that could have been chosen given the sliding window constraint
        at the current time step. This reflects the realistic scenario where
        the algorithm can only consider arms within its memory window.
        
        Args:
            pull_award (float): The actual reward received from the arm pull
            time_now (int): Current time step in the stream
            num_pulls (int): Number of pulls executed (for weighted regret)
        '''
        # Get the best possible reward at current time given window constraint
        best_reward_now = self.best_rewards[time_now]
        # Add the difference (regret) weighted by number of pulls
        self.regret_counter += (best_reward_now - pull_award) * num_pulls

    def regret_update_everlasting(self, pull_award, num_pulls):
        '''
        Update the regret counter using everlasting (global best) comparison.
        
        In everlasting regret, we compare the algorithm's performance against the
        globally best arm in the entire stream, regardless of when it appeared.
        This provides a more stringent evaluation metric that doesn't account
        for the sliding window constraint, representing the theoretical minimum
        regret if the algorithm had perfect global knowledge.
        
        Args:
            pull_award (float): The actual reward received from the arm pull
            num_pulls (int): Number of pulls executed (for weighted regret)
        '''
        # Add regret compared to the globally best arm
        self.regret_counter += (self.everlasting_reward - pull_award) * num_pulls





