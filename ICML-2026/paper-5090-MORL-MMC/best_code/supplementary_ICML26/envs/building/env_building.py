"""
The module implements the BuildingEnvReal class.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
from numpy import linalg as LA
import random
import torch.optim as optim
import gymnasium as gym
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import inv
import pandas as pd

from typing import Optional, Any, Dict

from sklearn import linear_model
        
class BuildingEnv_9d(gym.Env):
    """BuildingEnvReal class.
    This classes simulates the zonal temperature of a building controlled by a user selected agent.
    It constructs the physics based building simulation model based on the RC model with an
    nonlinear residual model. The simulation is based on the EPW weather file provided by the Building
    Energy Codes Program.

    n = number of zones in the building
    k = number of steps for the MOER CO2 forecast
    Actions:
        Type: Box(n)
        Action                                           Shape       Min         Max
        HVAC power consumption(cool in - ,heat in +)     n           -1          1
    Observations:
        Type: Box(3n+2)
                                                         Shape       Min         Max
        Temperature of zones (celsius)                   n           temp_min    temp_max
        Temperature of outdoor (celsius)                 1           temp_min    temp_max
        Global Horizontal Irradiance (W)                 n           0           heat_max
        Temperature of ground (celsius)                  1           temp_min    temp_max
        Occupancy power (W)                              n           0           heat_max
    Attributes:
        Parameter (dict): Dictionary containing the parameters for the environment.
        observation_space: structure of observations returned by environment
        timestep: current timestep in episode, from 0 to 288
        action_space: structure of actions expected by environment
    """
    metadata = {'render.modes': []}

    #Occupancy nonlinear coefficients
    OCCU_COEF1=6.461927
    OCCU_COEF2=0.946892
    OCCU_COEF3=0.0000255737
    OCCU_COEF4=0.0627909
    OCCU_COEF5=0.0000589172
    OCCU_COEF6=0.19855
    OCCU_COEF7=0.000940018
    OCCU_COEF8=0.00000149532

    #Occupancy linear coefficient
    OCCU_COEF9=7.139322

    #Discrete space length
    DISCRETE_LENGTH=100

    #Scaling factor for the reward function weight
    SCALING_FACTOR=24

    def __init__(self, Parameter,user_reward_function=None,reward_breakdown_keys=None):
        """Initializes the environment with the given parameters.
        Args:
            Parameter (dict): Dictionary containing the parameters for the environment.
                'OutTemp' (np.array): Outdoor temperature.
                'connectmap' (np.array): Connection map of the rooms.
                'RCtable' (np.array): RC table of the rooms.
                'roomnum' (int): Number of rooms.
                'weightcmap' (np.array): Weight of the connection map.
                'target' (np.array): Target temperature.
                'gamma' (list): Weight factor for the reward function.
                'ghi' (np.array): Global Horizontal Irradiance.
                'GroundTemp' (np.array): Ground temperature.
                'Occupancy' (np.array): Occupancy of the rooms.
                'ACmap' (np.array): Air conditioning map.
                'max_power' (int): Maximum power for the air conditioning system.
                'nonlinear' (np.array): Nonlinear factor.
                'temp_range' (list): Temperature range (min and max).
                'spacetype' (str): Type of space ('continuous' or 'discrete').
                'time_resolution' (int): Time resolution of the simulation.
        Initializes:
            action_space: Action space for the environment (gym.spaces.Box).
            observation_space: Observation space for the environment (gym.spaces.Box).
            A_d: Discrete-time system matrix A (numpy array).
            B_d: Discrete-time system matrix B (numpy array).
            rewardsum: Cumulative reward in the environment (float).
            statelist: List of states in the environment (list).
            actionlist: List of actions taken in the environment (list).
            epochs: Counter for the number of epochs (int).
        """
        self.OutTemp = Parameter['OutTemp']
        self.length_of_weather=len(self.OutTemp)
        self.connectmap = Parameter['connectmap']
        self.RCtable = Parameter['RCtable']
        self.roomnum = Parameter['roomnum']
        self.weightCmap = Parameter['weightcmap']
        self.target = Parameter['target']
        self.gamma=Parameter['gamma']
        self.ghi=Parameter['ghi']
        self.GroundTemp=Parameter['GroundTemp']
        self.Occupancy=Parameter['Occupancy']
        self.acmap=Parameter['ACmap']
        self.maxpower=Parameter['max_power']
        self.nonlinear=Parameter['nonlinear']
        self.temp_range=Parameter['temp_range']
        self.spacetype=Parameter['spacetype']
        self.Occupower=0
        self.timestep = Parameter['time_resolution']
        self.CarbonIntensity = Parameter['CarbonIntensity']
        self.Carbon_factor = 1/max(self.CarbonIntensity)
        self.ElectricityPrice = Parameter['ElectricityPrice']
        self.Price_factor = 1/max(self.ElectricityPrice)
        self.datadriven=False

        # Define action space bounds based on room number and air conditioning map
        self.Qlow = np.ones(self.roomnum, dtype=np.float32) * (-1.0)*self.acmap.astype(np.float32)+1e-12
        self.Qhigh = np.ones(self.roomnum, dtype=np.float32) * (1.0)*self.acmap.astype(np.float32)

        # Set the action space based on the space type
        if self.spacetype == 'continuous':
          self.action_space = gym.spaces.Box(self.Qlow, self.Qhigh, dtype=np.float32)
        else:
          self.action_space = gym.spaces.MultiDiscrete((self.Qhigh*self.DISCRETE_LENGTH-self.Qlow*self.DISCRETE_LENGTH))

        # Set the observation space bounds based on the minimum and maximum temperature
        self.min_T = self.temp_range[0]
        self.max_T = self.temp_range[1]
        self.low = np.array(self.roomnum*[0]+[-40, 0, -40, -1, 0, 0])
        self.high = np.array(self.roomnum*[40]+[40, 1, 40, 1, 1, 1])
        self.observation_space = gym.spaces.Box(np.zeros(self.roomnum+6), np.ones(self.roomnum+6), dtype=np.float32)

        # Set the weight for the power consumption and comfort range
        self.q_rate = Parameter['gamma'][0]*self.SCALING_FACTOR
        self.error_rate = Parameter['gamma'][1]

        # If a user-defined reward function is provided, use it.
        # Otherwise, use the default reward function.
        if user_reward_function is not None and callable(user_reward_function):
            self.reward_function = functools.partial(user_reward_function, self)
        else:
            self.reward_function = self.default_reward_function


        # If user-defined reward breakdown keys are provided, use them.
        # Otherwise, use the default keys.
        if reward_breakdown_keys is not None and isinstance(reward_breakdown_keys, list):
            self._reward_breakdown = {key: 0.0 for key in reward_breakdown_keys}
        else:
            self._reward_breakdown = {'comfort_level': 0.0, 'power_consumption': 0.0}

        # Define matrices for the state update equations
        connect_indoor = self.connectmap[:, :-1]
        Amatrix = self.RCtable[:, :-1]
        diagvalue = (-self.RCtable) @ self.connectmap.T - np.array([self.weightCmap.T[1]]).T
        np.fill_diagonal(Amatrix, np.diag(diagvalue))
        Amatrix+=self.nonlinear*self.OCCU_COEF9/self.roomnum
        Bmatrix = self.weightCmap.T
        Bmatrix[2] = self.connectmap[:, -1] * (self.RCtable[:, -1])
        Bmatrix = (Bmatrix.T)

        # Initialize reward sum, state list, action list, and epoch counter
        self.rewardsum = 0
        self.statelist = []
        self.actionlist=[]
        self._elapsed_steps = 0

        # Initialize zonal temperature
        self.X_new= self.target

        # Compute the discrete-time system matrices
        self.A_d = expm(Amatrix * self.timestep)
        self.B_d = inv(Amatrix) @ (self.A_d - np.eye(self.A_d.shape[0])) @ Bmatrix
        
        self._max_episode_steps = 480
        
        self.last_action = np.zeros(self.roomnum)

    def default_reward_function(self, state, action, error, state_new, step_idx):
        botton_action = action[0:9]
        middle_action = action[9:16]
        top_action = action[16:]
        
        botton_error = error[0:9]
        middle_error = error[9:16]
        top_error = error[16:]

        reward_comfort_botton = (20*9-LA.norm(botton_error, 1))/20
        reward_comfort_middle = (20*7-LA.norm(middle_error, 1))/20
        reward_comfort_top = (20*7-LA.norm(top_error, 1))/20
        
        reward_energy_botton = (20*9-LA.norm(botton_action, 1))/20
        reward_energy_middle = (20*7-LA.norm(middle_action, 1))/20
        reward_energy_top = (20*7-LA.norm(top_action, 1))/20
        
        if len(self.actionlist)>=2:
            reward_ramping_botton = 9-abs(LA.norm(self.actionlist[-1][0:9], 1)-LA.norm(self.actionlist[-2][0:9], 1))
            reward_ramping_middle = 7-abs(LA.norm(self.actionlist[-1][9:16], 1)-LA.norm(self.actionlist[-2][9:16], 1))
            reward_ramping_top = 7-abs(LA.norm(self.actionlist[-1][16:], 1)-LA.norm(self.actionlist[-2][16:], 1))
        else:
            reward_ramping_botton = 0
            reward_ramping_middle = 0
            reward_ramping_top = 0

        self._reward_breakdown['comfort_level'] -= LA.norm(error, 2) * self.error_rate
        self._reward_breakdown['power_consumption'] -= LA.norm(action, 2) * self.q_rate
        return reward_comfort_botton, reward_comfort_middle, reward_comfort_top, reward_energy_botton, reward_energy_middle, reward_energy_top, reward_ramping_botton, reward_ramping_middle, reward_ramping_top

    def modified_reward_function(self, action, error):
        # comfort -> vector reward (3dim)
        botton_action, middle_action, top_action = action[0:9], action[9:16], action[16:]
        botton_error, middle_error, top_error = error[0:9], error[9:16], error[16:]

        reward_comfort_botton = (180 - LA.norm(botton_error, 1)) / 20
        reward_comfort_middle = (140 - LA.norm(middle_error, 1)) / 20
        reward_comfort_top = (140 - LA.norm(top_error, 1)) / 20

        reward_vector = np.array([
            reward_comfort_botton, reward_comfort_middle, reward_comfort_top
        ], dtype=np.float32)

        cost_energy_botton = LA.norm(botton_action, 1)
        cost_energy_middle = LA.norm(middle_action, 1)
        cost_energy_top = LA.norm(top_action, 1)
        total_energy_cost = cost_energy_botton + cost_energy_middle + cost_energy_top

        if len(self.actionlist) >= 2:
            cost_ramping_botton = abs(LA.norm(self.actionlist[-1][0:9], 1) - LA.norm(self.actionlist[-2][0:9], 1))
            cost_ramping_middle = abs(LA.norm(self.actionlist[-1][9:16], 1) - LA.norm(self.actionlist[-2][9:16], 1))
            cost_ramping_top = abs(LA.norm(self.actionlist[-1][16:], 1) - LA.norm(self.actionlist[-2][16:], 1))
        else:
            cost_ramping_botton = cost_ramping_middle = cost_ramping_top = 0

        total_ramping_cost = cost_ramping_botton + cost_ramping_middle + cost_ramping_top

        scale = 5.0 
        energy = - total_energy_cost / scale  

        constraints_info = {
            "energy": energy,
        }

        return reward_vector, constraints_info

    def step(self, action):
        """Steps the environment.
        Updates the state of the environment based on the given action and calculates the
        reward, done, and info values for the current timestep.
        Args:
            action (np.ndarray): Action to be taken in the environment.
            return_info: info is always returned, but if return_info is set
                to False, only 'zone_temperature' and 'reward_breakdown' are
                returned.
        Returns:
            state (np.ndarray): Updated state of the environment.
                'X_new': shape [roomnum], new temperatures of the rooms.
                'OutTemp': shape [1], outdoor temperature at the current timestep.
                'ghi': shape [1], global horizontal irradiance at the current timestep.
                'GroundTemp': shape [1], ground temperature at the current timestep.
                'Occupower': shape [1], occupancy power at the current timestep.
            reward (float): Reward for the current timestep.
            done (bool): Whether the episode is terminated.
            truncated (bool): Whether the episode has reached a time limit.
            info (dict[str, Any]): Dictionary containing auxiliary information.
                'statelist': List of states in the environment.
                'actionlist': List of actions taken in the environment.
                'epochs': Counter for the number of epochs (int).
        """
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()     

        step_idx = self._elapsed_steps+self.random
        
        # Scale the action if the space type is not continuous
        if self.spacetype != 'continuous':
          action=(action+self.Qlow*self.DISCRETE_LENGTH)/self.DISCRETE_LENGTH

        # Store the current state in the statelist
        self.statelist.append(self.state)

        # Initialize the 'done' flag as False
        done = False

        # Prepare the input matrices X and Y
        X = self.state[:self.roomnum].T
        Y = np.insert(np.append(action,self.ghi[step_idx]), 0, self.OutTemp[step_idx]).T
        Y = np.insert(Y, 0, self.GroundTemp[step_idx]).T
        avg_temp=np.sum(self.state[:self.roomnum])/self.roomnum
        Meta = self.Occupancy[step_idx]

        # If the environment is data-driven, add additional features to the Y matrix
        if self.datadriven==True:
           Y = np.insert(Y, 0, Meta).T
           Y = np.insert(Y, 0, Meta**2).T
           Y = np.insert(Y, 0, avg_temp).T
           Y = np.insert(Y, 0, avg_temp**2).T
        else:
           # Calculate Occupower based on the given formula
           self.Occupower=self.OCCU_COEF1+self.OCCU_COEF2*Meta+self.OCCU_COEF3*Meta**2 - self.OCCU_COEF4*avg_temp*Meta+self.OCCU_COEF5*avg_temp*Meta**2 - self.OCCU_COEF6*avg_temp**2+self.OCCU_COEF7*avg_temp**2*Meta - self.OCCU_COEF8*avg_temp**2*Meta**2

           # Insert Occupower at the beginning of the Y matrix
           Y = np.insert(Y, 0, self.Occupower).T

        # Update the state using the A_d and B_d matrices
        X_new = self.A_d @ X + self.B_d @ Y

        # Calculate the error
        error = X_new*self.acmap - self.target*self.acmap

        # Get reward and constraints 
        reward_vector, constraints_info = self.modified_reward_function(action, error)

        # retrieve environment info
        self.X_new = X_new
        info = self._get_info()

        # Update the state
        ghi_repeated = np.array([self.ghi[step_idx]])
        occ_repeated= np.array([self.Occupower/1000])


        self.state = np.concatenate((X_new, self.OutTemp[step_idx].reshape(-1,), ghi_repeated, self.GroundTemp[step_idx].reshape(-1), occ_repeated, self.CarbonIntensity_Gauss[step_idx].reshape(-1), self.ElectricityPrice_Gauss[step_idx].reshape(-1)), axis=0)
        self.state = np.maximum(self.state, self.low)
        self.state = np.minimum(self.state, self.high)
        state = (self.state-self.low)/(self.high-self.low)

        # Store the action in the actionlist
        self.actionlist.append(action)

        # Increment the epochs counter
        self._elapsed_steps += 1


        # Check termination conditions
        terminated = False
        truncated = self._elapsed_steps >= self._max_episode_steps

        if truncated:
            self._elapsed_steps = 0
            
        self.last_action = action

        info = {
            "constraint": constraints_info["energy"],  
        }

        # Return the new state, reward, done flag, and info
        return state, reward_vector, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        """Resets the environment.
        Prepares the environment for the next episode by setting the initial
        temperatures, average temperature, occupancy, and occupower. The initial state
        is constructed by concatenating these variables.

        Args:
            seed: seed for resetting the environment. An episode is entirely
                reproducible no matter the generator used.
            options: resetting options
                'verbose': set verbosity level [0-2]
        Returns:
            state: the initial state of the environment.
            info: information.
        """
      
        # Initialize the episode counter
        self._elapsed_steps = 0
        self.random = self.np_random.integers(1, 201)
        self.CarbonIntensity_Gauss = np.array([c + self.np_random.normal(0, 0.02) for c in self.CarbonIntensity])
        self.ElectricityPrice_Gauss = np.array([c + self.np_random.normal(0, 0.02) for c in self.ElectricityPrice])    
        
        # Initialize state and action lists
        self.statelist = []
        self.actionlist=[]

        # Use options to get T_initial or use the default value if not provided
        T_initial = self.target if options is None else options.get('T_initial', self.target)


        # Calculate the average initial temperature
        avg_temp=np.sum(T_initial)/self.roomnum

        # Get the occupancy value for the current epoch
        Meta = self.Occupancy[self.random]

        # Calculate the occupower based on occupancy and average temperature
        self.Occupower=self.OCCU_COEF1+self.OCCU_COEF2*Meta+self.OCCU_COEF3*Meta**2 - self.OCCU_COEF4*avg_temp*Meta+self.OCCU_COEF5*avg_temp*Meta**2 - self.OCCU_COEF6*avg_temp**2+self.OCCU_COEF7*avg_temp**2*Meta - self.OCCU_COEF8*avg_temp**2*Meta**2

        # Construct the initial state by concatenating relevant variables
        self.X_new=T_initial

        ghi_repeated = np.array([self.ghi[self.random]])
        occ_repeated= np.array([self.Occupower/1000])
        self.state = np.concatenate((T_initial,self.OutTemp[self.random].reshape(-1,), ghi_repeated, self.GroundTemp[self.random].reshape(-1), occ_repeated, self.CarbonIntensity_Gauss[self.random].reshape(-1), self.ElectricityPrice_Gauss[self.random].reshape(-1)), axis=0)
        self.state = np.maximum(self.state, self.low)
        self.state = np.minimum(self.state, self.high)
        state = (self.state-self.low)/(self.high-self.low)
        state = state.astype(np.float32)
        # Initialize the rewards
        self.flag=1
        self.rewardsum = 0
        for re in self._reward_breakdown:
          self._reward_breakdown[re] = 0.0

        info: dict[str, Any] = {} 
        # Return the initial state and an empty dictionary(for gymnasium grammar)
        return state.astype(np.float32), info   
    
    def _get_info(self, all: bool = False) -> Dict[str, Any]:
        """
        Returns info. See step().

        Args:
            all: whether all information should be returned. Otherwise, only
                'zone_temperature' and 'reward_breakdown' are returned.
        """
        if all:
            return {
                'zone_temperature': self.X_new,
                'out_temperature': self.OutTemp[self._elapsed_steps+self.random].reshape(-1,),
                'ghi': self.ghi[self._elapsed_steps+self.random].reshape(-1),
                'ground_temperature': self.GroundTemp[self._elapsed_steps+self.random].reshape(-1),
                'reward_breakdown': self._reward_breakdown,
            }
        else:
            return {
                'zone_temperature': self.X_new,
                'reward_breakdown': self._reward_breakdown,
            }
        
    def train(self, states: np.ndarray, actions: np.ndarray) -> None:
        """Trains the linear regression model using the given states and actions.
        The model is trained to predict the next state based on the current state and action.
        The trained coefficients are stored in the environment for later use.

        Args:
            states: a list of states.
            actions: a list of actions corresponding to each state.
        """
        # Initialize lists to store current states and next states
        current_state=[]
        next_state=[]

        # Iterate through states and actions to create input and output data for the model
        for i in range(len(states)-1):
          X=states[i]
          Y = np.insert(np.append(actions[i]/self.maxpower,self.ghi[i]), 0, self.OutTemp[i]).T
          Y = np.insert(Y, 0, self.GroundTemp[i]).T
          avg_temp=np.sum(X)/self.roomnum
          Meta = self.Occupancy[i]

          # Calculate the occupower based on occupancy and average temperature
          self.Occupower=self.OCCU_COEF1+self.OCCU_COEF2*Meta+self.OCCU_COEF3*Meta**2 - self.OCCU_COEF4*avg_temp*Meta+self.OCCU_COEF5*avg_temp*Meta**2 - self.OCCU_COEF6*avg_temp**2+self.OCCU_COEF7*avg_temp**2*Meta - self.OCCU_COEF8*avg_temp**2*Meta**2

          # Add relevant variables to Y
          Y = np.insert(Y, 0, Meta).T
          Y = np.insert(Y, 0, Meta**2).T
          Y = np.insert(Y, 0, avg_temp).T
          Y = np.insert(Y, 0, avg_temp**2).T

          # Concatenate X and Y to form the input data for the model
          stackxy=np.concatenate((X,Y), axis=0)

          # Append the input data and next state to their respective lists
          current_state.append(stackxy)
          next_state.append(states[i+1])

        # Create a linear regression model with non-negative coefficients and no intercept
        model = linear_model.LinearRegression(fit_intercept=False,positive=True)

        # Fit the model using the input and output data
        modelfit = model.fit(np.array(current_state),np.array(next_state))

        # Get the coefficients of the fitted model
        beta = modelfit.coef_

        # Update the A_d and B_d matrices with the coefficients from the fitted model
        self.A_d=beta[:,:self.roomnum]
        self.B_d=beta[:,self.roomnum:]

        # Set the data-driven flag to True
        self.datadriven=True


    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]    
    