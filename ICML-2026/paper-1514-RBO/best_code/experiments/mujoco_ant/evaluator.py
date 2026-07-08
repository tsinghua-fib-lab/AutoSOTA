"""Mujoco Ant locomotion evaluator using the BO framework."""

import numpy as np
import torch
import gymnasium as gym
from typing import Dict, Any, Union, Optional
import warnings

from bo_framework.base.evaluator import BaseEvaluator
from bo_framework.base.evaluation_result import EvaluationResult


class LinearContinuousPolicy:
    """A linear policy for continuous control actions.
    
    Maps observations to continuous actions via: actions = tanh(obs @ W + b)
    The tanh ensures actions stay within [-1, 1] bounds.
    """
    
    def __init__(self, obs_dim: int, action_dim: int):
        """Initialize the linear continuous policy.
        
        Args:
            obs_dim: Dimension of observation space
            action_dim: Dimension of continuous action space
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_params = obs_dim * action_dim + action_dim  # weights + biases
        
        # Initialize parameters
        self.weights = np.zeros((obs_dim, action_dim))
        self.biases = np.zeros(action_dim)
    
    def set_params(self, params: np.ndarray) -> None:
        """Set policy parameters from flat array.
        
        Args:
            params: Flat array [W_00, W_01, ..., W_mn, b_0, ..., b_n]
        """
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        
        # Reshape weights and biases
        n_weights = self.obs_dim * self.action_dim
        self.weights = params[:n_weights].reshape(self.obs_dim, self.action_dim)
        self.biases = params[n_weights:]
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get continuous action from observation.
        
        Args:
            obs: Observation array of shape (obs_dim,)
            
        Returns:
            Continuous action array of shape (action_dim,) in [-1, 1]
        """
        # Compute linear transformation: obs @ W + b
        logits = np.dot(obs, self.weights) + self.biases
        # Apply tanh to keep actions in [-1, 1]
        actions = np.tanh(logits)
        return actions


class AntEvaluator(BaseEvaluator):
    """Evaluator for Mujoco Ant locomotion optimization.
    
    Evaluates linear continuous control policies in the Ant environment
    and returns cumulative episode rewards (forward locomotion).
    """
    
    def __init__(self, env_name: str = "Ant-v5", 
                 max_steps: int = 1000, render: bool = False,
                 seed: Optional[int] = None, penalty_value: float = -2000.0,
                 param_bounds: tuple = (-1.0, 1.0)):
        """Initialize Mujoco Ant evaluator.
        
        Args:
            env_name: Gymnasium environment name (Ant-v4, Ant-v5, etc.)
            max_steps: Maximum steps per episode  
            render: Whether to render episodes (disabled for lightweight simulation)
            seed: Random seed for environment
            penalty_value: Very bad score for simulation failures
            param_bounds: Parameter bounds for policy weights/biases
        """
        self.env_name = env_name
        self.max_steps = max_steps
        self.render = render
        self.seed = seed
        self.penalty_value = penalty_value
        self.param_bounds = param_bounds
        
        # Create environment (no rendering for lightweight simulation)
        render_mode = None  # Always headless for fast evaluation
        try:
            self.env = gym.make(env_name, render_mode=render_mode)
        except Exception as e:
            # Fallback to Ant-v5 if specific version not available
            warnings.warn(f"Could not create {env_name}, trying Ant-v5: {e}")
            self.env = gym.make("Ant-v5", render_mode=render_mode)
        
        # Set seed if provided
        if seed is not None:
            self.env.reset(seed=seed)
        
        # Get environment dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Verify action space is continuous and bounded
        assert hasattr(self.env.action_space, 'low'), "Action space must be continuous"
        assert hasattr(self.env.action_space, 'high'), "Action space must be continuous"
        
        # Create policy instance
        self.policy = LinearContinuousPolicy(self.obs_dim, self.action_dim)
        
        print(f"Initialized Mujoco Ant evaluator: {self.obs_dim}D obs → {self.action_dim}D actions")
        print(f"Policy parameters: {self.policy.n_params} ({self.obs_dim}×{self.action_dim} weights + {self.action_dim} biases)")
        print(f"Action bounds: [{self.env.action_space.low[0]:.1f}, {self.env.action_space.high[0]:.1f}]")
    
    def evaluate(self, params: Union[Dict[str, Any], torch.Tensor]) -> EvaluationResult:
        """Evaluate linear policy parameters in Mujoco Ant environment.
        
        Args:
            params: Either parameter dict with indexed keys or tensor of policy parameters
            
        Returns:
            EvaluationResult with cumulative episode reward (or penalty for failures)
        """
        try:
            # Handle different parameter formats
            if isinstance(params, dict):
                # SearchSpace.from_bounds() creates params like {'x0': val, 'x1': val, ...}
                param_values = []
                for i in range(self.policy.n_params):
                    key = f"x{i}"
                    if key not in params:
                        raise ValueError(f"Missing parameter '{key}' in params dict")
                    # Denormalize from [0,1] to original bounds
                    normalized_val = params[key]
                    actual_val = normalized_val * (self.param_bounds[1] - self.param_bounds[0]) + self.param_bounds[0]
                    param_values.append(actual_val)
                param_array = np.array(param_values)
            elif isinstance(params, torch.Tensor):
                param_array = params.detach().cpu().numpy()
                # If tensor is normalized, denormalize it
                if torch.all(params >= 0) and torch.all(params <= 1):
                    param_array = param_array * (self.param_bounds[1] - self.param_bounds[0]) + self.param_bounds[0]
            else:
                param_array = np.array(params)
            
            # Validate parameter count
            if len(param_array) != self.policy.n_params:
                raise ValueError(f"Expected {self.policy.n_params} parameters, got {len(param_array)}")
            
            # Set policy parameters
            self.policy.set_params(param_array)
            
            # Run episode with error handling for simulation failures
            obs, info = self.env.reset()
            total_reward = 0.0
            step_count = 0
            
            for step in range(self.max_steps):
                # Get continuous action from policy
                action = self.policy.get_action(obs)
                
                # Ensure action is within valid bounds (should be due to tanh, but double-check)
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                
                # Take step in environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                step_count += 1
                
                # Check for simulation instabilities (NaN values, extreme states)
                if np.isnan(obs).any() or np.isinf(obs).any():
                    print(f"Simulation instability detected at step {step}: NaN/Inf in observations")
                    return EvaluationResult.from_true_value(params, self.penalty_value)
                
                if np.isnan(reward) or np.isinf(reward):
                    print(f"Simulation instability detected at step {step}: NaN/Inf reward")
                    return EvaluationResult.from_true_value(params, self.penalty_value)
                
                # Episode finished
                if terminated or truncated:
                    break
            
            # Additional check: if total reward is suspiciously bad, likely a simulation failure
            if total_reward < -1000:
                print(f"Suspiciously low reward ({total_reward:.1f}), treating as simulation failure")
                return EvaluationResult.from_true_value(params, self.penalty_value)
            
            # Create evaluation result
            return EvaluationResult.from_true_value(params, total_reward)
            
        except Exception as e:
            print(f"Error evaluating Ant policy (assigning penalty): {e}")
            return EvaluationResult.from_true_value(params, self.penalty_value)
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'env'):
            self.env.close()
    
    def __del__(self):
        """Cleanup when evaluator is deleted."""
        self.close()
    
    @property
    def is_deterministic(self) -> bool:
        """Whether the objective is deterministic."""
        return False  # Stochastic due to random initial conditions and physics simulation