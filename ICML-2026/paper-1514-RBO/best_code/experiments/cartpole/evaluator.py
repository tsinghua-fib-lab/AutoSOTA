"""CartPole RL policy evaluator using the new API."""

import numpy as np
import torch
import gymnasium as gym
from typing import Dict, Any, Union, Optional
import warnings

from bo_framework.base.evaluator import BaseEvaluator
from bo_framework.base.evaluation_result import EvaluationResult


class LinearPolicy:
    """A simple linear policy that maps observations to discrete actions.

    This policy represents a linear controller with weights W and biases b,
    computing action logits as: logits = obs @ W + b
    """

    def __init__(self, obs_dim: int = 4, action_dim: int = 2):
        """Initialize the linear policy.

        Args:
            obs_dim: Dimension of observation space (4 for CartPole)
            action_dim: Number of discrete actions (2 for CartPole)
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

    def get_action(self, obs: np.ndarray) -> int:
        """Get action from observation.

        Args:
            obs: Observation array of shape (obs_dim,)

        Returns:
            Integer action (0 to action_dim-1)
        """
        # Compute logits: obs @ W + b
        logits = np.dot(obs, self.weights) + self.biases
        return int(np.argmax(logits))


class CartPoleEvaluator(BaseEvaluator):
    """Evaluator for CartPole RL policy optimization.

    Evaluates linear policies in the CartPole environment and returns
    cumulative episode rewards.
    """

    def __init__(self, env_name: str = "CartPole-v1",
                 max_steps: int = 500, render: bool = False,
                 seed: Optional[int] = None, penalty_value: float = -100.0,
                 param_bounds: tuple = (-2.0, 2.0), num_episodes: int = 5):
        """Initialize CartPole evaluator.

        Args:
            env_name: Gymnasium environment name
            max_steps: Maximum steps per episode
            render: Whether to render episodes
            seed: Random seed for environment
            penalty_value: Value for failed episodes
            param_bounds: Bounds for policy parameters
            num_episodes: Number of episodes to average over
        """
        self.env_name = env_name
        self.max_steps = max_steps
        self.render = render
        self.seed = seed
        self.penalty_value = penalty_value
        self.param_bounds = param_bounds
        self.num_episodes = num_episodes

        # Create environment
        render_mode = "human" if render else None
        try:
            self.env = gym.make(env_name, render_mode=render_mode)
        except Exception as e:
            # Fallback to v1 if specific version not available
            warnings.warn(f"Could not create {env_name}, trying CartPole-v1: {e}")
            self.env = gym.make("CartPole-v1", render_mode=render_mode)

        # Set seed if provided
        if seed is not None:
            self.env.reset(seed=seed)

        # Get environment dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # Create policy instance
        self.policy = LinearPolicy(self.obs_dim, self.action_dim)

        print(f"Initialized CartPole evaluator: {self.obs_dim}D obs → {self.action_dim} actions")
        print(f"Policy parameters: {self.policy.n_params} ({self.obs_dim}×{self.action_dim} weights + {self.action_dim} biases)")

    def evaluate(self, params: Union[Dict[str, Any], torch.Tensor]) -> EvaluationResult:
        """Evaluate linear policy parameters in CartPole.

        Args:
            params: Either parameter dict with 'params' key or tensor of policy parameters

        Returns:
            EvaluationResult with cumulative episode reward
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
                if torch.all(param_array >= 0) and torch.all(param_array <= 1):
                    param_array = param_array * (self.param_bounds[1] - self.param_bounds[0]) + self.param_bounds[0]
            else:
                param_array = np.array(params)

            # Validate parameter count
            if len(param_array) != self.policy.n_params:
                raise ValueError(f"Expected {self.policy.n_params} parameters, got {len(param_array)}")

            # Set policy parameters
            self.policy.set_params(param_array)

            # Run multiple episodes and average rewards
            episode_rewards = []
            for episode in range(self.num_episodes):
                obs, info = self.env.reset()
                total_reward = 0.0
                step_count = 0

                for step in range(self.max_steps):
                    # Get action from policy
                    action = self.policy.get_action(obs)

                    # Take step in environment
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    total_reward += reward
                    step_count += 1

                    # Episode finished
                    if terminated or truncated:
                        break

                episode_rewards.append(total_reward)

            # Calculate average reward
            avg_reward = np.mean(episode_rewards)

            # Create evaluation result
            return EvaluationResult.from_true_value(params, avg_reward)

        except Exception as e:
            print(f"Error evaluating policy: {e}")
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
        return False  # Stochastic due to random initial conditions
