"""
ExperienceBuffer.py - Experience buffer module.

Stores historical execution experience, including edge information, embeddings, rewards, etc.
Supports random sampling, retrieving recent experiences, and saving/loading to disk.

Author: AgentAC Team
"""

import torch
import numpy as np
import pickle
import os
import time
from typing import List, Dict, Optional
from collections import deque

class ExperienceBuffer:
    """
    Experience buffer for storing historical execution experiences,
    including edge information, embeddings, rewards, and other metadata.
    """
    
    def __init__(self, max_size: int = 10000) -> None:
        """
        Initialize the experience buffer.
        
        Args:
            max_size: Maximum capacity of the buffer.
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        
    def add_experience(self, 
                      edge_info: str,
                      edge_embedding: torch.Tensor,
                      ans_embedding: torch.Tensor,
                      critic_value: float,
                      reward: float,
                      utility: float,
                      edge_type: str,
                      node_info: Dict[str, str]) -> None:
        """
        Add a single experience entry to the buffer.
        
        Args:
            edge_info: Text description of the edge.
            edge_embedding: Edge embedding vector.
            ans_embedding: Answer embedding vector.
            critic_value: Predicted value from the Critic.
            reward: Reward value.
            utility: Utility value.
            edge_type: Edge type ("spatial" or "temporal").
            node_info: Dictionary of node information.
        """
        experience: Dict = {
            'edge_info': edge_info,
            'edge_embedding': edge_embedding.detach().cpu(),
            'ans_embedding': ans_embedding.detach().cpu(),
            'critic_value': critic_value,
            'reward': reward,
            'utility': utility,
            'edge_type': edge_type,
            'node_info': node_info,
            'timestamp': time.time()
        }
        self.buffer.append(experience)
    
    def add(self, experience: Dict) -> None:
        """
        Add a single experience to the buffer (generic dict-based interface).

        Args:
            experience: Experience dict, may contain arbitrary fields.
        """
        # Add a timestamp if it is missing
        if 'timestamp' not in experience:
            experience['timestamp'] = time.time()
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """
        Randomly sample a batch of experiences (non-destructive, returns copies).

        Args:
            batch_size: Batch size.

        Returns:
            List[Dict]: List of sampled experience copies.
        """
        if len(self.buffer) < batch_size:
            return [dict(exp) for exp in self.buffer]  # Return shallow copies

        indices: np.ndarray = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [dict(self.buffer[i]) for i in indices]  # Return shallow copies
    
    def sample_batch(self, batch_size: int) -> List[Dict]:
        """
        Randomly sample a batch of experiences.
        
        Args:
            batch_size: Batch size.
            
        Returns:
            List[Dict]: List of sampled experiences.
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices: np.ndarray = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_recent_experiences(self, n: int) -> List[Dict]:
        """
        Get the most recent n experiences.
        
        Args:
            n: Number of experiences to retrieve.
            
        Returns:
            List[Dict]: List of recent experiences.
        """
        return list(self.buffer)[-n:] if len(self.buffer) >= n else list(self.buffer)
    
    def save_buffer(self, filepath: str) -> None:
        """
        Save the experience buffer to a file.
        
        Args:
            filepath: Path to the file to save to.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load_buffer(self, filepath: str) -> None:
        """
        Load the experience buffer from a file.
        
        Args:
            filepath: Path to the file to load from.
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.buffer = deque(pickle.load(f), maxlen=self.max_size)
    
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()
    
    def size(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            int: Number of stored experiences.
        """
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        """
        Check whether the buffer is empty.
        
        Returns:
            bool: True if empty, False otherwise.
        """
        return len(self.buffer) == 0
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics over the experiences currently in the buffer.
        
        Returns:
            Dict[str, float]: Dictionary of statistics (size, average reward/utility/critic value, etc.).
        """
        if self.is_empty():
            return {
                'size': 0,
                'avg_reward': 0.0,
                'avg_utility': 0.0,
                'avg_critic_value': 0.0
            }
        
        rewards = [exp['reward'] for exp in self.buffer]
        utilities = [exp['utility'] for exp in self.buffer]
        critic_values = [exp['critic_value'] for exp in self.buffer]
        
        return {
            'size': len(self.buffer),
            'avg_reward': np.mean(rewards),
            'avg_utility': np.mean(utilities),
            'avg_critic_value': np.mean(critic_values),
            'std_reward': np.std(rewards),
            'std_utility': np.std(utilities),
            'std_critic_value': np.std(critic_values)
        }