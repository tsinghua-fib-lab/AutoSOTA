"""Agent quality and true label generation."""

import numpy as np
from typing import Tuple, List


def generate_true_labels(
    n_agents: int,
    n_items: int,
    agent_qualities: List[float],
    partial_correct_prob: float = 0.35,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Implemented Step 1: Generate true latent labels for all agent-item pairs.
    
    For each agent a and item i, sample z_ai:
        P(z_ai = 2) = q_a  (correct)
        P(z_ai = 1) = partial_correct_prob * (1 - q_a)  (partial)
        P(z_ai = 0) = (1 - partial_correct_prob) * (1 - q_a)  (incorrect)
    
    Args:
        n_agents: Number of agents
        n_items: Number of items per agent
        agent_qualities: Quality q_a for each agent
        partial_correct_prob: P(z=1 | z != 2)
        rng: Random number generator
    
    Returns:
        z: Array of shape (n_agents, n_items) with true labels {0, 1, 2}
    """
    if rng is None:
        rng = np.random.default_rng()
    
    z = np.zeros((n_agents, n_items), dtype=np.int32)
    
    for a, q_a in enumerate(agent_qualities):
        # Probability distribution over labels
        p_correct = q_a
        p_partial = partial_correct_prob * (1 - q_a)
        p_incorrect = (1 - partial_correct_prob) * (1 - q_a)
        
        probs = [p_incorrect, p_partial, p_correct]
        
        # Sample labels for all items of this agent
        z[a] = rng.choice([0, 1, 2], size=n_items, p=probs)
    
    return z


def generate_item_ambiguity(
    n_agents: int,
    n_items: int,
    hard_item_prob: float = 0.2,
    easy_beta_params: Tuple[float, float] = (2, 12),
    hard_beta_params: Tuple[float, float] = (6, 3),
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 2: Generate item ambiguity values.
    
    For each item:
        1. Sample h_ai ~ Bernoulli(hard_item_prob)
        2. If h_ai = 0 (easy): d_ai ~ Beta(easy_params)
           If h_ai = 1 (hard): d_ai ~ Beta(hard_params)
    
    Args:
        n_agents: Number of agents
        n_items: Number of items per agent
        hard_item_prob: Probability of hard item
        easy_beta_params: Beta parameters for easy items
        hard_beta_params: Beta parameters for hard items
        rng: Random number generator
    
    Returns:
        h: Array of shape (n_agents, n_items) indicating hard items
        d: Array of shape (n_agents, n_items) with ambiguity values in [0, 1]
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sample hard/easy indicator
    h = rng.binomial(1, hard_item_prob, size=(n_agents, n_items))
    
    # Sample ambiguity values
    d = np.zeros((n_agents, n_items))
    
    easy_mask = (h == 0)
    hard_mask = (h == 1)
    
    n_easy = easy_mask.sum()
    n_hard = hard_mask.sum()
    
    if n_easy > 0:
        d[easy_mask] = rng.beta(easy_beta_params[0], easy_beta_params[1], size=n_easy)
    if n_hard > 0:
        d[hard_mask] = rng.beta(hard_beta_params[0], hard_beta_params[1], size=n_hard)
    
    return h, d


def compute_ground_truth_scores(
    z: np.ndarray,
    credit_mapping: List[float]
) -> np.ndarray:
    """
    Implemented Step 1: Compute ground truth scores for each agent.
    
    S*_a = (1/N) * sum_i v(z_ai)
    
    Args:
        z: True labels of shape (n_agents, n_items)
        credit_mapping: Credit values v for each class [v_0, v_1, v_2]
    
    Returns:
        scores: Array of shape (n_agents,) with ground truth scores
    """
    credit = np.array(credit_mapping)
    
    # Map labels to credits and compute mean per agent
    credits_per_item = credit[z]  # Shape: (n_agents, n_items)
    scores = credits_per_item.mean(axis=1)
    
    return scores


def get_ground_truth_ranking(scores: np.ndarray) -> np.ndarray:
    """
    Helper function to get ranking of agents by score in descending order
    
    Args:
        scores: Agent scores
    
    Returns:
        ranking: Agent indices sorted by score (best first)
    """
    return np.argsort(-scores)