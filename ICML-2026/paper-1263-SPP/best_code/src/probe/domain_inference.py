"""
Domain inference module
Infers the domain or combination of domains a sample belongs to from probe predictions
Addresses domain identification in cross-domain and OOD scenarios
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import logging

from ..utils import get_logger

logger = get_logger(__name__)


class DomainInference:
    """
    Domain inferencer
    Infers the domain or combination of domains a sample belongs to from probe predictions

    Core idea:
    1. Different domains are distributed across different layers
    2. We want the domain distribution to be more concentrated (spread over as few layers as possible)
    3. Averaging over all layers weakens the score of domains that occupy fewer layers
    4. We need the most sensible design to identify the domain or combination of domains a sample belongs to
    """
    
    def __init__(
        self,
        num_domains: int,
        min_probability_threshold: float = 0.05,  # Minimum probability threshold (filters out low-probability domains)
        layer_importance_weight: float = 0.5,  # Layer importance weight (used for weighted averaging)
        use_top_k_layers: Optional[int] = None,  # Use only the Top-K layers (if None, use all layers)
        domain_concentration_weight: float = 0.3,  # Domain concentration weight (encourages selecting domains concentrated in a few layers)
        cross_domain_threshold: float = 0.15  # Cross-domain threshold (if multiple domain probabilities all exceed this value, treat as cross-domain)
    ):
        """
        Initialize the Domain inferencer

        Args:
            num_domains: number of domains
            min_probability_threshold: minimum probability threshold (filters out low-probability domains)
            layer_importance_weight: layer importance weight (used for weighted averaging)
            use_top_k_layers: use only the Top-K layers (if None, use all layers)
            domain_concentration_weight: domain concentration weight (encourages selecting domains concentrated in a few layers)
            cross_domain_threshold: cross-domain threshold
        """
        self.num_domains = num_domains
        self.min_probability_threshold = min_probability_threshold
        self.layer_importance_weight = layer_importance_weight
        self.use_top_k_layers = use_top_k_layers
        self.domain_concentration_weight = domain_concentration_weight
        self.cross_domain_threshold = cross_domain_threshold
        
        logger.info(f"Domain inferencer initialized:")
        logger.info(f"  Minimum probability threshold: {min_probability_threshold:.3f}")
        logger.info(f"  Layer importance weight: {layer_importance_weight:.3f}")
        logger.info(f"  Use Top-K layers: {use_top_k_layers if use_top_k_layers else 'all layers'}")
        logger.info(f"  Domain concentration weight: {domain_concentration_weight:.3f}")
        logger.info(f"  Cross-domain threshold: {cross_domain_threshold:.3f}")
    
    def infer_domains(
        self,
        layer_probe_predictions: Dict[int, torch.Tensor]
    ) -> Tuple[List[int], Dict]:
        """
        Infer the domain or combination of domains from probe predictions

        Strategy:
        1. Filter out low-probability domains (< min_probability_threshold)
        2. Consider the distribution of domains across layers (avoid weakening minority domains via layer averaging)
        3. Support multi-domain combinations (cross-domain scenarios)
        4. Consider domain concentration (encourage selecting domains concentrated in a few layers)

        Args:
            layer_probe_predictions: {layer_idx: domain_probabilities [num_domains]}

        Returns:
            (inferred_domains, metadata)
            - inferred_domains: list of inferred domains (may contain multiple domains)
            - metadata: dictionary containing detailed information
        """
        if not layer_probe_predictions:
            logger.warning("No probe predictions; returning empty domain list")
            return [], {'method': 'empty', 'reason': 'no_predictions'}

        # 1. Process the predictions of each layer
        layer_probs = {}
        for layer_idx, probs in layer_probe_predictions.items():
            if isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy()
            if isinstance(probs, np.ndarray):
                probs = probs.flatten()
            # Ensure it is a numpy array
            if not isinstance(probs, np.ndarray):
                probs = np.array(probs)
            # Normalize
            if probs.sum() > 0:
                probs = probs / probs.sum()
            layer_probs[layer_idx] = probs

        # 2. Compute the distribution of each domain across layers
        domain_layer_distribution = {}  # {domain_idx: {layer_idx: prob}}
        domain_layer_scores = {}  # {domain_idx: [layer_scores]}
        
        for domain_idx in range(self.num_domains):
            domain_layer_distribution[domain_idx] = {}
            domain_layer_scores[domain_idx] = []
            
            for layer_idx, probs in layer_probs.items():
                if domain_idx < len(probs):
                    prob = float(probs[domain_idx])
                    domain_layer_distribution[domain_idx][layer_idx] = prob
                    domain_layer_scores[domain_idx].append(prob)
        
        # 3. Compute a composite score for each domain (considering layer distribution and concentration)
        domain_scores = {}
        domain_metadata = {}

        for domain_idx in range(self.num_domains):
            layer_scores = domain_layer_scores[domain_idx]

            if not layer_scores:
                continue

            # 3.1 Compute the average probability (over all layers)
            avg_prob = np.mean(layer_scores)

            # 3.2 Compute the maximum probability (max over a single layer)
            max_prob = np.max(layer_scores)

            # 3.3 Compute the Top-K layer average (consider only the Top-K layers, to avoid dilution by low-probability layers)
            # This avoids weakening domains that occupy fewer layers
            sorted_scores = sorted(layer_scores, reverse=True)
            top_k = min(5, len(sorted_scores))  # Top-5 layers
            top_k_avg = np.mean(sorted_scores[:top_k])

            # 3.4 Compute domain concentration (in how many layers it has significant probability)
            # Higher concentration means the domain is concentrated in fewer layers, so the score should be higher
            significant_layers = sum(1 for score in layer_scores if score > self.min_probability_threshold)
            concentration = 1.0 / (significant_layers + 1)  # Fewer layers means higher concentration

            # 3.5 Compute the layer-importance weighted average (handled uniformly, not distinguishing layer types)
            weighted_avg = avg_prob

            # 3.6 Composite score (combining multiple metrics)
            # Use the Top-K average and concentration to avoid dilution by low-probability layers
            base_score = top_k_avg * 0.6 + max_prob * 0.4  # Top-K average and maximum probability
            concentration_bonus = concentration * self.domain_concentration_weight  # Concentration bonus
            final_score = base_score * (1 + concentration_bonus)
            
            domain_scores[domain_idx] = final_score
            domain_metadata[domain_idx] = {
                'avg_prob': float(avg_prob),
                'max_prob': float(max_prob),
                'top_k_avg': float(top_k_avg),
                'concentration': float(concentration),
                'significant_layers': significant_layers,
                'weighted_avg': float(weighted_avg),
                'final_score': float(final_score)
            }
        
        # 4. Filter out low-probability domains
        filtered_domains = {
            domain_idx: score
            for domain_idx, score in domain_scores.items()
            if score >= self.min_probability_threshold
        }

        if not filtered_domains:
            logger.warning("All domains were filtered out; using the domain with the maximum probability")
            if domain_scores:
                max_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
                return [max_domain], {
                    'method': 'fallback_max',
                    'domains': [max_domain],
                    'scores': {max_domain: domain_scores[max_domain]},
                    'metadata': {max_domain: domain_metadata[max_domain]}
                }
            else:
                return [], {'method': 'empty', 'reason': 'no_valid_domains'}
        
        # 5. Determine whether this is a cross-domain scenario
        # If multiple domains' scores all exceed cross_domain_threshold, treat as cross-domain
        high_score_domains = [
            domain_idx for domain_idx, score in filtered_domains.items()
            if score >= self.cross_domain_threshold
        ]
        
        if len(high_score_domains) > 1:
            # Cross-domain scenario: return multiple domains
            # Sort by score and select the Top-K domains
            sorted_domains = sorted(
                high_score_domains,
                key=lambda d: filtered_domains[d],
                reverse=True
            )
            # Select all domains exceeding cross_domain_threshold (a more lenient strategy)
            # If multiple domains are relevant, all should be selected, rather than only those with similar scores
            selected_domains = sorted_domains  # Select all domains exceeding the threshold

            # Optional: if there are too many domains, we could limit to Top-K (but usually there aren't many)
            # No limit imposed here for now; let all relevant domains participate

            logger.info(f"Detected cross-domain scenario: {len(selected_domains)} domains")
            logger.info(f"  Domain list: {selected_domains}")
            logger.info(f"  Domain scores: {[filtered_domains[d] for d in selected_domains]}")
            
            return selected_domains, {
                'method': 'cross_domain',
                'domains': selected_domains,
                'scores': {d: filtered_domains[d] for d in selected_domains},
                'metadata': {d: domain_metadata[d] for d in selected_domains},
                'is_cross_domain': True
            }
        else:
            # Single-domain scenario: return a single domain
            max_domain = max(filtered_domains.items(), key=lambda x: x[1])[0]
            logger.info(f"Detected single-domain scenario: domain={max_domain}, score={filtered_domains[max_domain]:.3f}")
            
            return [max_domain], {
                'method': 'single_domain',
                'domains': [max_domain],
                'scores': {max_domain: filtered_domains[max_domain]},
                'metadata': {max_domain: domain_metadata[max_domain]},
                'is_cross_domain': False
            }
    
    def get_domain_probabilities(
        self,
        layer_probe_predictions: Dict[int, torch.Tensor]
    ) -> np.ndarray:
        """
        Get the domain probability distribution (considering layer distribution and concentration)

        Args:
            layer_probe_predictions: {layer_idx: domain_probabilities [num_domains]}

        Returns:
            domain_probabilities: [num_domains] normalized probability distribution
        """
        inferred_domains, metadata = self.infer_domains(
            layer_probe_predictions
        )
        
        # Build the probability distribution
        probs = np.zeros(self.num_domains)
        if inferred_domains and 'scores' in metadata:
            # Build the probability distribution from the inferred domain scores
            total_score = sum(metadata['scores'].values())
            if total_score > 0:
                for domain_idx, score in metadata['scores'].items():
                    probs[domain_idx] = score / total_score
            else:
                # If the total score is 0, use a uniform distribution
                probs[inferred_domains] = 1.0 / len(inferred_domains)
        else:
            # Fallback: uniform distribution
            if inferred_domains:
                probs[inferred_domains] = 1.0 / len(inferred_domains)
        
        return probs
    
    def get_domain_similarity_probs(
        self,
        layer_probe_predictions: Dict[int, torch.Tensor]
    ) -> np.ndarray:
        """
        Get domain similarity probabilities (unnormalized)

        Strategy: aggregate the probe outputs across layers
        - Use a max strategy: take the maximum value for each domain across layers
        - Note: the returned probabilities do not sum to 1 (unnormalized)

        Args:
            layer_probe_predictions: {layer_idx: domain_probabilities [num_domains]}

        Returns:
            domain_similarity_probs: [num_domains] similarity probability for each domain (unnormalized)
        """
        if not layer_probe_predictions:
            logger.warning("No probe predictions; returning a zero vector")
            return np.zeros(self.num_domains)

        # Initialize domain probabilities
        domain_probs = np.zeros(self.num_domains)

        # Aggregate the probe outputs across layers
        for layer_idx, probs in layer_probe_predictions.items():
            if isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy()
            if isinstance(probs, np.ndarray):
                probs = probs.flatten()
            # Ensure it is a numpy array
            if not isinstance(probs, np.ndarray):
                probs = np.array(probs)

            # Ensure dimensions match
            if len(probs) == self.num_domains:
                # Use a max strategy: take the maximum value for each domain across layers
                domain_probs = np.maximum(domain_probs, probs)

        # Replace NaN and inf values with 0
        domain_probs = np.nan_to_num(domain_probs, nan=0.0, posinf=0.0, neginf=0.0)

        # If all values are zero, fall back to a uniform distribution
        if domain_probs.sum() == 0.0 or np.all(domain_probs == 0.0):
            logger.warning("All domain probabilities are zero or NaN; using uniform distribution as fallback")
            domain_probs = np.ones(self.num_domains) / self.num_domains

        logger.info(f"Domain similarity probabilities: {domain_probs}")
        logger.info(f"  Sum: {domain_probs.sum():.3f} (unnormalized)")

        return domain_probs






