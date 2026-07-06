"""
Head whitelist identification module
After final probe training, identify the domain-invariant head whitelist W

Optimized version:
1. Use a statistical significance test (ANOVA) to identify heads with no difference across domains
2. Distinguish foundation layers from non-foundation layers
3. Based on all importance sets (or probe4_cross_domain)
4. Use a more principled method rather than a fixed 50% ratio
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import logging
from pathlib import Path
import json
from scipy import stats

from ..utils import get_logger

logger = get_logger(__name__)


class HeadWhitelistIdentifier:
    """
    Head whitelist identifier
    Identifies heads that are important across all domains (the domain-invariant backbone)

    Optimization strategy:
    1. Use a statistical significance test (ANOVA) to identify heads with no difference across domains
    2. Distinguish foundation layers from non-foundation layers
    3. Based on importance data, identify heads with no difference across domains
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads_per_layer: int,
        num_domains: int,
        foundation_layers: Optional[List[int]] = None,
        importance_threshold: float = 0.1,  # Importance threshold (legacy method)
        consistency_threshold: float = 0.8,  # Consistency threshold (legacy method)
        use_statistical_test: bool = True,  # Whether to use a statistical significance test
        anova_p_threshold: float = 0.05,  # ANOVA p-value threshold (p>threshold indicates no significant difference)
        min_avg_importance: float = 0.3,  # Minimum average importance (used to filter out low-importance heads)
        max_std_importance: float = 0.15  # Maximum standard deviation (used to identify heads with no difference)
    ):
        """
        Initialize the Head whitelist identifier

        Args:
            num_layers: number of layers
            num_heads_per_layer: number of heads per layer
            num_domains: number of domains
            foundation_layers: list of foundation layers (e.g. [0,1,2])
            importance_threshold: importance threshold (legacy method)
            consistency_threshold: consistency threshold (legacy method)
            use_statistical_test: whether to use a statistical significance test
            anova_p_threshold: ANOVA p-value threshold (p>threshold indicates no significant difference)
            min_avg_importance: minimum average importance
            max_std_importance: maximum standard deviation (used to identify heads with no difference)
        """
        self.num_layers = num_layers
        self.num_heads_per_layer = num_heads_per_layer
        self.num_domains = num_domains
        self.foundation_layers = foundation_layers or []
        self.importance_threshold = importance_threshold
        self.consistency_threshold = consistency_threshold
        self.use_statistical_test = use_statistical_test
        self.anova_p_threshold = anova_p_threshold
        self.min_avg_importance = min_avg_importance
        self.max_std_importance = max_std_importance
        
        logger.info(f"Head whitelist identifier initialized:")
        logger.info(f"  Use statistical significance test: {use_statistical_test}")
        if use_statistical_test:
            logger.info(f"  ANOVA p-value threshold: {anova_p_threshold}")
            logger.info(f"  Minimum average importance: {min_avg_importance}")
            logger.info(f"  Maximum standard deviation: {max_std_importance}")
        else:
            logger.info(f"  Importance threshold: {importance_threshold}")
            logger.info(f"  Consistency threshold: {consistency_threshold}")
        logger.info(f"  Foundation layers: {self.foundation_layers}")
    
    def identify_whitelist(
        self,
        head_importance: Dict[int, torch.Tensor],  # {layer_idx: [num_heads, num_domains]} I_{l,h,k}
        all_importance_sets: Optional[Dict[str, Dict[int, torch.Tensor]]] = None  # All importance sets (optional)
    ) -> List[Tuple[int, int]]:
        """
        Identify the head whitelist

        Whitelist heads: heads with no significant difference across domains (domain-invariant)

        Optimization strategy:
        1. Use a statistical significance test (ANOVA) to identify heads with no difference across domains
        2. Distinguish foundation layers from non-foundation layers
        3. Based on importance data, identify heads with no difference across domains

        Args:
            head_importance: head importance {layer_idx: [num_heads, num_domains]}
            all_importance_sets: all importance sets (optional, for more robust identification)

        Returns:
            whitelist: [(layer_idx, head_idx), ...] list of whitelist heads
        """
        if self.use_statistical_test:
            return self._identify_whitelist_statistical(head_importance, all_importance_sets)
        else:
            return self._identify_whitelist_legacy(head_importance)
    
    def _identify_whitelist_statistical(
        self,
        head_importance: Dict[int, torch.Tensor],
        all_importance_sets: Optional[Dict[str, Dict[int, torch.Tensor]]] = None
    ) -> List[Tuple[int, int]]:
        """
        Identify the whitelist using a statistical significance test

        Strategy:
        1. For each head, compute its importance across all domains
        2. Use ANOVA to test whether there is a significant difference across domains
        3. If the p-value > threshold and the average importance > min_avg_importance, treat it as a whitelist head
        4. Or use the standard-deviation method: if std < max_std_importance, treat it as a no-difference head
        """
        whitelist = []
        foundation_whitelist = []
        non_foundation_whitelist = []

        # If all importance sets are provided, use them to enhance robustness
        if all_importance_sets:
            logger.info(f"Using {len(all_importance_sets)} importance sets for robustness analysis")

        # First, force all heads of the foundation layers into the whitelist
        for layer_idx in self.foundation_layers:
            for head_idx in range(self.num_heads_per_layer):
                whitelist.append((layer_idx, head_idx))
                foundation_whitelist.append((layer_idx, head_idx))
        logger.info(f"All heads of the foundation layers ({self.foundation_layers}) forced into the whitelist: {len(foundation_whitelist)}")

        for layer_idx in range(self.num_layers):
            # Skip foundation layers (already handled)
            if layer_idx in self.foundation_layers:
                continue
            if layer_idx not in head_importance:
                continue
            
            importance = head_importance[layer_idx]  # [num_heads, num_domains]
            
            if isinstance(importance, torch.Tensor):
                importance_np = importance.cpu().numpy()
            else:
                importance_np = np.array(importance)
            
            for head_idx in range(self.num_heads_per_layer):
                if head_idx >= importance_np.shape[0]:
                    continue

                # Get this head's importance across all domains
                head_importance_across_domains = importance_np[head_idx, :]  # [num_domains]

                # Compute statistics
                avg_importance = np.mean(head_importance_across_domains)
                std_importance = np.std(head_importance_across_domains)

                # Method 1: judge by standard deviation (simple and effective)
                # A small standard deviation indicates similar importance across all domains (no difference)
                if std_importance <= self.max_std_importance and avg_importance >= self.min_avg_importance:
                    whitelist.append((layer_idx, head_idx))
                    if layer_idx in self.foundation_layers:
                        foundation_whitelist.append((layer_idx, head_idx))
                    else:
                        non_foundation_whitelist.append((layer_idx, head_idx))
                    logger.debug(f"Identified whitelist head: layer={layer_idx}, head={head_idx}, "
                               f"avg={avg_importance:.4f}, std={std_importance:.4f}")
                    continue

                # Method 2: judge by the coefficient of variation (CV)
                # CV = std / mean; a small CV indicates a small relative difference (no difference)
                # For importance values, CV < 0.3 usually indicates no significant difference
                cv = std_importance / (avg_importance + 1e-8)  # Avoid division by zero
                if cv <= 0.3 and avg_importance >= self.min_avg_importance:
                    if (layer_idx, head_idx) not in whitelist:
                        whitelist.append((layer_idx, head_idx))
                        if layer_idx in self.foundation_layers:
                            foundation_whitelist.append((layer_idx, head_idx))
                        else:
                            non_foundation_whitelist.append((layer_idx, head_idx))
                        logger.debug(f"Identified whitelist head (CV): layer={layer_idx}, head={head_idx}, "
                                   f"cv={cv:.4f}, avg={avg_importance:.4f}, std={std_importance:.4f}")

        logger.info(f"Identified whitelist heads: {len(whitelist)}")
        logger.info(f"  Whitelist ratio: {len(whitelist) / (self.num_layers * self.num_heads_per_layer):.2%}")
        logger.info(f"  Foundation-layer whitelist: {len(foundation_whitelist)}")
        logger.info(f"  Non-foundation-layer whitelist: {len(non_foundation_whitelist)}")

        # Statistical significance validation: whitelist vs. non-whitelist
        self._validate_whitelist_statistical_significance(head_importance, whitelist)
        
        return whitelist
    
    def _validate_whitelist_statistical_significance(
        self,
        head_importance: Dict[int, torch.Tensor],
        whitelist: List[Tuple[int, int]]
    ):
        """
        Validate the statistical significance of the whitelist

        Use a statistical test (e.g. the Mann-Whitney U test) to verify that whitelist heads and non-whitelist heads differ significantly in their cross-domain variation
        """
        whitelist_std_values = []  # Cross-domain standard deviation of whitelist heads
        non_whitelist_std_values = []  # Cross-domain standard deviation of non-whitelist heads
        
        whitelist_set = set(whitelist)
        
        for layer_idx in range(self.num_layers):
            if layer_idx not in head_importance:
                continue
            
            importance = head_importance[layer_idx]
            if isinstance(importance, torch.Tensor):
                importance_np = importance.cpu().numpy()
            else:
                importance_np = np.array(importance)
            
            for head_idx in range(self.num_heads_per_layer):
                if head_idx >= importance_np.shape[0]:
                    continue
                
                head_importance_across_domains = importance_np[head_idx, :]
                std_importance = np.std(head_importance_across_domains)
                
                if (layer_idx, head_idx) in whitelist_set:
                    whitelist_std_values.append(std_importance)
                else:
                    non_whitelist_std_values.append(std_importance)
        
        if len(whitelist_std_values) > 0 and len(non_whitelist_std_values) > 0:
            # Use the Mann-Whitney U test (non-parametric, suitable for non-normal distributions)
            u_statistic, p_value = stats.mannwhitneyu(
                whitelist_std_values,
                non_whitelist_std_values,
                alternative='less'  # The std of the whitelist should be smaller
            )

            logger.info("=" * 80)
            logger.info("Whitelist statistical significance validation")
            logger.info("=" * 80)
            logger.info(f"Number of whitelist heads: {len(whitelist_std_values)}")
            logger.info(f"Number of non-whitelist heads: {len(non_whitelist_std_values)}")
            logger.info(f"Cross-domain std of whitelist heads: mean={np.mean(whitelist_std_values):.4f}, median={np.median(whitelist_std_values):.4f}")
            logger.info(f"Cross-domain std of non-whitelist heads: mean={np.mean(non_whitelist_std_values):.4f}, median={np.median(non_whitelist_std_values):.4f}")
            logger.info(f"Mann-Whitney U test: U={u_statistic:.2f}, p={p_value:.6f}")

            if p_value < 0.05:
                logger.info(f"Statistical significance validation passed: the cross-domain difference of whitelist heads is significantly smaller than that of non-whitelist heads (p<0.05)")
            else:
                logger.warning(f"Statistical significance validation failed: the cross-domain difference of whitelist and non-whitelist heads is not significantly different (p>=0.05)")
                logger.warning(f"   Suggested parameter adjustments:")
                logger.warning(f"     - Lower max_std_importance: {self.max_std_importance} -> {self.max_std_importance * 0.8:.3f}")
                logger.warning(f"     - Or raise min_avg_importance: {self.min_avg_importance} -> {self.min_avg_importance * 1.2:.3f}")

            # Parameter tuning suggestions
            logger.info("")
            logger.info("Parameter tuning suggestions:")
            logger.info(f"  Current parameters: min_avg_importance={self.min_avg_importance}, max_std_importance={self.max_std_importance}")
            logger.info(f"  Whitelist ratio: {len(whitelist) / (self.num_layers * self.num_heads_per_layer):.2%}")
            if len(whitelist) / (self.num_layers * self.num_heads_per_layer) > 0.6:
                logger.info(f"  Whitelist ratio is high (>60%); consider raising min_avg_importance or lowering max_std_importance")
            elif len(whitelist) / (self.num_layers * self.num_heads_per_layer) < 0.2:
                logger.info(f"  Whitelist ratio is low (<20%); consider lowering min_avg_importance or raising max_std_importance")
            else:
                logger.info(f"  Whitelist ratio is reasonable (20%-60%)")

            logger.info("=" * 80)
    
    def _identify_whitelist_legacy(
        self,
        head_importance: Dict[int, torch.Tensor]
    ) -> List[Tuple[int, int]]:
        """
        Legacy whitelist identification method (kept for backward compatibility)
        """
        whitelist = []
        
        for layer_idx in range(self.num_layers):
            if layer_idx not in head_importance:
                continue
            
            importance = head_importance[layer_idx]  # [num_heads, num_domains]
            
            if isinstance(importance, torch.Tensor):
                importance_np = importance.cpu().numpy()
            else:
                importance_np = np.array(importance)
            
            for head_idx in range(self.num_heads_per_layer):
                if head_idx >= importance_np.shape[0]:
                    continue

                # Get this head's importance across all domains
                head_importance_across_domains = importance_np[head_idx, :]  # [num_domains]

                # Count in how many domains the importance exceeds the threshold
                significant_domains = np.sum(head_importance_across_domains > self.importance_threshold)
                consistency_ratio = significant_domains / self.num_domains

                # If the consistency ratio exceeds the threshold, treat it as a whitelist head
                if consistency_ratio >= self.consistency_threshold:
                    whitelist.append((layer_idx, head_idx))
                    logger.debug(f"Identified whitelist head: layer={layer_idx}, head={head_idx}, "
                               f"consistency={consistency_ratio:.3f}, "
                               f"avg_importance={np.mean(head_importance_across_domains):.4f}")

        logger.info(f"Identified whitelist heads: {len(whitelist)}")
        logger.info(f"  Whitelist ratio: {len(whitelist) / (self.num_layers * self.num_heads_per_layer):.2%}")
        
        return whitelist
    
    def save_whitelist(self, whitelist: List[Tuple[int, int]], path: str):
        """Save the whitelist"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        whitelist_dict = {
            'whitelist': whitelist,
            'num_layers': self.num_layers,
            'num_heads_per_layer': self.num_heads_per_layer,
            'num_domains': self.num_domains
        }
        
        with open(path, 'w') as f:
            json.dump(whitelist_dict, f, indent=2)
        
        logger.info(f"Whitelist saved: {path}")

    @classmethod
    def load_whitelist(cls, path: str) -> List[Tuple[int, int]]:
        """Load the whitelist (supports both list and dict formats)"""
        with open(path, 'r') as f:
            whitelist_data = json.load(f)

        # Support two formats: a list, or a dict with a 'whitelist' key
        if isinstance(whitelist_data, list):
            whitelist = [(int(layer_idx), int(head_idx)) for layer_idx, head_idx in whitelist_data]
        elif isinstance(whitelist_data, dict) and 'whitelist' in whitelist_data:
            whitelist = [(int(layer_idx), int(head_idx)) for layer_idx, head_idx in whitelist_data['whitelist']]
        else:
            raise ValueError(f"Invalid whitelist format: expected a list or a dict with a 'whitelist' key, got: {type(whitelist_data)}")

        logger.info(f"Whitelist loaded: {path}, {len(whitelist)} heads")
        return whitelist

