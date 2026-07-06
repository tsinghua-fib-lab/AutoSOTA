"""
Session-level pruning module
Prunes based on a session-level description, supporting multi-turn scenarios
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import re
from collections import Counter

from ..utils import get_logger
from .domain_inference import DomainInference

logger = get_logger(__name__)


class SessionPruner:
    """
    Session-level pruner
    Prunes based on a session-level description; subsequent turns reuse the first pruning result
    """

    # List of common words (frequent words to be removed)
    COMMON_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'what', 'which', 'who', 'when', 'where', 'why', 'how', 'if', 'then', 'than', 'so',
        'because', 'although', 'though', 'while', 'until', 'since', 'during', 'before', 'after'
    }
    
    def __init__(
        self,
        layer_probes,
        domain_inference: DomainInference,
        selected_domains: List[str],
        retention_rate: float = 0.6,
        num_heads_per_layer: int = 12,
        pruning_strength: float = 0.5,
        head_budget: Optional[int] = None,
        rho_min: float = 0.25,
        whitelist: Optional[List[Tuple[int, int]]] = None,
        head_importance: Optional[Dict[int, torch.Tensor]] = None,
        head_importance_path: Optional[str] = None
    ):
        """
        Initialize the Session-level pruner

        Following the paper's design, uses the pruning-strength parameter eta and a budget mapping

        Args:
            layer_probes: trained probes
            domain_inference: Domain inferencer
            selected_domains: list of selected domains (K_*)
            retention_rate: retention rate (legacy parameter, kept for compatibility)
            num_heads_per_layer: number of heads per layer
            pruning_strength: pruning strength parameter eta in (0, 1], controls the overall pruning strength
            head_budget: head budget K_B (if None, computed from retention_rate)
            rho_min: minimum retention rate (floor keep-ratio)
            whitelist: whitelist heads, always retained [(layer_idx, head_idx), ...]
        """
        self.layer_probes = layer_probes
        self.domain_inference = domain_inference
        self.selected_domains = selected_domains
        self.num_heads_per_layer = num_heads_per_layer
        self.pruning_strength = pruning_strength
        self.head_budget = head_budget
        self.rho_min = rho_min
        self.whitelist = whitelist or []

        # Ablation experiment flags (all enabled by default)
        self.use_multi_domain_mixing = True  # Whether to use multi-domain mixing
        self.use_probe_alignment = True  # Whether to use the probe alignment score

        # Load head importance (required)
        if head_importance is not None:
            self._head_importance = head_importance
        elif head_importance_path is not None:
            try:
                from .head_importance import HeadImportanceCalculator
                self._head_importance = HeadImportanceCalculator.load(head_importance_path)
                logger.info(f"Loaded head importance from {head_importance_path}")
            except Exception as e:
                raise ValueError(
                    f"Failed to load head importance file {head_importance_path}: {e}. "
                    f"Please ensure the file exists and is correctly formatted, or provide the head_importance parameter directly."
                )
        else:
            raise ValueError(
                "head_importance or head_importance_path must be provided. "
                "Head importance is a core component of pruning; a simplified method cannot be used. "
                "Please compute and save head importance first."
            )

        # Backward compatibility with legacy parameters
        if head_budget is None:
            # Estimate head_budget from retention_rate
            # Assume L layers, each with H heads
            # The total number of layers is needed here; for now use retention_rate
            self.retention_rate = retention_rate
        else:
            self.retention_rate = None

        # Store the first pruning result (for subsequent turns)
        self.first_pruning_mask: Optional[Dict[int, List[int]]] = None

        logger.info(f"Session-level pruner initialized:")
        logger.info(f"  Retention rate: {retention_rate:.1%}")
        logger.info(f"  Heads per layer: {num_heads_per_layer}")
        logger.info(f"  Selected domains: {selected_domains}")
    
    def extract_session_description(
        self,
        session_data: Dict,
        remove_common_words: bool = True
    ) -> str:
        """
        Extract the description from session data

        Args:
            session_data: session data, containing turns or a description
            remove_common_words: whether to remove common words

        Returns:
            session description text
        """
        # Prefer the session-level description
        if 'description' in session_data and session_data['description']:
            description = session_data['description']
        # Use the concatenated prompts of all turns (multi-turn context)
        elif 'turns' in session_data and len(session_data['turns']) > 0:
            # Concatenate prompts from all turns for full context
            # Limit to first 8 turns to avoid excessive length
            max_turns_for_context = 8
            turn_prompts = []
            for turn in session_data['turns'][:max_turns_for_context]:
                prompt = turn.get('prompt', '') or turn.get('query', '')
                if prompt:
                    turn_prompts.append(prompt)
            description = ' '.join(turn_prompts)
            if not description:
                raise ValueError(
                    "Failed to extract description from session data. "
                    "Please ensure session_data contains a 'description' field, or that 'turns' contains a valid 'prompt' or 'query'."
                )
        else:
            raise ValueError(
                "Failed to extract the session description. "
                "Please ensure session_data contains a 'description' field or a 'turns' list."
            )

        if remove_common_words:
            description = self._remove_common_words(description)

        return description
    
    def _remove_common_words(self, text: str) -> str:
        """
        Remove common words

        Args:
            text: the original text

        Returns:
            the text with common words removed
        """
        # Convert to lowercase and tokenize
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter out common words
        filtered_words = [w for w in words if w not in self.COMMON_WORDS and len(w) >= 3]

        # Recombine
        filtered_text = ' '.join(filtered_words)

        return filtered_text
    
    def prune_for_session(
        self,
        session_data: Dict,
        base_model,
        is_first_turn: bool = True,
        return_metadata: bool = False
    ) -> Dict[int, List[int]] | Tuple[Dict[int, List[int]], Dict[str, Any]]:
        """
        Prune for a session

        Args:
            session_data: session data
            base_model: the base model
            is_first_turn: whether this is the first turn
            return_metadata: whether to return pruning-related metadata

        Returns:
            if return_metadata=False: pruning mask {layer_idx: [retained head indices]}
            if return_metadata=True: (pruning mask, metadata dict)
        """
        # If not the first turn, directly return the first pruning result
        if not is_first_turn and self.first_pruning_mask is not None:
            logger.info("Using the first pruning result (reused by subsequent turns)")
            return self.first_pruning_mask.copy()

        # First turn: prune based on the session description
        logger.info("First turn: pruning based on the session description")

        # Extract the session description
        description = self.extract_session_description(session_data)
        logger.info(f"Session description: {description[:100]}...")

        # Use the probes to infer the domain
        # First obtain the activations
        inputs = base_model.tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(base_model.device)
        
        # Obtain the activations
        activations = base_model.get_activations(
            inputs,
            layer_indices=list(range(base_model.get_num_layers()))
        )

        # Predict using the probes
        layer_probe_predictions = {}
        for layer_idx, activation in activations.items():
            if layer_idx in self.layer_probes.probes:
                probe = self.layer_probes.probes[layer_idx]
                with torch.no_grad():
                    # Mean-pool the activations (cast to FP32 first to avoid overflow)
                    if len(activation.shape) > 2:
                        activation_pooled = activation.float().mean(dim=1)  # (batch, hidden_dim)
                    else:
                        activation_pooled = activation.float()

                    # Probe prediction (using sigmoid, 1-vs-rest)
                    logits = probe(activation_pooled)
                    probs = torch.sigmoid(logits)  # [batch, num_domains] (unnormalized)
                    layer_probe_predictions[layer_idx] = probs[0]  # Take the first sample [num_domains]

        # Session breadth will be computed outside _prune_based_on_domains

        # Get domain similarity probabilities (unnormalized)
        domain_similarity_probs = self.domain_inference.get_domain_similarity_probs(
            layer_probe_predictions
        )
        logger.info(f"Domain similarity probabilities: {domain_similarity_probs}")

        # Infer the domain (for logging)
        inferred_domains, metadata = self.domain_inference.infer_domains(
            layer_probe_predictions
        )
        logger.info(f"Inferred domains: {inferred_domains}")

        # Perform full pruning based on domain similarity probabilities and probe importance
        # Get head importance
        head_importance = self._head_importance
        if head_importance is not None:
            # If it is a HeadImportanceCalculator object, get the importance cache
            if hasattr(head_importance, 'importance_cache'):
                head_importance = head_importance.importance_cache
            elif hasattr(head_importance, 'get_importance'):
                # Convert to dict format
                head_importance_dict = {}
                for layer_idx in range(base_model.get_num_layers()):
                    imp = head_importance.get_importance(layer_idx)
                    if imp is not None:
                        head_importance_dict[layer_idx] = imp
                head_importance = head_importance_dict

        # Compute session breadth (computed externally and passed into _prune_based_on_domains)
        session_breadth = self._compute_session_breadth(layer_probe_predictions)

        pruning_mask, pruning_metadata = self._prune_based_on_domains(
            domain_similarity_probs,
            layer_probe_predictions,
            base_model.get_num_layers(),
            head_importance=head_importance,
            session_breadth=session_breadth,
            inferred_domains=inferred_domains,  # Pass inferred_domains for filtering
            return_metadata=return_metadata
        )

        # Save the first pruning result
        if is_first_turn:
            self.first_pruning_mask = pruning_mask.copy()
            logger.info("Saved the first pruning result; subsequent turns will reuse it")

        if return_metadata:
            # Add extra metadata
            pruning_metadata.update({
                "domain_similarity_probs": domain_similarity_probs.tolist() if isinstance(domain_similarity_probs, np.ndarray) else domain_similarity_probs,
                "session_breadth": session_breadth,
                "inferred_domains": inferred_domains,
                "layer_head_counts": {layer_idx: len(heads) for layer_idx, heads in pruning_mask.items()}
            })
            return pruning_mask, pruning_metadata
        else:
            return pruning_mask
    
    def _compute_session_breadth(
        self,
        layer_probe_predictions: Dict[int, torch.Tensor]
    ) -> float:
        """
        Compute session breadth c(s)

        Following Eq. (6) in the paper: c(s) = H(q(s)) / log(|K_*|)
        where q_k(s) = (1/L) * sum_{l=1}^L p_bar_{l,k}(s)

        Args:
            layer_probe_predictions: layer probe predictions {layer_idx: [num_domains]}

        Returns:
            session_breadth: c(s) in [0, 1]
        """
        # Compute the session-level domain mixture q_k(s)
        num_layers = len(layer_probe_predictions)
        session_domain_mixture = np.zeros(len(self.selected_domains))

        for layer_idx, probs in layer_probe_predictions.items():
            if isinstance(probs, torch.Tensor):
                probs_np = probs.cpu().numpy()
            else:
                probs_np = np.array(probs)

            # Ensure dimensions match
            if len(probs_np) == len(self.selected_domains):
                session_domain_mixture += probs_np

        if num_layers > 0:
            session_domain_mixture /= num_layers

        # Compute session breadth c(s)
        total_prob = session_domain_mixture.sum()
        if total_prob > 0:
            normalized_probs = session_domain_mixture / total_prob
            # Compute entropy
            entropy = -np.sum(normalized_probs * np.log(normalized_probs + 1e-8))
            # Normalize the entropy
            max_entropy = np.log(len(self.selected_domains))
            session_breadth = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            session_breadth = 0.0

        logger.info(f"Session breadth c(s) = {session_breadth:.3f}")
        return session_breadth
    
    def _prune_based_on_domains(
        self,
        domain_similarity_probs: np.ndarray,  # [num_domains] similarity probabilities (unnormalized)
        layer_probe_predictions: Dict[int, torch.Tensor],
        num_layers: int,
        head_importance: Optional[Dict[int, torch.Tensor]] = None,
        session_breadth: float = 0.5,
        inferred_domains: Optional[List[int]] = None,  # Domains identified by the probes (used for filtering)
        return_metadata: bool = False
    ) -> Dict[int, List[int]] | Tuple[Dict[int, List[int]], Dict[str, Any]]:
        """
        Process each selected domain separately, then take the union

        Revised logic:
        1. For each selected domain k:
           - Compute score: score_{l,h,k} = similarity_prob[k] * I_{l,h,k}
           - Select that domain's heads based on the pruning strength
        2. Take the union of the head sets across all selected domains

        Does not use domain relevance, since I_{l,h,k} may already contain the relevant information

        Args:
            domain_similarity_probs: domain similarity probabilities [num_domains] (unnormalized)
            layer_probe_predictions: layer probe predictions {layer_idx: [num_domains]} (sigmoid probabilities)
            num_layers: total number of layers
            head_importance: head importance {layer_idx: [num_heads, num_domains]} (I_{l,h,k})
            session_breadth: session breadth

        Returns:
            pruning mask {layer_idx: [retained head indices]}
        """
        # Store the heads selected for each domain
        all_domain_heads = set()  # {(layer_idx, head_idx)}

        # Compute the base keep-ratio (based on session breadth and pruning strength)
        base_keep_ratio = self.rho_min + self.pruning_strength * session_breadth
        base_keep_ratio = min(base_keep_ratio, 1.0)

        # Key fix: only prune domains in inferred_domains
        # If inferred_domains is empty or None, use selected_domains as a fallback
        domains_to_process = inferred_domains if inferred_domains and len(inferred_domains) > 0 else self.selected_domains

        # Ensure all domains in domains_to_process are in selected_domains (safety check)
        domains_to_process = [d for d in domains_to_process if d in self.selected_domains]

        if not domains_to_process:
            logger.warning("inferred_domains is empty and cannot fall back; using all selected_domains")
            domains_to_process = self.selected_domains

        # Ablation experiment: wo_multi_domain_mixing - use only the single domain with the highest similarity
        if not self.use_multi_domain_mixing:
            if len(domains_to_process) > 0:
                # Find the domain with the highest similarity
                max_similarity = -1
                best_domain = domains_to_process[0]
                for domain_idx in domains_to_process:
                    if domain_idx < len(domain_similarity_probs):
                        similarity = domain_similarity_probs[domain_idx]
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_domain = domain_idx
                domains_to_process = [best_domain]
                logger.info(f"Ablation setting: using only the single domain with the highest similarity {best_domain} (similarity={max_similarity:.3f})")

        logger.info(f"Starting to process pruning for each domain separately...")
        logger.info(f"  Inferred domains (used for pruning): {inferred_domains}")
        logger.info(f"  Selected domains (candidates): {self.selected_domains}")
        logger.info(f"  Domains actually processed: {domains_to_process}")
        logger.info(f"  Base keep-ratio: {base_keep_ratio:.3f} (rho_min={self.rho_min}, pruning_strength={self.pruning_strength}, session_breadth={session_breadth:.3f})")

        # Check whether all domain similarities are very low
        max_similarity = np.max(domain_similarity_probs) if len(domain_similarity_probs) > 0 else 0.0
        all_domains_low = max_similarity < 0.1

        # Compute similarity statistics for normalization
        valid_similarities = [s for s in domain_similarity_probs if s >= 0.1]  # Filter out similarities that are too low
        if valid_similarities:
            max_valid_similarity = max(valid_similarities)
            min_valid_similarity = min(valid_similarities)
        else:
            max_valid_similarity = max_similarity
            min_valid_similarity = 0.1

        # Initialize use_global_importance (used by the fallback mechanism below)
        use_global_importance = False

        if all_domains_low:
            logger.warning(
                f"All domain similarities are very low (max={max_similarity:.3f}); "
                f"will use global head importance as a fallback to avoid model collapse"
            )
            # Use global head importance (averaged over all domains)
            # This way important heads can still be selected even when no domain matches
            use_global_importance = True

        # Process each inferred domain separately (key fix: only process inferred_domains)
        for domain_idx in domains_to_process:
            domain_similarity = domain_similarity_probs[domain_idx]

            # If the similarity is too low, skip this domain (but if all domains are very low, use global importance)
            if domain_similarity < 0.1 and not use_global_importance:  # Threshold is adjustable
                logger.info(f"  Domain {domain_idx} similarity {domain_similarity:.3f} is too low; skipping")
                continue

            if use_global_importance:
                logger.info(f"  Processing Domain {domain_idx}: using global head importance (all domain similarities are very low)")
            else:
                logger.info(f"  Processing Domain {domain_idx}: similarity={domain_similarity:.3f}")

            # Compute this domain's head scores per layer
            domain_head_scores = {}  # {layer_idx: [num_heads]}

            for layer_idx in range(num_layers):
                # Get head importance
                if head_importance is not None and layer_idx in head_importance:
                    I_lh = head_importance[layer_idx]  # [num_heads, num_domains]

                    # Ensure dimensions match
                    if isinstance(I_lh, torch.Tensor):
                        I_lh = I_lh.cpu().numpy()

                    if use_global_importance:
                        # Use global head importance (averaged over all domains)
                        # This way important heads can still be selected even when no domain matches
                        I_lhk = np.mean(I_lh, axis=1)  # [num_heads] averaged over all domains
                        # Use a smaller weight to avoid over-retaining heads
                        effective_similarity = max(domain_similarity, 0.3)  # At least 0.3, to ensure not too few
                    else:
                        # Get this domain's head importance
                        if domain_idx < I_lh.shape[1]:
                            I_lhk = I_lh[:, domain_idx]  # [num_heads]
                        else:
                            logger.warning(f"Layer {layer_idx} domain {domain_idx} exceeds head importance dimensions; skipping")
                            continue
                        effective_similarity = domain_similarity
                else:
                    logger.warning(f"Layer {layer_idx} has no head importance data; skipping")
                    continue

                # Compute score: similarity_prob * I (relevance not used)
                # Ablation experiment: wo_probe_alignment - do not use the probe alignment score, use importance only
                if self.use_probe_alignment:
                    scores = effective_similarity * I_lhk
                else:
                    # Use importance only, not similarity
                    scores = I_lhk
                    logger.debug(f"Ablation setting: layer {layer_idx} domain {domain_idx} uses importance only, not the probe alignment score")
                domain_head_scores[layer_idx] = scores

            if not domain_head_scores:
                logger.warning(f"  Domain {domain_idx} has no valid head scores; skipping")
                continue

            # Select heads based on the pruning strength
            # Use an improved similarity adjustment: high-similarity domains retain more, low-similarity domains retain fewer
            if use_global_importance:
                effective_similarity = max(domain_similarity, 0.3)
            else:
                # Normalize similarity to [0,1], then use a nonlinear function to amplify the effect of high similarity
                # Formula: normalized_sim = (sim - min) / (max - min), then apply a power function
                if max_valid_similarity > min_valid_similarity:
                    normalized_sim = (domain_similarity - min_valid_similarity) / (max_valid_similarity - min_valid_similarity)
                    # Use a power function to amplify high similarity: normalized_sim^1.5
                    # This way high-similarity domains retain more heads and low-similarity domains retain fewer
                    adjusted_sim = normalized_sim ** 1.5
                    # Map back to the original range
                    effective_similarity = min_valid_similarity + adjusted_sim * (max_valid_similarity - min_valid_similarity)
                else:
                    effective_similarity = domain_similarity

            domain_heads = self._select_heads_for_domain(
                domain_head_scores,
                effective_similarity,
                base_keep_ratio,
                max_valid_similarity=max_valid_similarity,
                min_valid_similarity=min_valid_similarity
            )

            logger.info(f"  Domain {domain_idx} selected {len(domain_heads)} heads")

            # Add to the union
            all_domain_heads.update(domain_heads)

        logger.info(f"Union of heads across all domains: {len(all_domain_heads)} heads")

        # Convert to pruning_mask format
        pruning_mask = {}
        for layer_idx in range(num_layers):
            pruning_mask[layer_idx] = []

        # First add the whitelist heads
        whitelist_added = set()
        for layer_idx, head_idx in self.whitelist:
            if layer_idx < num_layers and head_idx < self.num_heads_per_layer:
                pruning_mask[layer_idx].append(head_idx)
                whitelist_added.add((layer_idx, head_idx))
                all_domain_heads.add((layer_idx, head_idx))  # Ensure the whitelist is also in the union

        # Add all heads selected by the domains
        for layer_idx, head_idx in all_domain_heads:
            if layer_idx < num_layers and head_idx < self.num_heads_per_layer:
                if head_idx not in pruning_mask[layer_idx]:
                    pruning_mask[layer_idx].append(head_idx)

        # Sort
        for layer_idx in range(num_layers):
            pruning_mask[layer_idx] = sorted(pruning_mask[layer_idx])

        # ========== Fallback mechanism: prevent structural collapse and over-pruning ==========
        # 1. Ensure each layer retains at least 1 head (prevent a layer from being completely empty)
        min_heads_per_layer = 1
        for layer_idx in range(num_layers):
            if len(pruning_mask[layer_idx]) < min_heads_per_layer:
                # If this layer has no heads, select the highest-scoring head for this layer across all domains
                # Collect this layer's head scores across all domains (using domains_to_process)
                layer_scores = {}
                max_similarity = np.max(domain_similarity_probs) if len(domain_similarity_probs) > 0 else 0.0
                use_global = max_similarity < 0.1

                for domain_idx in domains_to_process:
                    domain_similarity = domain_similarity_probs[domain_idx]
                    if domain_similarity < 0.1 and not use_global:
                        continue
                    if head_importance is not None and layer_idx in head_importance:
                        I_lh = head_importance[layer_idx]
                        if isinstance(I_lh, torch.Tensor):
                            I_lh = I_lh.cpu().numpy()

                        if use_global:
                            # Use global head importance
                            I_lhk = np.mean(I_lh, axis=1)
                            effective_similarity = max(domain_similarity, 0.3)
                        else:
                            if domain_idx < I_lh.shape[1]:
                                I_lhk = I_lh[:, domain_idx]
                                effective_similarity = domain_similarity
                            else:
                                continue

                        scores = effective_similarity * I_lhk
                        for head_idx, score in enumerate(scores):
                            if head_idx not in layer_scores or layer_scores[head_idx] < score:
                                layer_scores[head_idx] = score

                if layer_scores:
                    # Find the highest-scoring head for this layer (excluding those already in the whitelist)
                    available_heads = [
                        (h_idx, layer_scores[h_idx])
                        for h_idx in range(self.num_heads_per_layer)
                        if (layer_idx, h_idx) not in whitelist_added
                    ]
                    if available_heads:
                        best_head_idx, _ = max(available_heads, key=lambda x: x[1])
                        pruning_mask[layer_idx].append(best_head_idx)
                        logger.warning(
                            f"Layer {layer_idx} has no heads; forcibly retaining the highest-scoring head {best_head_idx} "
                            f"(to prevent structural collapse)"
                        )
                    else:
                        # If all heads are in the whitelist, select the first one
                        pruning_mask[layer_idx].append(0)
                        logger.warning(
                            f"All heads of layer {layer_idx} are in the whitelist; retaining head 0 "
                            f"(to prevent structural collapse)"
                        )
                else:
                    # If there are no scores, retain the first head
                    pruning_mask[layer_idx].append(0)
                    logger.warning(
                        f"Layer {layer_idx} has no score data; retaining head 0 (to prevent structural collapse)"
                    )
                pruning_mask[layer_idx] = sorted(pruning_mask[layer_idx])

        # 2. Ensure the total number of retained heads does not fall below a reasonable range
        # Prevent over-pruning: retain at least 5% of the total number of heads
        total_heads = num_layers * self.num_heads_per_layer
        min_total_heads = max(
            int(total_heads * 0.05),  # At least 5%
            num_layers * min_heads_per_layer  # At least 1 per layer
        )
        retained_heads = sum(len(heads) for heads in pruning_mask.values())

        if retained_heads < min_total_heads:
            logger.warning(
                f"The number of retained heads {retained_heads} is too few (minimum required {min_total_heads}); "
                f"forcibly topping up to the minimum requirement (to prevent over-pruning)"
            )
            # Collect the scores of all heads (for topping up, using domains_to_process)
            all_head_scores = []
            max_similarity = np.max(domain_similarity_probs) if len(domain_similarity_probs) > 0 else 0.0
            use_global = max_similarity < 0.1

            for layer_idx in range(num_layers):
                for domain_idx in domains_to_process:
                    domain_similarity = domain_similarity_probs[domain_idx]
                    if domain_similarity < 0.1 and not use_global:
                        continue
                    if head_importance is not None and layer_idx in head_importance:
                        I_lh = head_importance[layer_idx]
                        if isinstance(I_lh, torch.Tensor):
                            I_lh = I_lh.cpu().numpy()

                        if use_global:
                            # Use global head importance
                            I_lhk = np.mean(I_lh, axis=1)
                            effective_similarity = max(domain_similarity, 0.3)
                        else:
                            if domain_idx < I_lh.shape[1]:
                                I_lhk = I_lh[:, domain_idx]
                                effective_similarity = domain_similarity
                            else:
                                continue

                        scores = effective_similarity * I_lhk
                        for head_idx, score in enumerate(scores):
                            if (layer_idx, head_idx) not in whitelist_added:
                                all_head_scores.append((layer_idx, head_idx, score))

            # Sort by score
            all_head_scores.sort(key=lambda x: x[2], reverse=True)

            # Top up heads until the minimum requirement is met
            added_count = retained_heads
            for layer_idx, head_idx, score in all_head_scores:
                if added_count >= min_total_heads:
                    break
                if head_idx not in pruning_mask[layer_idx]:
                    pruning_mask[layer_idx].append(head_idx)
                    added_count += 1

            # Re-sort
            for layer_idx in range(num_layers):
                pruning_mask[layer_idx] = sorted(pruning_mask[layer_idx])

        # 3. Prevent over-pruning of some layers: each layer retains at least 10% of its heads (or at least 1)
        min_heads_per_layer_ratio = 0.1
        for layer_idx in range(num_layers):
            min_for_layer = max(
                int(self.num_heads_per_layer * min_heads_per_layer_ratio),
                min_heads_per_layer
            )
            if len(pruning_mask[layer_idx]) < min_for_layer:
                logger.warning(
                    f"Layer {layer_idx} retains too few heads ({len(pruning_mask[layer_idx])}); "
                    f"forcibly topping up to {min_for_layer} (to prevent over-pruning of the layer)"
                )
                # Collect this layer's head scores across all domains (using domains_to_process)
                layer_scores = {}
                max_similarity = np.max(domain_similarity_probs) if len(domain_similarity_probs) > 0 else 0.0
                use_global = max_similarity < 0.1

                for domain_idx in domains_to_process:
                    domain_similarity = domain_similarity_probs[domain_idx]
                    if domain_similarity < 0.1 and not use_global:
                        continue
                    if head_importance is not None and layer_idx in head_importance:
                        I_lh = head_importance[layer_idx]
                        if isinstance(I_lh, torch.Tensor):
                            I_lh = I_lh.cpu().numpy()

                        if use_global:
                            # Use global head importance
                            I_lhk = np.mean(I_lh, axis=1)
                            effective_similarity = max(domain_similarity, 0.3)
                        else:
                            if domain_idx < I_lh.shape[1]:
                                I_lhk = I_lh[:, domain_idx]
                                effective_similarity = domain_similarity
                            else:
                                continue

                        scores = effective_similarity * I_lhk
                        for head_idx, score in enumerate(scores):
                            if head_idx not in layer_scores or layer_scores[head_idx] < score:
                                layer_scores[head_idx] = score

                # Top up this layer's heads
                available_heads = [
                    (h_idx, layer_scores.get(h_idx, 0.0))
                    for h_idx in range(self.num_heads_per_layer)
                    if h_idx not in pruning_mask[layer_idx]
                ]
                available_heads.sort(key=lambda x: x[1], reverse=True)

                for h_idx, _ in available_heads[:min_for_layer - len(pruning_mask[layer_idx])]:
                    pruning_mask[layer_idx].append(h_idx)

                pruning_mask[layer_idx] = sorted(pruning_mask[layer_idx])

        # Compute the final sparsity (pruned fraction = 1 - retained fraction)
        total_heads = num_layers * self.num_heads_per_layer
        retained_heads = sum(len(heads) for heads in pruning_mask.values())
        retention_ratio = (retained_heads / total_heads) if total_heads > 0 else 0.0
        pruned_head_fraction = 1.0 - retention_ratio  # Pruned fraction (consistent with the pruning ratio semantics in the paper/tables)

        logger.info(
            f"Full pruning complete: pruned fraction={pruned_head_fraction:.2%}, retained fraction={retention_ratio:.2%}, "
            f"retained heads={retained_heads}/{total_heads}"
        )
        logger.info(f"  Fallback mechanism: at least {min_heads_per_layer} head per layer, at least {min_total_heads} heads globally")

        if return_metadata:
            # Collect all head scores for the return value (for visualization)
            all_head_scores_dict = {}  # {layer_idx: [num_heads]}
            max_similarity = np.max(domain_similarity_probs) if len(domain_similarity_probs) > 0 else 0.0
            use_global = max_similarity < 0.1

            for layer_idx in range(num_layers):
                layer_scores = np.zeros(self.num_heads_per_layer)
                for domain_idx in domains_to_process:
                    domain_similarity = domain_similarity_probs[domain_idx]
                    if domain_similarity < 0.1 and not use_global:
                        continue
                    if head_importance is not None and layer_idx in head_importance:
                        I_lh = head_importance[layer_idx]
                        if isinstance(I_lh, torch.Tensor):
                            I_lh = I_lh.cpu().numpy()

                        if use_global:
                            I_lhk = np.mean(I_lh, axis=1)
                            effective_similarity = max(domain_similarity, 0.3)
                        else:
                            if domain_idx < I_lh.shape[1]:
                                I_lhk = I_lh[:, domain_idx]
                                effective_similarity = domain_similarity
                            else:
                                continue

                        if self.use_probe_alignment:
                            scores = effective_similarity * I_lhk
                        else:
                            scores = I_lhk

                        # Take the maximum (when there are multiple domains)
                        layer_scores = np.maximum(layer_scores, scores)

                all_head_scores_dict[layer_idx] = layer_scores.tolist()

            metadata = {
                "selected_heads_count": retained_heads,
                "total_heads": total_heads,
                # pruning_ratio: pruned fraction (0~1); retention_ratio: retained fraction
                "pruning_ratio": pruned_head_fraction,
                "retention_ratio": retention_ratio,
                # sparsity is kept for compatibility with legacy fields: synonymous with pruning_ratio (both are the pruned fraction)
                "sparsity": pruned_head_fraction,
                "whitelist_size": len(whitelist_added),
                "use_global_importance": use_global_importance,
                "all_head_scores": all_head_scores_dict  # Add all head scores
            }
            return pruning_mask, metadata
        else:
            return pruning_mask
    
    def _select_heads_for_domain(
        self,
        domain_head_scores: Dict[int, np.ndarray],  # {layer_idx: [num_heads]}
        domain_similarity: float,
        base_keep_ratio: float,
        max_valid_similarity: float = 1.0,
        min_valid_similarity: float = 0.1
    ) -> List[Tuple[int, int]]:
        """
        Select the heads this domain should retain based on the pruning strength

        Improvement: use a nonlinear mapping so that high-similarity domains retain more heads and low-similarity domains retain fewer

        Args:
            domain_head_scores: this domain's head scores per layer
            domain_similarity: this domain's similarity probability (may already be adjusted)
            base_keep_ratio: base keep-ratio (based on session breadth and pruning strength)
            max_valid_similarity: maximum valid similarity (used for normalization, optional)
            min_valid_similarity: minimum valid similarity (used for normalization, optional)

        Returns:
            selected_heads: [(layer_idx, head_idx), ...]
        """
        # Compute the number of heads this domain should retain
        # Improved formula: use a nonlinear mapping to amplify the effect of high similarity
        # If domain_similarity was already adjusted before the call, use it directly
        # Otherwise, use normalization + a nonlinear function
        if max_valid_similarity > min_valid_similarity and domain_similarity >= min_valid_similarity:
            # Normalize to [0,1]
            normalized_sim = (domain_similarity - min_valid_similarity) / (max_valid_similarity - min_valid_similarity)
            # Use a power function to amplify high similarity: normalized_sim^1.5
            # This way high-similarity domains retain more heads and low-similarity domains retain fewer
            adjusted_sim = normalized_sim ** 1.5
            # Map back to the original range
            effective_similarity = min_valid_similarity + adjusted_sim * (max_valid_similarity - min_valid_similarity)
        else:
            effective_similarity = domain_similarity

        domain_keep_ratio = base_keep_ratio * effective_similarity
        domain_keep_ratio = min(domain_keep_ratio, 1.0)

        # Collect the scores of all heads
        all_scores = []
        for layer_idx, scores in domain_head_scores.items():
            for head_idx, score in enumerate(scores):
                all_scores.append((layer_idx, head_idx, score))

        if not all_scores:
            return []

        # Sort by score
        all_scores.sort(key=lambda x: x[2], reverse=True)

        # Select the top-K
        num_heads_to_keep = max(1, int(len(all_scores) * domain_keep_ratio))
        selected_heads = [(layer_idx, head_idx) for layer_idx, head_idx, _ in all_scores[:num_heads_to_keep]]

        return selected_heads


