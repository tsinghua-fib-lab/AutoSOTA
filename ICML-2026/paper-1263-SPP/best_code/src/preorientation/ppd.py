"""
Foundation-layer identification.

This module identifies the "foundation layers" L_0 used by the offline pipeline.
Foundation layers are domain-invariant layers (uniform, low domain relevance);
they are excluded from scenario-specific head importance / whitelist computation.

Note: the parameter-pathway-decoupling (PPD) model-training stage from the
research prototype is not part of the inference-time method and has been removed.
Only the lightweight, training-free foundation-layer analysis is kept here.
"""

import torch
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)


def identify_foundation_layers(
    layer_probes,
    base_model,
    domain_data: Dict[int, List[Dict[str, torch.Tensor]]],
    domain_axes: torch.Tensor,  # [num_domains, num_domains] one-hot
    entropy_threshold: float = 1.5,  # entropy threshold (theoretical max entropy for 8 domains is approx. 2.08)
    device: Optional[torch.device] = None,
    return_metadata: bool = True  # whether to return metadata (entropy and domain relevances)
) -> Dict:
    """
    Identify the foundation layers L_0.

    Foundation layers: layers with uniform domain relevance (high entropy).
    The per-layer domain relevance distribution is computed with the pretrained probes.

    Following the paper: the domain relevance of foundation layers is approximately uniform.

    Args:
        layer_probes: pretrained probes
        base_model: base model (used to obtain activations)
        domain_data: per-domain data {domain_k: [inputs]}
        domain_axes: domain axes [num_domains, num_domains] one-hot matrix
        entropy_threshold: entropy threshold (above this value a layer is considered a foundation layer)
        device: compute device

    Returns:
        foundation_layers: list of indices of the foundation layers L_0
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Identifying foundation layers...")

    # Per-layer domain relevance distribution
    layer_entropies = {}  # {layer_idx: entropy}

    num_domains = len(domain_axes)
    num_samples_per_domain = 20  # number of samples per domain

    for layer_idx in layer_probes.probes.keys():
        probe = layer_probes.probes[layer_idx]
        probe.eval()

        # Collect this layer's relevance over all domains
        domain_relevances = {k: [] for k in range(num_domains)}

        for domain_k in range(num_domains):
            if domain_k not in domain_data:
                continue

            domain_inputs_list = domain_data[domain_k]
            sample_inputs = domain_inputs_list[:num_samples_per_domain]

            for inputs in sample_inputs:
                try:
                    # Ensure inputs have a batch dimension (key fix: avoid 1D tensor issues)
                    batch_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            if v.dim() == 1:
                                batch_inputs[k] = v.unsqueeze(0)  # [seq_len] -> [1, seq_len]
                            elif v.dim() == 0:
                                # scalar, skip or handle
                                continue
                            else:
                                batch_inputs[k] = v
                        else:
                            batch_inputs[k] = v

                    # Obtain activations (post-attention residual)
                    activations = base_model.get_activations(
                        batch_inputs,
                        layer_indices=[layer_idx],
                        extract_post_attn_residual=True
                    )

                    if layer_idx in activations:
                        activation = activations[layer_idx]  # [batch, seq_len, hidden_dim] or [batch, hidden_dim] or [seq_len, hidden_dim]

                        # Pool to [batch, hidden_dim]
                        if len(activation.shape) == 3:
                            # [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
                            activation = activation.mean(dim=1)
                        elif len(activation.shape) == 2:
                            # could be [batch, hidden_dim] or [seq_len, hidden_dim]
                            if activation.shape[0] == 1 or activation.shape[1] > activation.shape[0]:
                                # if the second dimension is larger, it may be [seq_len, hidden_dim] and needs pooling
                                # but typically hidden_dim > seq_len, so if shape[1] > shape[0] it may be [seq_len, hidden_dim]
                                # to be safe, if the first dimension is small (<10) it may be seq_len and needs pooling
                                if activation.shape[0] < 10:
                                    # likely [seq_len, hidden_dim]; add a batch dimension and pool
                                    activation = activation.mean(dim=0, keepdim=True)  # [1, hidden_dim]
                                # otherwise assume [batch, hidden_dim] and leave unchanged
                            # if the first dimension is >= 10, assume [batch, hidden_dim] and leave unchanged
                        elif len(activation.shape) == 1:
                            # [hidden_dim] -> [1, hidden_dim]
                            activation = activation.unsqueeze(0)
                        else:
                            continue

                        # Finally ensure activation is 2D [batch, hidden_dim]
                        if len(activation.shape) != 2 or activation.shape[0] == 0 or activation.shape[1] == 0:
                            continue

                        # Compute domain relevance with the probe
                        with torch.no_grad():
                            try:
                                logits = probe(activation)  # input should be [batch, hidden_dim]
                                # logits may be [batch, num_domains] or [num_domains]

                                # Ensure logits is 2D
                                if len(logits.shape) == 1:
                                    logits = logits.unsqueeze(0)  # [num_domains] -> [1, num_domains]
                                elif len(logits.shape) == 0:
                                    continue
                                elif len(logits.shape) > 2:
                                    # take the first batch
                                    if logits.shape[0] > 0:
                                        logits = logits[0]
                                        if len(logits.shape) == 1:
                                            logits = logits.unsqueeze(0)
                                    else:
                                        continue

                                # Confirm again that logits is 2D
                                if len(logits.shape) != 2:
                                    continue

                                probs = torch.sigmoid(logits)  # [batch, num_domains]

                                # Ensure probs is 2D with the correct shape (re-check, since sigmoid may change the shape)
                                if len(probs.shape) == 1:
                                    # if probs is 1D, add a batch dimension
                                    probs = probs.unsqueeze(0)  # [num_domains] -> [1, num_domains]
                                elif len(probs.shape) == 0:
                                    continue
                                elif len(probs.shape) > 2:
                                    # if 3D or higher, flatten or take the first batch
                                    if probs.shape[0] > 0:
                                        probs = probs[0]  # take the first batch
                                        if len(probs.shape) == 1:
                                            probs = probs.unsqueeze(0)
                                    else:
                                        continue

                                # Final check: ensure probs is 2D with the correct shape
                                if len(probs.shape) != 2 or probs.shape[0] == 0 or probs.shape[1] <= domain_k:
                                    continue

                                # Safely take this domain's probability (ensure it is a scalar before calling .item())
                                try:
                                    if probs.shape[0] == 1:
                                        # if there is only one batch, take it directly
                                        domain_prob = probs[0, domain_k].item()
                                    else:
                                        # if there are multiple batches, take the mean
                                        domain_prob = probs[:, domain_k].mean().item()
                                    domain_relevances[domain_k].append(domain_prob)
                                except (IndexError, RuntimeError) as e:
                                    # if it still errors, skip
                                    continue
                            except (IndexError, RuntimeError, AttributeError) as e:
                                # silently skip indexing errors to avoid many repeated warnings
                                continue

                except Exception as e:
                    # reduce log output: only log on first occurrence to avoid many repeated warnings
                    if not hasattr(identify_foundation_layers, '_warning_logged'):
                        logger.warning(f"Error while processing domain {domain_k} layer {layer_idx}: {e} (subsequent similar errors will be silently skipped)")
                        identify_foundation_layers._warning_logged = True
                    continue

        # Compute this layer's domain relevance distribution
        # For each domain, compute the average relevance
        domain_avg_relevances = []
        for k in range(num_domains):
            if len(domain_relevances[k]) > 0:
                avg_relevance = np.mean(domain_relevances[k])
                domain_avg_relevances.append(avg_relevance)
            else:
                domain_avg_relevances.append(0.0)

        # Key insight: foundation layers should be domain-invariant layers,
        # i.e. layers whose relevance to all domains is low (not specific to any domain),
        # rather than layers whose relevance to all domains is high (relevant to all).

        # Compute the mean and std of the relevance
        avg_relevance_mean = np.mean(domain_avg_relevances)
        avg_relevance_std = np.std(domain_avg_relevances)

        # Normalize into a probability distribution (used to compute entropy, measuring uniformity)
        total_relevance = sum(domain_avg_relevances)
        if total_relevance > 0:
            domain_probs = [r / total_relevance for r in domain_avg_relevances]

            # Compute entropy (measures the uniformity of the normalized distribution)
            domain_probs_array = np.array(domain_probs)
            domain_probs_array = domain_probs_array[domain_probs_array > 0]  # only count non-zero probabilities
            if len(domain_probs_array) > 0:
                entropy = -np.sum(domain_probs_array * np.log(domain_probs_array + 1e-8))
            else:
                entropy = 0.0
        else:
            entropy = 0.0

        # Foundation-layer score: combine mean relevance and entropy.
        # Foundation layers should satisfy:
        # 1. low mean relevance (not specific to any domain)
        # 2. high entropy (after normalization the distribution is uniform, i.e. relevance to all domains is similar)
        #
        # Scoring formula: score = entropy * (1 - avg_relevance_mean)
        # This accounts for both the uniformity of the distribution (entropy) and the
        # absolute relevance value (lower relevance is better).
        foundation_score = entropy * (1.0 - avg_relevance_mean)

        # Save foundation_score and avg_relevance_mean for later filtering
        layer_entropies[layer_idx] = {
            'foundation_score': foundation_score,
            'avg_relevance_mean': avg_relevance_mean,
            'entropy': entropy
        }

        logger.info(f"Layer {layer_idx}: entropy={entropy:.4f}, mean relevance={avg_relevance_mean:.3f}, "
                   f"foundation_score={foundation_score:.4f}, "
                   f"domain relevances={[f'{r:.3f}' for r in domain_avg_relevances]}")

        # Save domain relevances for later config saving
        if not hasattr(identify_foundation_layers, '_domain_relevances'):
            identify_foundation_layers._domain_relevances = {}
        identify_foundation_layers._domain_relevances[layer_idx] = domain_avg_relevances

    # Identify foundation layers based on foundation_score.
    # Foundation layers should be domain-invariant layers:
    # - low relevance to all domains (not specific to any domain)
    # - uniform distribution after normalization (relevance to all domains is similar)
    #
    # foundation_score = entropy * (1 - avg_relevance_mean)
    # The higher the score, the more likely the layer is a foundation layer.

    # Key fix: add a mean-relevance threshold to exclude layers with overly high mean relevance.
    # Layers with mean relevance > 0.7 discriminate domains too strongly and should not be foundation layers.
    max_avg_relevance_threshold = 0.7

    # Filter candidate layers: mean relevance <= 0.7
    candidate_layers = []
    for layer_idx, metadata in layer_entropies.items():
        if isinstance(metadata, dict):
            avg_relevance_mean = metadata['avg_relevance_mean']
            foundation_score = metadata['foundation_score']
            if avg_relevance_mean <= max_avg_relevance_threshold:
                candidate_layers.append((layer_idx, foundation_score, avg_relevance_mean))
            else:
                logger.debug(f"Layer {layer_idx} excluded: mean relevance={avg_relevance_mean:.3f} > {max_avg_relevance_threshold}")
        else:
            # backward-compatible old format (if present)
            candidate_layers.append((layer_idx, metadata, 0.0))

    # Sort by foundation_score in descending order
    candidate_layers.sort(key=lambda x: x[1], reverse=True)

    # Constraint: foundation layers cannot exceed 30% of the total number of layers
    max_foundation_layers = max(1, int(len(layer_entropies) * 0.3))  # at least 1 layer, at most 30%

    if len(candidate_layers) > max_foundation_layers:
        logger.info(f"Identified {len(candidate_layers)} candidate foundation layers (exceeds the 30% cap = {max_foundation_layers} layers), "
                   f"selecting the top {max_foundation_layers} layers by foundation_score")
        foundation_layers = [layer_idx for layer_idx, _, _ in candidate_layers[:max_foundation_layers]]
    else:
        foundation_layers = [layer_idx for layer_idx, _, _ in candidate_layers]

    # Sort foundation_layers for readability
    foundation_layers.sort()

    # Print statistics for all layers (sorted by layer index)
    logger.info("=" * 80)
    logger.info("foundation_score statistics for all layers:")
    sorted_entropies = sorted(layer_entropies.items(), key=lambda x: x[0])
    for layer_idx, metadata in sorted_entropies:
        if isinstance(metadata, dict):
            foundation_score = metadata['foundation_score']
            avg_relevance_mean = metadata['avg_relevance_mean']
            entropy = metadata['entropy']
            is_foundation = "[F]" if layer_idx in foundation_layers else "   "
            excluded_reason = ""
            if avg_relevance_mean > max_avg_relevance_threshold:
                excluded_reason = f" (excluded: avg_relevance={avg_relevance_mean:.3f} > {max_avg_relevance_threshold})"
            logger.info(f"  Layer {layer_idx:2d}: foundation_score={foundation_score:.4f}, "
                       f"avg_relevance={avg_relevance_mean:.3f}, entropy={entropy:.4f} {is_foundation}{excluded_reason}")
        else:
            # backward-compatible old format
            is_foundation = "[F]" if layer_idx in foundation_layers else "   "
            logger.info(f"  Layer {layer_idx:2d}: foundation_score={metadata:.4f} {is_foundation}")
    logger.info("=" * 80)

    # Extract foundation_score for range display
    foundation_scores = []
    for metadata in layer_entropies.values():
        if isinstance(metadata, dict):
            foundation_scores.append(metadata['foundation_score'])
        else:
            foundation_scores.append(metadata)

    logger.info(f"Identified foundation layers ({len(foundation_layers)}): {foundation_layers}")
    logger.info(f"  foundation_score range: [{min(foundation_scores):.4f}, {max(foundation_scores):.4f}]")
    logger.info(f"  30% cap: at most {max_foundation_layers} layers (total layers={len(layer_entropies)})")
    logger.info(f"  mean-relevance threshold: <= {max_avg_relevance_threshold}")
    if len(candidate_layers) > max_foundation_layers:
        logger.info(f"  filtered-out layers (lower foundation_score): {[layer_idx for layer_idx, _, _ in candidate_layers[max_foundation_layers:]]}")

    # Get domain relevances data (if present)
    domain_relevances_data = getattr(identify_foundation_layers, '_domain_relevances', {})
    # Clean up temporary data
    if hasattr(identify_foundation_layers, '_domain_relevances'):
        delattr(identify_foundation_layers, '_domain_relevances')

    # For compatibility, save both foundation_score and entropy.
    # Convert layer_entropies into a compatible format (keep the dict format with all information).
    layer_entropies_compat = {}
    for layer_idx, metadata in layer_entropies.items():
        if isinstance(metadata, dict):
            layer_entropies_compat[layer_idx] = metadata
        else:
            # backward-compatible old format
            layer_entropies_compat[layer_idx] = {
                'foundation_score': metadata,
                'avg_relevance_mean': 0.0,
                'entropy': 0.0
            }

    # Return foundation layers and metadata
    if return_metadata:
        result = {
            'foundation_layers': foundation_layers,
            'layer_entropies': layer_entropies_compat,
            'layer_domain_relevances': domain_relevances_data
        }
        return result
    else:
        # backward-compatible interface: return only the list of foundation layers
        return foundation_layers


def save_foundation_layers_config(
    foundation_layers: List[int],
    layer_entropies: Dict[int, float],
    layer_domain_relevances: Dict[int, List[float]],
    output_path: Path
):
    """
    Save the foundation-layers config (including entropy and domain-relevance data).

    Args:
        foundation_layers: list of foundation layers
        layer_entropies: per-layer entropy values {layer_idx: entropy}
        layer_domain_relevances: per-layer domain relevances {layer_idx: [relevance_0, ..., relevance_k]}
        output_path: output file path
    """
    config = {
        'foundation_layers': foundation_layers,
        'layer_entropies': layer_entropies,
        'layer_domain_relevances': layer_domain_relevances,
        'num_layers': len(layer_entropies),
        'num_domains': len(list(layer_domain_relevances.values())[0]) if layer_domain_relevances else 0
    }

    import json
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Foundation layers config saved to: {output_path}")


def load_foundation_layers_config(config_path: Path) -> Optional[Dict]:
    """
    Load the foundation-layers config.

    Args:
        config_path: config file path

    Returns:
        config dict, or None if the file does not exist
    """
    import json
    if not config_path.exists():
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info(f"Loading foundation layers config from {config_path}")
    return config
