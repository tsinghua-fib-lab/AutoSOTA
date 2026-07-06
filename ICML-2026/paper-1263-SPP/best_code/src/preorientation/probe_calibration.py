"""
Probe calibration module
Implements Temperature Scaling calibration, supporting a 4-probe system
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
from pathlib import Path

from ..utils import get_logger
from .linear_probe import MultiLayerProbe

logger = get_logger(__name__)


def learn_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    device: Optional[torch.device] = None,
    max_iter: int = 100
) -> float:
    """
    Learn the optimal temperature parameter (Temperature Scaling) - global temperature

    Args:
        logits: [N, num_domains] logits output by the model
        labels: [N, num_domains] ground-truth labels (one-hot or similarity)
        device: Compute device
        max_iter: Maximum number of iterations

    Returns:
        temperature: The optimal temperature parameter (scalar)
    """
    if device is None:
        device = logits.device

    # Ensure the data is on the correct device
    logits = logits.to(device)
    labels = labels.to(device)

    # Initialize temperature
    T = torch.tensor(1.0, requires_grad=True, device=device)
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        # Calibrated probabilities (apply sigmoid independently per domain)
        calibrated_probs = torch.sigmoid(logits / T)
        # Compute BCE loss
        loss = F.binary_cross_entropy(calibrated_probs, labels)
        loss.backward()
        return loss

    # Learn the temperature
    optimizer.step(closure)

    temperature = T.item()
    logger.info(f"Learned global temperature: {temperature:.4f}")
    
    return temperature


def learn_layer_wise_temperatures(
    layer_logits: Dict[int, torch.Tensor],
    layer_labels: Dict[int, torch.Tensor],
    device: Optional[torch.device] = None,
    max_iter: int = 100
) -> Dict[int, float]:
    """
    Learn an independent temperature parameter for each layer (Layer-wise Temperature Scaling)

    Args:
        layer_logits: {layer_idx: logits [N, num_domains]} logits of each layer
        layer_labels: {layer_idx: labels [N, num_domains]} labels of each layer
        device: Compute device
        max_iter: Maximum number of iterations

    Returns:
        layer_temperatures: {layer_idx: temperature} temperature parameter of each layer
    """
    if device is None:
        device = next(iter(layer_logits.values())).device

    layer_temperatures = {}

    for layer_idx in layer_logits:
        if layer_idx not in layer_labels:
            logger.warning(f"Layer {layer_idx} has no labels, skipping")
            continue

        logits = layer_logits[layer_idx].to(device)
        labels = layer_labels[layer_idx].to(device)

        # Initialize the temperature for this layer
        T = torch.tensor(1.0, requires_grad=True, device=device)
        optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            calibrated_probs = torch.sigmoid(logits / T)
            loss = F.binary_cross_entropy(calibrated_probs, labels)
            loss.backward()
            return loss

        # Learn the temperature for this layer
        optimizer.step(closure)

        temperature = T.item()
        layer_temperatures[layer_idx] = temperature
        logger.info(f"  Layer {layer_idx} temperature: {temperature:.4f}")

    logger.info(f"Learned independent temperatures for {len(layer_temperatures)} layers")
    
    return layer_temperatures


class CalibratedProbe:
    """
    Calibrated probe wrapper
    Applies Temperature Scaling on top of the original probe
    """
    
    def __init__(
        self, 
        base_probe: MultiLayerProbe, 
        temperature: float = 1.0,
        layer_temperatures: Optional[Dict[int, float]] = None
    ):
        """
        Initialize the calibrated probe

        Args:
            base_probe: Base probe (MultiLayerProbe)
            temperature: Global temperature parameter (used if layer_temperatures is None)
            layer_temperatures: Per-layer independent temperature parameters {layer_idx: temperature}
                                If provided, layer_temperatures takes precedence
        """
        self.base_probe = base_probe
        self.temperature = temperature
        self.layer_temperatures = layer_temperatures

        if layer_temperatures is not None:
            logger.info(f"CalibratedProbe using per-layer temperatures: {len(layer_temperatures)} layers")
        else:
            logger.info(f"CalibratedProbe using global temperature: {temperature:.4f}")

    def forward(self, layer_activations: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Forward pass (applies Temperature Scaling)

        Args:
            layer_activations: {layer_idx: activations [batch, dim]}

        Returns:
            calibrated_probs: {layer_idx: calibrated_probs [batch, num_domains]}
        """
        # Get the original logits
        logits_dict = {}
        for layer_idx, activations in layer_activations.items():
            if layer_idx in self.base_probe.probes:
                probe = self.base_probe.probes[layer_idx]
                logits = probe(activations)
                logits_dict[layer_idx] = logits

        # Apply Temperature Scaling
        calibrated_probs = {}
        for layer_idx, logits in logits_dict.items():
            # If a per-layer temperature is provided, use it; otherwise use the global temperature
            if self.layer_temperatures is not None and layer_idx in self.layer_temperatures:
                T = self.layer_temperatures[layer_idx]
            else:
                T = self.temperature
            
            calibrated_probs[layer_idx] = torch.sigmoid(logits / T)
        
        return calibrated_probs
    
    def get_layer_domain_importance(self, layer_activations: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Get the layer-domain importance (applies Temperature Scaling)

        Args:
            layer_activations: {layer_idx: activations [batch, dim]}

        Returns:
            importance: {layer_idx: importance [batch, num_domains]}
        """
        return self.forward(layer_activations)


class MultiProbeSystem:
    """
    Manager for the 4-probe system
    """
    
    def __init__(
        self,
        base_probe: MultiLayerProbe,
        probe2_temperature: float = 1.0,  # Single-domain
        probe3_temperature: float = 1.0,  # OOD
        probe4_temperature: float = 1.0,  # Cross-domain
        probe2_layer_temperatures: Optional[Dict[int, float]] = None,  # Single-domain per-layer temperature
        probe3_layer_temperatures: Optional[Dict[int, float]] = None,  # OOD per-layer temperature
        probe4_layer_temperatures: Optional[Dict[int, float]] = None   # Cross-domain per-layer temperature
    ):
        """
        Initialize the 4-probe system

        Args:
            base_probe: Base probe (probe 1)
            probe2_temperature: Global temperature of probe 2 (Single-domain, used if probe2_layer_temperatures is None)
            probe3_temperature: Global temperature of probe 3 (OOD, used if probe3_layer_temperatures is None)
            probe4_temperature: Global temperature of probe 4 (Cross-domain, used if probe4_layer_temperatures is None)
            probe2_layer_temperatures: Per-layer temperatures of probe 2 (Single-domain)
            probe3_layer_temperatures: Per-layer temperatures of probe 3 (OOD)
            probe4_layer_temperatures: Per-layer temperatures of probe 4 (Cross-domain)
        """
        self.probe1 = base_probe  # Original probe (uncalibrated)
        self.probe2 = CalibratedProbe(base_probe, temperature=probe2_temperature, layer_temperatures=probe2_layer_temperatures)  # Single-domain
        self.probe3 = CalibratedProbe(base_probe, temperature=probe3_temperature, layer_temperatures=probe3_layer_temperatures)  # OOD
        self.probe4 = CalibratedProbe(base_probe, temperature=probe4_temperature, layer_temperatures=probe4_layer_temperatures)  # Cross-domain

        logger.info(f"4-probe system initialized:")
        logger.info(f"  Probe 1 (original): temperature=1.0 (uncalibrated)")
        if probe2_layer_temperatures is not None:
            logger.info(f"  Probe 2 (Single-domain): per-layer temperature ({len(probe2_layer_temperatures)} layers)")
            logger.info(f"    Mean temperature: {np.mean(list(probe2_layer_temperatures.values())):.4f}")
        else:
            logger.info(f"  Probe 2 (Single-domain): global temperature={probe2_temperature:.4f}")
        if probe3_layer_temperatures is not None:
            logger.info(f"  Probe 3 (OOD): per-layer temperature ({len(probe3_layer_temperatures)} layers)")
            logger.info(f"    Mean temperature: {np.mean(list(probe3_layer_temperatures.values())):.4f}")
        else:
            logger.info(f"  Probe 3 (OOD): global temperature={probe3_temperature:.4f}")
        if probe4_layer_temperatures is not None:
            logger.info(f"  Probe 4 (Cross-domain): per-layer temperature ({len(probe4_layer_temperatures)} layers)")
            logger.info(f"    Mean temperature: {np.mean(list(probe4_layer_temperatures.values())):.4f}")
        else:
            logger.info(f"  Probe 4 (Cross-domain): global temperature={probe4_temperature:.4f}")
    
    def get_probe(self, scenario: str = "default"):
        """
        Get the corresponding probe based on the scenario

        Args:
            scenario: "single-domain", "ood", "cross-domain", or "default"

        Returns:
            probe: The corresponding probe
        """
        if scenario == "single-domain":
            return self.probe2
        elif scenario == "ood":
            return self.probe3
        elif scenario == "cross-domain":
            return self.probe4
        else:
            return self.probe1  # Use the original probe by default
    
    def save(self, output_dir: Path):
        """
        Save the 4-probe system

        Args:
            output_dir: Output directory
        """
        # Save the base probe (probe 1)
        # If final_probes.pt already exists, rename/link it directly to avoid saving twice
        final_probes_path = output_dir / "final_probes.pt"
        probe1_path = output_dir / "probe1_base.pt"

        if final_probes_path.exists() and not probe1_path.exists():
            # If final_probes.pt exists and probe1_base.pt does not, copy it directly
            import shutil
            shutil.copytree(str(final_probes_path), str(probe1_path), dirs_exist_ok=True)
            logger.info(f"Copied from final_probes.pt to probe1_base.pt (avoids saving twice)")
        else:
            # Otherwise save normally
            self.probe1.save(str(probe1_path))

        # Save the temperature parameters
        temperatures = {
            "probe1": 1.0,
            "probe2_single_domain": self.probe2.temperature,
            "probe3_ood": self.probe3.temperature,
            "probe4_cross_domain": self.probe4.temperature
        }
        
        # Save the per-layer temperature parameters (if present)
        if self.probe2.layer_temperatures is not None:
            temperatures["probe2_single_domain_layer_wise"] = self.probe2.layer_temperatures
        if self.probe3.layer_temperatures is not None:
            temperatures["probe3_ood_layer_wise"] = self.probe3.layer_temperatures
        if self.probe4.layer_temperatures is not None:
            temperatures["probe4_cross_domain_layer_wise"] = self.probe4.layer_temperatures
        
        temp_path = output_dir / "probe_temperatures.json"
        with open(temp_path, 'w') as f:
            json.dump(temperatures, f, indent=2)
        
        logger.info(f"4-probe system saved to: {output_dir}")

    @classmethod
    def load(cls, output_dir: Path, base_probe: MultiLayerProbe):
        """
        Load the 4-probe system

        Args:
            output_dir: Output directory
            base_probe: Base probe (must be loaded first)

        Returns:
            multi_probe_system: The 4-probe system
        """
        # Load the temperature parameters
        temp_path = output_dir / "probe_temperatures.json"
        with open(temp_path, 'r') as f:
            temperatures = json.load(f)
        
        return cls(
            base_probe=base_probe,
            probe2_temperature=temperatures.get("probe2_single_domain", 1.0),
            probe3_temperature=temperatures.get("probe3_ood", 1.0),
            probe4_temperature=temperatures.get("probe4_cross_domain", 1.0),
            probe2_layer_temperatures=temperatures.get("probe2_single_domain_layer_wise", None),
            probe3_layer_temperatures=temperatures.get("probe3_ood_layer_wise", None),
            probe4_layer_temperatures=temperatures.get("probe4_cross_domain_layer_wise", None)
        )


def load_ood_avg_similarity(
    ood_domain_name: str,
    selected_domains: List[str],
    domain_selection_file: str = "outputs/domain_selection_v2/selected_domains.json"
) -> np.ndarray:
    """
    Load the average similarity of an OOD domain from domain_selection_v2

    Note: the similarity data may already be stored directly in the file, or it may need to be computed
    from the orthogonality matrix. If direct similarity data is available in the file, it is preferred;
    otherwise use 1 - orthogonality.

    Args:
        ood_domain_name: OOD domain name (e.g., "economics")
        selected_domains: List of selected domains
        domain_selection_file: Path to the domain_selection_v2 file

    Returns:
        avg_similarity: [num_selected_domains] average similarity vector (not normalized, keeps original values)
    """
    with open(domain_selection_file, 'r') as f:
        data = json.load(f)

    all_domains = data['selection_info']['all_domains']
    orthogonal_matrix = np.array(data['selection_info']['all_orthogonal_matrix'])

    # Compute similarity from the orthogonality matrix (similarity = 1 - orthogonality)
    ood_idx = all_domains.index(ood_domain_name)
    selected_indices = [all_domains.index(d) for d in selected_domains]

    # Get the orthogonalities, then compute the similarities
    orthogonalities = orthogonal_matrix[ood_idx, selected_indices]
    similarities = 1 - orthogonalities

    # Enhance similarity contrast: use softmax temperature scaling to make similar domains more prominent
    # The smaller the temperature, the sharper the distribution (high similarities higher, low similarities lower)
    temperature = 0.3  # Tunable parameter; smaller is sharper
    enhanced_similarities = enhance_similarity_with_temperature(similarities, temperature)

    logger.info(f"Average similarity of OOD domain '{ood_domain_name}' to the selected domains (after enhancement):")
    for i, domain in enumerate(selected_domains):
        logger.info(f"  {domain}: {similarities[i]:.4f} -> {enhanced_similarities[i]:.4f}")

    logger.info(f"  Sum: {enhanced_similarities.sum():.4f}")
    
    return enhanced_similarities


def enhance_similarity_with_temperature(
    similarities: np.ndarray,
    temperature: float = 0.3
) -> np.ndarray:
    """
    Enhance similarity contrast using softmax temperature scaling

    Makes similar domains more prominent and dissimilar ones weaker.
    The smaller the temperature, the sharper the distribution (high similarities higher, low similarities lower).

    Note: the enhanced similarities are clamped to the [0, 1] range.

    Args:
        similarities: Original similarity vector [num_domains] (should be in the [0, 1] range)
        temperature: Temperature parameter (smaller is sharper, recommended 0.2-0.5)

    Returns:
        enhanced_similarities: Enhanced similarity vector [num_domains] (clamped to [0, 1])
    """
    # Use softmax temperature scaling
    exp_sim = np.exp(similarities / temperature)
    normalized = exp_sim / exp_sim.sum()

    # Preserve the original sum (do not change the overall similarity level)
    enhanced = normalized * similarities.sum()

    # Ensure values do not exceed 1.0 (similarities should be in the [0, 1] range)
    enhanced = np.clip(enhanced, 0.0, 1.0)
    
    return enhanced


def load_all_domain_to_selected_mapping(
    selected_domains: List[str],
    domain_selection_file: str = "outputs/domain_selection_v2/selected_domains.json",
    enhance: bool = True,
    temperature: float = 0.3
) -> Dict[str, np.ndarray]:
    """
    Load the similarity mapping from all domains to the selected domains (computed from all_orthogonal_matrix)

    Note: similarity = 1 - orthogonality
    Optional: use temperature scaling to enhance contrast

    Args:
        selected_domains: List of selected domains
        domain_selection_file: Path to the domain_selection_v2 file
        enhance: Whether to enhance similarity contrast (default True)
        temperature: Temperature parameter (smaller is sharper, recommended 0.2-0.5)

    Returns:
        domain_similarities: {domain_name: similarity_vector [num_selected_domains]} (not normalized)
            Includes all domains (both selected domains and OOD domains)
    """
    with open(domain_selection_file, 'r') as f:
        data = json.load(f)

    all_domains = data['selection_info']['all_domains']

    # Compute similarity from the orthogonality matrix (similarity = 1 - orthogonality)
    orthogonal_matrix = np.array(data['selection_info']['all_orthogonal_matrix'])
    selected_indices = [all_domains.index(d) for d in selected_domains]

    domain_similarities = {}
    for domain in all_domains:
        domain_idx = all_domains.index(domain)
        # Get the orthogonalities, then compute the similarities
        orthogonalities = orthogonal_matrix[domain_idx, selected_indices]
        similarities = 1 - orthogonalities

        # Enhance similarity contrast (make similar domains more prominent and dissimilar ones weaker)
        if enhance:
            similarities = enhance_similarity_with_temperature(similarities, temperature)

        domain_similarities[domain] = similarities

    logger.info(f"Computed similarity mapping from {len(domain_similarities)} domains to {len(selected_domains)} selected domains from the orthogonality matrix ({'enhanced' if enhance else 'not enhanced'})")
    
    return domain_similarities


def load_all_ood_avg_similarities(
    selected_domains: List[str],
    domain_selection_file: str = "outputs/domain_selection_v2/selected_domains.json",
    enhance: bool = True,
    temperature: float = 0.3
) -> Dict[str, np.ndarray]:
    """
    Load the average similarities of all OOD domains (computed from all_orthogonal_matrix, not normalized)

    Note: similarity = 1 - orthogonality
    Optional: use temperature scaling to enhance contrast

    Args:
        selected_domains: List of selected domains
        domain_selection_file: Path to the domain_selection_v2 file
        enhance: Whether to enhance similarity contrast (default True)
        temperature: Temperature parameter (smaller is sharper, recommended 0.2-0.5)

    Returns:
        ood_similarities: {ood_domain_name: similarity_vector [num_selected_domains]} (not normalized)
    """
    # Use the full mapping function, then filter out the OOD domains
    all_mapping = load_all_domain_to_selected_mapping(
        selected_domains,
        domain_selection_file,
        enhance=enhance,
        temperature=temperature
    )

    # Filter out the OOD domains (those not in selected)
    ood_similarities = {
        domain: similarities
        for domain, similarities in all_mapping.items()
        if domain not in selected_domains
    }

    logger.info(f"Extracted similarities for {len(ood_similarities)} OOD domains ({'enhanced' if enhance else 'not enhanced'})")
    
    return ood_similarities

