"""
Offline stage of Probe-based Scenario Pruning (PSP).

This script runs the offline phase of PSP. It:
  1. Trains lightweight per-layer linear probes that map a layer's residual
     representation onto the domain subspace.
  2. Calibrates those probes via temperature scaling (producing a 4-probe
     system: base / single-domain / OOD / cross-domain).
  3. Computes the axis-aligned head importance I_{l,h,k} for each layer l,
     head h and domain k.
  4. Identifies the domain-invariant head whitelist.

IMPORTANT: There is NO model fine-tuning or training of the base model. The
base model is kept frozen throughout; only the lightweight probes are trained
and calibrated.

Outputs are written to outputs/<model_name>/ppd_pipeline/.
"""

import sys
import json
import math
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.base_model import BaseModel
from src.preorientation.domain_axes import create_domain_axes_onehot
from src.preorientation.linear_probe import MultiLayerProbe
from src.preorientation.probe_calibration import (
    learn_temperature,
    learn_layer_wise_temperatures,
    MultiProbeSystem,
    load_ood_avg_similarity,
    load_all_ood_avg_similarities,
)
from src.probe.head_importance import HeadImportanceCalculator
from src.probe.whitelist_identification import HeadWhitelistIdentifier
from src.utils import setup_logger, get_logger

logger = get_logger(__name__)


STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'what', 'which', 'who', 'when', 'where', 'why', 'how', 'if', 'then', 'than', 'so',
    'because', 'although', 'though', 'while', 'until', 'since', 'during', 'before', 'after'
}


def load_domain_data(
    data_dir: Path,
    selected_domains: List[str],
    base_model: BaseModel,
    num_samples_per_domain: int = 100,
    remove_stopwords: bool = True
) -> Dict[int, List[Dict[str, torch.Tensor]]]:
    """
    Load domain data.

    Following the paper design, common words (stopwords) must be removed before
    computing embeddings and training probes.

    Args:
        data_dir: Data directory.
        selected_domains: List of selected domains.
        base_model: Base model (used for tokenization).
        num_samples_per_domain: Number of samples per domain.
        remove_stopwords: Whether to remove common words (per the paper, probes
            and PPD require stopword removal).

    Returns:
        domain_data: {domain_k: [inputs]}
    """
    import re

    domain_data = {}

    logger.info("=" * 80)
    logger.info("Loading domain data")
    logger.info(f"  Data directory: {data_dir}")
    logger.info(f"  Selected domains: {selected_domains}")
    logger.info(f"  Remove stopwords: {remove_stopwords}")
    logger.info("=" * 80)

    for k, domain_name in enumerate(selected_domains):
        # Try multiple file name formats
        possible_files = [
            data_dir / f"{domain_name}.json",
            data_dir / f"{domain_name}_data.json",
            data_dir / f"{domain_name}_train.json"
        ]

        domain_file = None
        for f in possible_files:
            if f.exists():
                domain_file = f
                break

        if domain_file is None:
            raise FileNotFoundError(
                f"Data file for domain {domain_name} does not exist. "
                f"Tried: {[str(f) for f in possible_files]}"
            )

        logger.info(f"Loading data for domain {domain_name}: {domain_file}")

        with open(domain_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle data format: could be a list or a dict
        if isinstance(data, dict):
            # If it is a dict, try to locate the data list
            if 'data' in data:
                data = data['data']
            elif 'samples' in data:
                data = data['samples']
            else:
                # Otherwise treat the dict values as the data
                data = list(data.values())

        if not isinstance(data, list):
            raise ValueError(f"Invalid data format, expected list but got {type(data)}")

        # Convert into model input format
        inputs_list = []
        tokenizer = base_model.tokenizer

        for idx, sample in enumerate(data[:num_samples_per_domain]):
            try:
                # Handle different data formats
                if isinstance(sample, str):
                    text = sample
                elif isinstance(sample, dict):
                    # Try several possible field names
                    text = sample.get('text') or sample.get('question') or sample.get('prompt') or sample.get('input')
                    if not text:
                        logger.warning(f"Sample {idx} is missing a text field, skipping")
                        continue
                else:
                    logger.warning(f"Sample {idx} has an invalid format, skipping")
                    continue

                # Remove common words (per the paper design)
                if remove_stopwords:
                    words = re.findall(r'\b\w+\b', text.lower())
                    filtered_words = [w for w in words if w not in STOPWORDS and len(w) >= 3]
                    text = ' '.join(filtered_words)
                    if not text:
                        logger.warning(f"Sample {idx} became empty after stopword removal, using original text")
                        text = sample if isinstance(sample, str) else (sample.get('text') or sample.get('question') or '')

                # Tokenize
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding='max_length'
                )

                inputs = {
                    'input_ids': encoded['input_ids'].squeeze(0),
                    'attention_mask': encoded['attention_mask'].squeeze(0)
                }

                inputs_list.append(inputs)

            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}, skipping")
                continue

        if len(inputs_list) == 0:
            raise ValueError(f"Domain {domain_name} has no valid samples")

        domain_data[k] = inputs_list
        logger.info(f"  Domain {domain_name}: loaded {len(inputs_list)} samples")

    total_samples = sum(len(v) for v in domain_data.values())
    logger.info(f"Loaded {total_samples} samples in total, covering {len(domain_data)} domains")

    return domain_data


def step4_final_probe_training(
    ppd_model: BaseModel,
    train_domain_data: Dict[int, List[Dict[str, torch.Tensor]]],
    val_domain_data: Dict[int, List[Dict[str, torch.Tensor]]],
    selected_domains: List[str],
    output_dir: Path,
    num_epochs: int = 20
) -> MultiLayerProbe:
    """
    Step 4: Final probe training (on the model, which may be the PPD model or
    the original model).

    Args:
        ppd_model: Model (may be the PPD model or the original model).
        train_domain_data: Training-set domain data.
        val_domain_data: Validation-set domain data.
        selected_domains: List of selected domains.
        output_dir: Output directory.
        num_epochs: Number of training epochs.

    Returns:
        final_layer_probes: The final trained probes.
    """
    logger.info("=" * 80)
    logger.info("Step 4: Final probe training")
    logger.info("=" * 80)

    # Check whether the 4-probe system already exists (if so, training and
    # calibration are already complete)
    probe1_path = output_dir / "probe1_base.pt"
    temp_path = output_dir / "probe_temperatures.json"

    if probe1_path.exists() and temp_path.exists():
        logger.info("Found an existing 4-probe system (probe1_base.pt + probe_temperatures.json)")
        logger.info("   This means final probe training and calibration are complete, skipping the training step")
        logger.info("   Will load the saved probes directly")

        # Load the saved probes
        final_layer_probes = MultiLayerProbe.load(str(probe1_path), device=ppd_model.device)
        logger.info(f"Probes loaded successfully: {len(final_layer_probes.probes)} layers")
        logger.info("=" * 80)
        return final_layer_probes

    # Check whether the final (uncalibrated) probes already exist
    final_probe_path = output_dir / "final_probes.pt"
    if final_probe_path.exists():
        logger.info("Found existing final probes (final_probes.pt)")
        logger.info("   Will load them directly, skipping the training step")

        final_layer_probes = MultiLayerProbe.load(str(final_probe_path), device=ppd_model.device)
        logger.info(f"Probes loaded successfully: {len(final_layer_probes.probes)} layers")
        logger.info("=" * 80)
        return final_layer_probes

    # Create the probes
    num_layers = ppd_model.model.config.num_hidden_layers
    num_domains = len(selected_domains)
    hidden_dim = ppd_model.model.config.hidden_size

    # Create the activation_dims dict
    activation_dims = {i: hidden_dim for i in range(num_layers)}

    final_layer_probes = MultiLayerProbe(
        num_layers=num_layers,
        activation_dims=activation_dims,
        num_domains=num_domains,
        hidden_dim=256,
        nonlinear=False
    )

    # Extract training-set activations
    logger.info("Starting final probe training...")
    logger.info("  Extracting training-set activations (post-attention residual)...")

    device = ppd_model.device  # Use the device of ppd_model
    train_layer_activations = {}
    train_domain_labels_list = []

    # Count total number of samples
    total_train_samples = sum(len(inputs_list) for inputs_list in train_domain_data.values())
    processed_train_samples = 0
    logger.info(f"  Need to process {total_train_samples} training samples in total...")

    for domain_k, inputs_list in train_domain_data.items():
        logger.info(f"  Processing training-set domain {domain_k}: {len(inputs_list)} samples...")
        for sample_idx, inputs in enumerate(inputs_list):
            try:
                # Ensure inputs have a batch dimension
                batch_inputs = {}
                for k, v in inputs.items():
                    if v.dim() == 1:
                        batch_inputs[k] = v.unsqueeze(0)  # [seq_len] -> [1, seq_len]
                    else:
                        batch_inputs[k] = v

                activations = ppd_model.get_activations(
                    batch_inputs,
                    layer_indices=list(range(num_layers)),
                    extract_post_attn_residual=True
                )

                for layer_idx in range(num_layers):
                    if layer_idx in activations:
                        activation = activations[layer_idx]
                        if len(activation.shape) == 3:
                            activation = activation.mean(dim=1)
                        elif len(activation.shape) == 2:
                            pass
                        elif len(activation.shape) == 1:
                            activation = activation.unsqueeze(0)
                        else:
                            activation = activation.view(1, -1)

                        if layer_idx not in train_layer_activations:
                            train_layer_activations[layer_idx] = []
                        train_layer_activations[layer_idx].append(activation)

                train_domain_labels_list.append(domain_k)
                processed_train_samples += 1
                if processed_train_samples % 50 == 0 or processed_train_samples == total_train_samples:
                    logger.info(f"     Processed {processed_train_samples}/{total_train_samples} training samples...")
            except Exception as e:
                logger.warning(f"Error processing training sample: {e}, skipping")
                continue

    for layer_idx in train_layer_activations:
        train_layer_activations[layer_idx] = torch.cat(train_layer_activations[layer_idx], dim=0)

    train_domain_labels = torch.tensor(train_domain_labels_list, dtype=torch.long)

    # Extract validation-set activations
    logger.info("  Extracting validation-set activations (for evaluating overfitting)...")
    val_layer_activations = {}
    val_domain_labels_list = []

    # Count total number of samples
    total_val_samples = sum(len(inputs_list) for inputs_list in val_domain_data.values())
    processed_val_samples = 0
    logger.info(f"  Need to process {total_val_samples} validation samples in total...")

    for domain_k, inputs_list in val_domain_data.items():
        logger.info(f"  Processing validation-set domain {domain_k}: {len(inputs_list)} samples...")
        for sample_idx, inputs in enumerate(inputs_list):
            try:
                # Ensure inputs have a batch dimension
                batch_inputs = {}
                for k, v in inputs.items():
                    if v.dim() == 1:
                        batch_inputs[k] = v.unsqueeze(0)
                    else:
                        batch_inputs[k] = v

                activations = ppd_model.get_activations(
                    batch_inputs,
                    layer_indices=list(range(num_layers)),
                    extract_post_attn_residual=True
                )

                for layer_idx in range(num_layers):
                    if layer_idx in activations:
                        activation = activations[layer_idx]
                        if len(activation.shape) == 3:
                            activation = activation.mean(dim=1)
                        elif len(activation.shape) == 2:
                            pass
                        elif len(activation.shape) == 1:
                            activation = activation.unsqueeze(0)
                        else:
                            activation = activation.view(1, -1)

                        if layer_idx not in val_layer_activations:
                            val_layer_activations[layer_idx] = []
                        val_layer_activations[layer_idx].append(activation)

                val_domain_labels_list.append(domain_k)
            except Exception as e:
                logger.warning(f"Error processing validation sample: {e}, skipping")
                continue

    for layer_idx in val_layer_activations:
        val_layer_activations[layer_idx] = torch.cat(val_layer_activations[layer_idx], dim=0)

    val_domain_labels = torch.tensor(val_domain_labels_list, dtype=torch.long)

    logger.info(f"Activation extraction complete")
    logger.info(f"  Number of training samples: {len(train_domain_labels)}")
    logger.info(f"  Number of validation samples: {len(val_domain_labels)}")
    logger.info(f"  Number of epochs: {num_epochs}")
    logger.info(f"  Note: each epoch is evaluated on the validation set to observe overfitting")
    logger.info("=" * 80)

    # Train the probes (with validation-set evaluation)
    logger.info("Starting probe training (each layer trained independently)...")
    final_layer_probes.train_all_layers(
        layer_activations=train_layer_activations,
        domain_labels=train_domain_labels,
        num_epochs=num_epochs,
        batch_size=32,
        learning_rate=1e-3,
        device=device,
        val_layer_activations=val_layer_activations,
        val_domain_labels=val_domain_labels
    )

    # Save the probes
    probe_path = output_dir / "final_probes.pt"
    final_layer_probes.save(str(probe_path))
    logger.info(f"Final probes saved to: {probe_path}")

    return final_layer_probes


def step4_5_calibrate_probes(
    base_probe: MultiLayerProbe,
    ppd_model: BaseModel,
    train_domain_data: Dict[int, List[Dict[str, torch.Tensor]]],
    val_domain_data: Dict[int, List[Dict[str, torch.Tensor]]],
    selected_domains: List[str],
    output_dir: Path,
    ood_domain_name: Optional[str] = "economics",
    foundation_layers: Optional[List[int]] = None
) -> MultiProbeSystem:
    """
    Step 4.5: Calibrate the probes (4-probe system).

    Args:
        base_probe: Base probe (probe 1).
        ppd_model: Model.
        train_domain_data: Training-set domain data.
        val_domain_data: Validation-set domain data.
        selected_domains: List of selected domains.
        output_dir: Output directory.
        ood_domain_name: OOD domain name (e.g. "economics").

    Returns:
        multi_probe_system: The 4-probe system.
    """
    logger.info("Starting probe calibration (4-probe system)...")

    # Check whether the 4-probe system already exists
    probe1_path = output_dir / "probe1_base.pt"
    temp_path = output_dir / "probe_temperatures.json"

    if probe1_path.exists() and temp_path.exists():
        logger.info("Found an existing 4-probe system (probe1_base.pt + probe_temperatures.json)")
        logger.info("   This means calibration is complete, will load it directly")

        multi_probe_system = MultiProbeSystem.load(output_dir, base_probe)
        logger.info("4-probe system loaded successfully")
        logger.info("=" * 80)
        return multi_probe_system

    device = ppd_model.device

    # Set foundation layers (default to layers 0-2 if not provided)
    if foundation_layers is None:
        foundation_layers = [0, 1, 2]
        logger.info(f"  foundation_layers not provided, defaulting to skipping layers {foundation_layers}")
    else:
        logger.info(f"  Skipping calibration of foundation layers {foundation_layers}")

    # 4.5.1: Single-domain calibration (probe 2)
    logger.info("4.5.1: Single-domain calibration (probe 2)...")
    T_single, T_single_layer_wise = calibrate_probe_for_single_domain(
        base_probe, ppd_model, val_domain_data, selected_domains, device, foundation_layers
    )

    # 4.5.2: OOD calibration (probe 3)
    # Calibrate using the average similarity over all OOD domains
    T_ood = 1.0
    T_ood_layer_wise = None
    logger.info("4.5.2: OOD calibration (probe 3, using all OOD domains)...")
    try:
        all_ood_similarities = load_all_ood_avg_similarities(selected_domains)
        logger.info(f"  Loaded similarity data for {len(all_ood_similarities)} OOD domains")

        # Calibrate for each OOD domain separately, then take the average temperature
        if all_ood_similarities:
            logger.info(f"  Will calibrate each of the {len(all_ood_similarities)} OOD domains separately...")
            for ood_name, sim_vec in list(all_ood_similarities.items())[:3]:  # Only show the first 3
                logger.info(f"    {ood_name}: {sim_vec}")

            T_ood, T_ood_layer_wise = calibrate_probe_for_ood_multi(
                base_probe, ppd_model, val_domain_data, all_ood_similarities, selected_domains, device, foundation_layers
            )
        else:
            logger.warning("  No OOD domains found, using default temperature=1.0")
            T_ood = 1.0
            T_ood_layer_wise = None
    except Exception as e:
        logger.warning(f"OOD calibration failed: {e}, using default temperature=1.0")
        T_ood = 1.0
        T_ood_layer_wise = None

    # 4.5.3: Cross-domain calibration (probe 4)
    logger.info("4.5.3: Cross-domain calibration (probe 4)...")
    # Cross-domain uses the average temperature of Single-domain and OOD (per layer)
    T_cross, T_cross_layer_wise = calibrate_probe_for_cross_domain(
        base_probe, ppd_model, val_domain_data, None,
        selected_domains, device, T_single, T_ood,
        T_single_layer_wise, T_ood_layer_wise, foundation_layers
    )

    # Create the 4-probe system (using per-layer independent temperatures)
    multi_probe_system = MultiProbeSystem(
        base_probe=base_probe,
        probe2_temperature=T_single,
        probe3_temperature=T_ood,
        probe4_temperature=T_cross,
        probe2_layer_temperatures=T_single_layer_wise,
        probe3_layer_temperatures=T_ood_layer_wise,
        probe4_layer_temperatures=T_cross_layer_wise
    )

    # Save the 4-probe system
    multi_probe_system.save(output_dir)

    logger.info("4-probe system calibration complete and saved")

    return multi_probe_system


def calibrate_probe_for_single_domain(
    base_probe: MultiLayerProbe,
    ppd_model: BaseModel,
    val_domain_data: Dict[int, List[Dict[str, torch.Tensor]]],
    selected_domains: List[str],
    device: torch.device,
    foundation_layers: Optional[List[int]] = None
) -> Tuple[float, Optional[Dict[int, float]]]:
    """Calibrate the probe for the Single-domain scenario."""
    logger.info("  Extracting validation-set activations...")

    val_layer_activations = {}
    val_domain_labels_list = []

    for domain_k, inputs_list in val_domain_data.items():
        for inputs in inputs_list:
            try:
                batch_inputs = {}
                for k, v in inputs.items():
                    if v.dim() == 1:
                        batch_inputs[k] = v.unsqueeze(0)
                    else:
                        batch_inputs[k] = v

                activations = ppd_model.get_activations(
                    batch_inputs,
                    layer_indices=list(range(ppd_model.model.config.num_hidden_layers)),
                    extract_post_attn_residual=True
                )

                for layer_idx in range(ppd_model.model.config.num_hidden_layers):
                    if layer_idx in activations:
                        activation = activations[layer_idx]
                        if len(activation.shape) == 3:
                            activation = activation.mean(dim=1)
                        elif len(activation.shape) == 1:
                            activation = activation.unsqueeze(0)

                        if layer_idx not in val_layer_activations:
                            val_layer_activations[layer_idx] = []
                        val_layer_activations[layer_idx].append(activation)

                val_domain_labels_list.append(domain_k)
            except Exception as e:
                logger.warning(f"Error processing validation sample: {e}, skipping")
                continue

    for layer_idx in val_layer_activations:
        val_layer_activations[layer_idx] = torch.cat(val_layer_activations[layer_idx], dim=0)

    val_domain_labels = torch.tensor(val_domain_labels_list, dtype=torch.long)
    logger.info(f"  Number of validation samples: {len(val_domain_labels)}")

    # Set foundation layers (default to layers 0-2 if not provided)
    if foundation_layers is None:
        foundation_layers = [0, 1, 2]

    # Collect per-layer logits and labels (for per-layer temperatures, skipping foundation layers)
    layer_logits = {}
    layer_labels = {}

    for layer_idx in range(ppd_model.model.config.num_hidden_layers):
        # Skip foundation layers (these layers do not need calibration)
        if layer_idx in foundation_layers:
            continue

        if layer_idx in val_layer_activations and layer_idx in base_probe.probes:
            activations = val_layer_activations[layer_idx].to(device)
            probe = base_probe.probes[layer_idx]
            probe.to(device)
            probe.eval()

            with torch.no_grad():
                logits = probe(activations)
                layer_logits[layer_idx] = logits
                binary_labels = torch.zeros_like(logits)
                binary_labels.scatter_(1, val_domain_labels.unsqueeze(1).to(device), 1.0)
                layer_labels[layer_idx] = binary_labels

    if not layer_logits:
        logger.warning("No valid logits, using default temperature=1.0")
        return 1.0, None

    # Learn per-layer independent temperatures
    logger.info("  Learning per-layer independent temperature parameters...")
    layer_temperatures = learn_layer_wise_temperatures(layer_logits, layer_labels, device)

    # Compute the average temperature (for compatibility)
    avg_temperature = np.mean(list(layer_temperatures.values()))
    logger.info(f"  Single-domain average temperature: {avg_temperature:.4f} (from {len(layer_temperatures)} layers)")

    return avg_temperature, layer_temperatures


def calibrate_probe_for_ood_multi(
    base_probe: MultiLayerProbe,
    ppd_model: BaseModel,
    val_domain_data: Dict[int, List[Dict[str, torch.Tensor]]],
    all_ood_similarities: Dict[str, np.ndarray],
    selected_domains: List[str],
    device: torch.device,
    foundation_layers: Optional[List[int]] = None
) -> Tuple[float, Optional[Dict[int, float]]]:
    """
    Calibrate the probe for the OOD scenario (using all OOD domains).

    Calibrate each OOD domain separately, then take the average temperature
    (similar to the Single-domain calibration approach).
    """
    logger.info(f"  Calibrating each of the {len(all_ood_similarities)} OOD domains separately, then averaging...")

    # Set foundation layers (default to layers 0-2 if not provided)
    if foundation_layers is None:
        foundation_layers = [0, 1, 2]

    # Extract validation-set activations (val data of all selected domains)
    logger.info("  Extracting validation-set activations for OOD calibration...")
    val_layer_activations = {}
    for domain_k, inputs_list in val_domain_data.items():
        for inputs in inputs_list:
            try:
                batch_inputs = {k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in inputs.items()}
                activations = ppd_model.get_activations(
                    batch_inputs,
                    layer_indices=list(range(ppd_model.model.config.num_hidden_layers)),
                    extract_post_attn_residual=True
                )
                for layer_idx in activations:
                    activation = activations[layer_idx]
                    if len(activation.shape) == 3:
                        activation = activation.mean(dim=1)
                    elif len(activation.shape) == 1:
                        activation = activation.unsqueeze(0)
                    if layer_idx not in val_layer_activations:
                        val_layer_activations[layer_idx] = []
                    val_layer_activations[layer_idx].append(activation)
            except Exception as e:
                logger.warning(f"Error processing validation sample: {e}, skipping")
                continue

    for layer_idx in val_layer_activations:
        val_layer_activations[layer_idx] = torch.cat(val_layer_activations[layer_idx], dim=0)

    if not val_layer_activations:
        logger.warning("  No validation-set activations, using default temperature=1.0")
        return 1.0, None

    # Calibrate each OOD domain separately (per-layer, similar to Single-domain)
    ood_temperatures = []
    layer_temps_list = []  # Stores the per-layer temperatures for each OOD domain

    for ood_domain_name, ood_similarity in all_ood_similarities.items():
        logger.info(f"  Calibrating OOD domain '{ood_domain_name}' (similarity: {ood_similarity})...")

        # Collect per-layer logits and labels (for per-layer temperatures, skipping foundation layers)
        layer_logits = {}
        layer_labels = {}

        for layer_idx in val_layer_activations:
            # Skip foundation layers (these layers do not need calibration)
            if layer_idx in foundation_layers:
                continue

            if layer_idx in base_probe.probes:
                activations = val_layer_activations[layer_idx].to(device)
                probe = base_probe.probes[layer_idx]
                probe.to(device)
                probe.eval()

                with torch.no_grad():
                    logits = probe(activations)
                    layer_logits[layer_idx] = logits
                    # Convert the OOD similarity vector into a tensor and repeat it
                    ood_similarity_tensor = torch.tensor(ood_similarity, dtype=torch.float, device=device)
                    ood_labels_repeated = ood_similarity_tensor.unsqueeze(0).repeat(logits.size(0), 1)
                    layer_labels[layer_idx] = ood_labels_repeated

        if not layer_logits:
            logger.warning(f"  OOD domain '{ood_domain_name}' has no valid logits, skipping")
            continue

        # Learn per-layer independent temperatures
        try:
            layer_temps = learn_layer_wise_temperatures(layer_logits, layer_labels, device=device)
            avg_temp = np.mean(list(layer_temps.values()))
            logger.info(f"    OOD domain '{ood_domain_name}' average temperature: {avg_temp:.4f} (from {len(layer_temps)} layers)")
            ood_temperatures.append(avg_temp)
            layer_temps_list.append(layer_temps)
        except Exception as e:
            logger.warning(f"  OOD domain '{ood_domain_name}' calibration failed: {e}, skipping")
            continue

    if not ood_temperatures:
        logger.warning("  All OOD domain calibrations failed, using default temperature=1.0")
        return 1.0, None

    # Take the average temperature (global)
    avg_T_ood = np.mean(ood_temperatures)
    logger.info(f"  OOD average temperature: {avg_T_ood:.4f} (from {len(ood_temperatures)} OOD domains)")
    logger.info(f"  OOD temperatures detail: {dict(zip(all_ood_similarities.keys(), ood_temperatures))}")

    # Compute per-layer independent temperatures (average the per-layer temperatures over all OOD domains)
    if layer_temps_list:
        # Collect the temperatures across all layers
        all_layers = set()
        for layer_temps in layer_temps_list:
            all_layers.update(layer_temps.keys())

        # For each layer, take the average temperature over all OOD domains
        ood_layer_temperatures = {}
        for layer_idx in all_layers:
            layer_temps_for_this_layer = [
                layer_temps[layer_idx]
                for layer_temps in layer_temps_list
                if layer_idx in layer_temps
            ]
            if layer_temps_for_this_layer:
                ood_layer_temperatures[layer_idx] = np.mean(layer_temps_for_this_layer)

        logger.info(f"  OOD per-layer independent temperatures: {len(ood_layer_temperatures)} layers")
        return float(avg_T_ood), ood_layer_temperatures
    else:
        return float(avg_T_ood), None


def calibrate_probe_for_cross_domain(
    base_probe: MultiLayerProbe,
    ppd_model: BaseModel,
    val_domain_data: Dict[int, List[Dict[str, torch.Tensor]]],
    avg_ood_similarity: Optional[np.ndarray],
    selected_domains: List[str],
    device: torch.device,
    T_single: float,
    T_ood: float,
    T_single_layer_wise: Optional[Dict[int, float]] = None,
    T_ood_layer_wise: Optional[Dict[int, float]] = None,
    foundation_layers: Optional[List[int]] = None
) -> Tuple[float, Optional[Dict[int, float]]]:
    """Calibrate the probe for the Cross-domain scenario."""
    logger.info("  Performing Cross-domain calibration using val data + average OOD similarity...")

    # Set foundation layers (default to layers 0-2 if not provided)
    if foundation_layers is None:
        foundation_layers = [0, 1, 2]

    # Global temperature: take the average
    T_cross = (T_single + T_ood) / 2.0

    # Per-layer independent temperature: if both are available, average per layer (skipping foundation layers)
    T_cross_layer_wise = None
    if T_single_layer_wise is not None and T_ood_layer_wise is not None:
        # Find all layers (excluding foundation layers)
        all_layers = set(T_single_layer_wise.keys()) & set(T_ood_layer_wise.keys())
        all_layers = [l for l in all_layers if l not in foundation_layers]
        T_cross_layer_wise = {
            layer_idx: (T_single_layer_wise[layer_idx] + T_ood_layer_wise[layer_idx]) / 2.0
            for layer_idx in all_layers
        }
        logger.info(f"  Cross-domain per-layer independent temperatures: {len(T_cross_layer_wise)} layers")
        logger.info(f"    Average temperature: {np.mean(list(T_cross_layer_wise.values())):.4f}")
    elif T_single_layer_wise is not None:
        # If only Single-domain has per-layer temperatures, use them (excluding foundation layers)
        T_cross_layer_wise = {k: v for k, v in T_single_layer_wise.items() if k not in foundation_layers}
        logger.info(f"  Cross-domain using Single-domain per-layer independent temperatures: {len(T_cross_layer_wise)} layers")
    elif T_ood_layer_wise is not None:
        # If only OOD has per-layer temperatures, use them (excluding foundation layers)
        T_cross_layer_wise = {k: v for k, v in T_ood_layer_wise.items() if k not in foundation_layers}
        logger.info(f"  Cross-domain using OOD per-layer independent temperatures: {len(T_cross_layer_wise)} layers")

    logger.info(f"  Cross-domain global temperature: {T_cross:.4f} (based on the average of Single-domain and OOD)")
    logger.warning("  The current implementation uses a simple average; using actual cross-domain data is recommended later")

    return T_cross, T_cross_layer_wise


def step5_compute_head_importance_multi(
    ppd_model: BaseModel,
    domain_data: Dict[int, List[Dict[str, torch.Tensor]]],
    domain_axes: torch.Tensor,
    multi_probe_system: MultiProbeSystem,
    output_dir: Path
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Step 5: Compute head importance (4 importance sets, one per probe).

    Args:
        ppd_model: The PPD model.
        domain_data: Domain data.
        domain_axes: Domain axes (one-hot).
        multi_probe_system: The 4-probe system.
        output_dir: Output directory.

    Returns:
        head_importance_dict: {
            "probe1": {layer_idx: [num_heads, num_domains]},
            "probe2_single_domain": {layer_idx: [num_heads, num_domains]},
            "probe3_ood": {layer_idx: [num_heads, num_domains]},
            "probe4_cross_domain": {layer_idx: [num_heads, num_domains]}
        }
    """
    logger.info("=" * 80)
    logger.info("Step 5: Compute head importance (4 importance sets)")
    logger.info("=" * 80)
    logger.info(f"  Model: PPD model")
    logger.info(f"  Number of layers: {ppd_model.model.config.num_hidden_layers}")
    logger.info(f"  Heads per layer: {ppd_model.model.config.num_attention_heads}")
    logger.info(f"  Number of domains: {len(domain_axes)}")
    logger.info(f"  Computing head importance separately for the 4 probes")
    logger.info("=" * 80)

    num_layers = ppd_model.model.config.num_hidden_layers
    num_heads_per_layer = ppd_model.model.config.num_attention_heads
    num_domains = len(domain_axes)

    head_importance_dict = {}

    # Compute importance for each probe
    probe_configs = [
        ("probe1", multi_probe_system.probe1, "Original probe (uncalibrated)"),
        ("probe2_single_domain", multi_probe_system.probe2.base_probe, "Single-domain probe (calibrated)"),
        ("probe3_ood", multi_probe_system.probe3.base_probe, "OOD probe (calibrated)"),
        ("probe4_cross_domain", multi_probe_system.probe4.base_probe, "Cross-domain probe (calibrated)")
    ]

    for probe_name, probe, description in probe_configs:
        logger.info(f"Computing head importance for {probe_name}: {description}")

        calculator = HeadImportanceCalculator(
            num_layers=num_layers,
            num_heads_per_layer=num_heads_per_layer,
            num_domains=num_domains,
            layer_probes=probe
        )

        # Compute importance
        # Load the foundation layers config (if it exists)
        foundation_layers = None
        layer_domain_relevances = None
        foundation_config_path = output_dir / "foundation_layers_config.json"
        if foundation_config_path.exists():
            from src.preorientation.ppd import load_foundation_layers_config
            config = load_foundation_layers_config(foundation_config_path)
            foundation_layers = config.get('foundation_layers', [])
            layer_domain_relevances = config.get('layer_domain_relevances', None)
            logger.info(f"  Loaded foundation layers: {foundation_layers} (will be skipped during computation)")
            if layer_domain_relevances:
                # Convert string keys into ints
                layer_domain_relevances = {int(k): v for k, v in layer_domain_relevances.items()}
                logger.info(f"  Loaded layer_domain_relevances: {len(layer_domain_relevances)} layers")

        # Time estimate
        logger.info(f"  Time estimate: about 2-3 minutes per layer, about 1.5-1.85 hours for 37 layers, about 6-7.4 hours for 4 importance sets")
        logger.info(f"  Current implementation: serial processing (stable and reliable)")
        logger.info(f"  GPU memory: 48G (parallelism is supported, but kept serial for stability)")

        head_importance = calculator.compute_importance(
            model=ppd_model.model,
            domain_data=domain_data,
            domain_axes=domain_axes,
            batch_size=32,  # Batch size (consistent with probe training)
            max_samples_per_domain=200,  # Maximum number of samples per domain (sampling)
            foundation_layers=foundation_layers,  # Foundation layers (skipped during computation)
            layer_group_size=5,  # Process 5 layers in parallel (supported by 48G GPU, but currently serial)
            layer_domain_relevances=layer_domain_relevances  # Layer domain relevances (used for comparison)
        )

        # Print statistics
        logger.info(f"  {probe_name} head importance statistics:")
        for layer_idx, importance in head_importance.items():
            if isinstance(importance, torch.Tensor):
                avg_importance = importance.mean().item()
                max_importance = importance.max().item()
                logger.info(f"    Layer {layer_idx}: average importance={avg_importance:.4f}, max importance={max_importance:.4f}")

        # Save importance
        importance_path = output_dir / f"head_importance_{probe_name}.pt"
        calculator.save(str(importance_path))
        logger.info(f"  {probe_name} head importance saved to: {importance_path}")
        logger.info(f"     File size: {importance_path.stat().st_size / 1024 / 1024:.2f} MB")

        head_importance_dict[probe_name] = head_importance

    # Save the importance mapping file
    importance_mapping = {
        "probe1": "Original probe (uncalibrated)",
        "probe2_single_domain": "Single-domain probe (calibrated)",
        "probe3_ood": "OOD probe (calibrated)",
        "probe4_cross_domain": "Cross-domain probe (calibrated)"
    }
    mapping_path = output_dir / "head_importance_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(importance_mapping, f, indent=2)
    logger.info(f"Importance mapping file saved to: {mapping_path}")

    return head_importance_dict


def step5_compute_head_importance(
    ppd_model: BaseModel,
    domain_data: Dict[int, List[Dict[str, torch.Tensor]]],
    domain_axes: torch.Tensor,
    final_layer_probes: MultiLayerProbe,
    output_dir: Path
) -> Dict[int, torch.Tensor]:
    """
    Step 5: Compute head importance.

    Args:
        ppd_model: The PPD model.
        domain_data: Domain data.
        domain_axes: Domain axes (one-hot).
        final_layer_probes: The final probes.
        output_dir: Output directory.

    Returns:
        head_importance: {layer_idx: [num_heads, num_domains]}
    """
    logger.info("=" * 80)
    logger.info("Step 5: Compute head importance")
    logger.info("=" * 80)
    logger.info(f"  Model: PPD model")
    logger.info(f"  Number of layers: {ppd_model.model.config.num_hidden_layers}")
    logger.info(f"  Heads per layer: {ppd_model.model.config.num_attention_heads}")
    logger.info(f"  Number of domains: {len(domain_axes)}")
    logger.info(f"  Using the final probes to project the residual into the domain space")
    logger.info("=" * 80)

    num_layers = ppd_model.model.config.num_hidden_layers
    num_heads_per_layer = ppd_model.model.config.num_attention_heads
    num_domains = len(domain_axes)

    calculator = HeadImportanceCalculator(
        num_layers=num_layers,
        num_heads_per_layer=num_heads_per_layer,
        num_domains=num_domains,
        layer_probes=final_layer_probes
    )

    # Compute importance
    logger.info("Starting head importance computation I_{l,h,k}...")
    logger.info("  Computing axis-aligned energy for each domain, layer and head")
    head_importance = calculator.compute_importance(
        model=ppd_model.model,
        domain_data=domain_data,
        domain_axes=domain_axes
    )

    # Print statistics
    logger.info("Head importance statistics:")
    for layer_idx, importance in head_importance.items():
        if isinstance(importance, torch.Tensor):
            avg_importance = importance.mean().item()
            max_importance = importance.max().item()
            logger.info(f"  Layer {layer_idx}: average importance={avg_importance:.4f}, max importance={max_importance:.4f}")

    # Save importance
    importance_path = output_dir / "head_importance.pt"
    calculator.save(str(importance_path))
    logger.info(f"Head importance saved to: {importance_path}")
    logger.info(f"   File size: {importance_path.stat().st_size / 1024 / 1024:.2f} MB")

    return head_importance


def step6_identify_whitelist(
    head_importance: Dict[int, torch.Tensor],
    num_layers: int,
    num_heads_per_layer: int,
    num_domains: int,
    output_dir: Path,
    foundation_layers: Optional[List[int]] = None,
    all_importance_sets: Optional[Dict[str, Dict[int, torch.Tensor]]] = None,
    use_statistical_test: bool = True
) -> List[tuple]:
    """
    Step 6: Identify the head whitelist.

    Args:
        head_importance: Head importance (primarily uses probe4_cross_domain).
        num_layers: Number of layers.
        num_heads_per_layer: Heads per layer.
        num_domains: Number of domains.
        output_dir: Output directory.
        foundation_layers: List of foundation layers.
        all_importance_sets: All importance sets (optional, for more robust identification).
        use_statistical_test: Whether to use a statistical significance test.

    Returns:
        whitelist: [(layer_idx, head_idx), ...]
    """
    logger.info("=" * 80)
    logger.info("Step 6: Identify the head whitelist")
    logger.info("=" * 80)

    identifier = HeadWhitelistIdentifier(
        num_layers=num_layers,
        num_heads_per_layer=num_heads_per_layer,
        num_domains=num_domains,
        foundation_layers=foundation_layers,
        use_statistical_test=use_statistical_test
    )

    whitelist = identifier.identify_whitelist(head_importance, all_importance_sets)

    logger.info(f"Identified {len(whitelist)} whitelist heads")

    # Save the whitelist
    whitelist_path = output_dir / "whitelist.json"
    with open(whitelist_path, 'w') as f:
        json.dump(whitelist, f)
    logger.info(f"Whitelist saved to: {whitelist_path}")

    return whitelist


DEFAULT_SELECTED_DOMAINS = ["chemistry", "finance", "history", "math", "philosophy", "technology"]


def main():
    parser = argparse.ArgumentParser(
        description="Offline stage of Probe-based Scenario Pruning (PSP): train probes, "
                    "calibrate, compute head importance, identify the whitelist. The base "
                    "model is frozen (no fine-tuning)."
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Model directory name under models/ (e.g. Qwen2.5-7B-Instruct). "
                             "Ignored if --model_path is given.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Full path to a HuggingFace model directory.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory containing train/ and val/ sub-directories.")
    parser.add_argument("--selected_domains", type=str, nargs="+",
                        default=DEFAULT_SELECTED_DOMAINS,
                        help="Selected domains; must match the per-domain JSON files in data/train/.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-generated from the model path if omitted).")
    parser.add_argument("--final_probe_epochs", type=int, default=20,
                        help="Number of epochs for probe training.")
    parser.add_argument("--num_samples_per_domain", type=int, default=1000,
                        help="Max input samples per domain used for probe training / importance.")
    parser.add_argument("--head_importance", type=str, default="single",
                        choices=["single", "multi"],
                        help="'single': one importance set from the base probe (fast, recommended). "
                             "'multi': four scenario-specific sets (single/ood/cross), uses cross-domain.")
    parser.add_argument("--ood_domain_name", type=str, default=None,
                        help="Optional OOD domain name used during calibration (falls back gracefully).")
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA device index (sets CUDA_VISIBLE_DEVICES). Omit to respect the environment.")
    parser.add_argument("--quantization", type=str, default="none",
                        help="BaseModel quantization (none/int4/int8/fp8; int4 recommended for large models).")
    args = parser.parse_args()

    import os
    from src.utils.model_utils import get_output_dir_for_model

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # When CUDA_VISIBLE_DEVICES is set, the visible GPU is remapped to cuda:0.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Resolve the model path
    if args.model_path:
        model_path = args.model_path
    elif args.model:
        model_path = str(project_root / "models" / args.model)
    else:
        parser.error("Provide either --model <name under models/> or --model_path <path>.")
    if not Path(model_path).exists():
        logger.error(f"Model path does not exist: {model_path}")
        return

    # Output directory
    if args.output_dir is None:
        output_dir = get_output_dir_for_model(
            base_output_dir=str(project_root / "outputs"),
            model_path=model_path,
            subdir="ppd_pipeline",
        )
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load the frozen base model
    logger.info(f"Loading model: {model_path}")
    base_model = BaseModel(
        model_name=model_path,
        quantization=args.quantization,
        mixed_precision=False,
        torch_dtype="float16",
        device=device,
    )

    # Domain axes (one-hot over the selected domains)
    num_domains = len(args.selected_domains)
    domain_axes = create_domain_axes_onehot(num_domains=num_domains)

    # Resolve train / val data directories
    data_dir = Path(args.data_dir)
    if (data_dir / "train").exists() and (data_dir / "val").exists():
        train_data_dir, val_data_dir = data_dir / "train", data_dir / "val"
    elif data_dir.name == "train":
        train_data_dir, val_data_dir = data_dir, data_dir.parent / "val"
    else:
        train_data_dir = val_data_dir = data_dir
    logger.info(f"Train data: {train_data_dir}")
    logger.info(f"Val data:   {val_data_dir}")

    # Load domain data (stopwords removed, as required by the probe/importance design)
    logger.info("Loading training data...")
    train_domain_data = load_domain_data(
        train_data_dir, args.selected_domains, base_model,
        num_samples_per_domain=args.num_samples_per_domain, remove_stopwords=True,
    )
    logger.info("Loading validation data...")
    val_domain_data = load_domain_data(
        val_data_dir, args.selected_domains, base_model,
        num_samples_per_domain=args.num_samples_per_domain, remove_stopwords=True,
    )
    all_domain_data = {k: train_domain_data[k] + val_domain_data.get(k, [])
                       for k in train_domain_data.keys()}

    # Step 4: train the final linear probes (the base model stays frozen)
    final_probes = step4_final_probe_training(
        base_model, train_domain_data, val_domain_data,
        args.selected_domains, output_dir, args.final_probe_epochs,
    )

    # Step 4.5: temperature-scaling calibration (4-probe system)
    multi_probe_system = step4_5_calibrate_probes(
        final_probes, base_model, train_domain_data, val_domain_data,
        args.selected_domains, output_dir, args.ood_domain_name, foundation_layers=None,
    )

    # Step 5: axis-aligned head importance I_{l,h,k}
    if args.head_importance == "multi":
        importance_sets = step5_compute_head_importance_multi(
            base_model, all_domain_data, domain_axes, multi_probe_system, output_dir,
        )
        head_importance = importance_sets.get("probe4_cross_domain",
                                              importance_sets.get("probe1"))
        # Save the chosen set under the unified name used at inference time.
        num_layers = base_model.model.config.num_hidden_layers
        num_heads_per_layer = base_model.model.config.num_attention_heads
        importance_data = {
            "num_layers": num_layers,
            "num_heads_per_layer": num_heads_per_layer,
            "num_domains": num_domains,
            "importance": {
                str(li): (head_importance[li].cpu().tolist()
                          if isinstance(head_importance[li], torch.Tensor)
                          else head_importance[li].tolist())
                for li in range(num_layers) if li in head_importance
            },
        }
        with open(output_dir / "head_importance.pt", "w") as f:
            json.dump(importance_data, f, indent=2)
        logger.info(f"Saved unified head importance to: {output_dir / 'head_importance.pt'}")
    else:
        head_importance = step5_compute_head_importance(
            base_model, all_domain_data, domain_axes, final_probes, output_dir,
        )

    # Step 6: identify the domain-invariant head whitelist
    num_layers = base_model.model.config.num_hidden_layers
    num_heads_per_layer = base_model.model.config.num_attention_heads
    whitelist = step6_identify_whitelist(
        head_importance, num_layers, num_heads_per_layer, num_domains, output_dir,
        foundation_layers=None, all_importance_sets=None, use_statistical_test=True,
    )

    logger.info("=" * 80)
    logger.info("Offline stage complete.")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Whitelist heads: {len(whitelist)}")
    logger.info("  Artifacts: probe1_base.pt, probe_temperatures.json, calibration/, "
                "head_importance.pt, whitelist.json")
    logger.info("  These can now be used by scripts/run_inference.py.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
