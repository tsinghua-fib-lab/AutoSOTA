"""
Linear probe module
Used to establish the layer-domain mapping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import logging

from ..utils import get_logger

logger = get_logger(__name__)


class LinearProbe(nn.Module):
    """
    Linear probe: trains a linear classifier on each layer to establish the layer-domain mapping
    """

    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dim: int = 256,
            nonlinear: bool = False
    ):
        """
        Initialize the linear probe

        Args:
            input_dim: Input dimension (activation dimension)
            num_domains: Number of domains
            hidden_dim: Hidden layer dimension (if nonlinear)
            nonlinear: Whether to use a nonlinear MLP
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_domains = num_domains
        self.nonlinear = nonlinear

        if nonlinear:
            # Nonlinear MLP
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_domains)
            )
        else:
            # Linear classifier (outputs logits, later converted to probabilities via sigmoid)
            self.classifier = nn.Linear(input_dim, num_domains)

        # Use BCE loss (1-vs-rest sigmoid)
        # Following the paper's design: one-hot labels are used during training, but cross-domain is supported at inference
        # Therefore sigmoid is used instead of softmax, allowing multiple domains to be active simultaneously
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Ensure the input dtype matches the model weight dtype
        if isinstance(self.classifier, nn.Sequential):
            # For Sequential, get the weight dtype of the first Linear layer
            target_dtype = next(m.weight.dtype for m in self.classifier if isinstance(m, nn.Linear))
        else:
            # For a single Linear layer, get the weight dtype directly
            target_dtype = self.classifier.weight.dtype
        
        if target_dtype != x.dtype:
            x = x.to(target_dtype)
        return self.classifier(x)

    def train_probe(
            self,
            activations: torch.Tensor,
            domain_labels: torch.Tensor,
            num_epochs: int = 10,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            device: torch.device = None,
            val_activations: Optional[torch.Tensor] = None,
            val_domain_labels: Optional[torch.Tensor] = None,
            **kwargs  # Supports early stopping parameters
    ) -> Dict[str, float]:
        """
        Train the probe

        Args:
            activations: Activation tensor [batch_size, input_dim]
            domain_labels: Domain labels [batch_size]
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            device: Compute device
            val_activations: Validation activation tensor [val_batch_size, input_dim] (optional)
            val_domain_labels: Validation domain labels [val_batch_size] (optional)

        Returns:
            Training history dictionary (contains train_loss, train_acc, val_loss, val_acc)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(device)
        self.train()

        # Type checking and conversion (YAML may parse numbers as strings)
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)
        if isinstance(batch_size, str):
            batch_size = int(batch_size)
        if isinstance(num_epochs, str):
            num_epochs = int(num_epochs)
        
        learning_rate = float(learning_rate)
        batch_size = int(batch_size)
        num_epochs = int(num_epochs)

        # Create the data loader
        # If the amount of data is smaller than batch_size, automatically adjust batch_size
        actual_batch_size = min(batch_size, len(activations))
        if actual_batch_size < batch_size:
            logger.warning(f"Data size ({len(activations)}) is smaller than batch_size ({batch_size}), adjusting to {actual_batch_size}")

        dataset = TensorDataset(activations, domain_labels)
        dataloader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        history = {'loss': [], 'accuracy': []}
        if val_activations is not None:
            history['val_loss'] = []
            history['val_accuracy'] = []

        # Early stopping configuration (use validation accuracy if a validation set is provided, otherwise training accuracy)
        early_stopping_patience = kwargs.get('early_stopping_patience', None)
        early_stopping_min_delta = kwargs.get('early_stopping_min_delta', 0.001)
        best_accuracy = 0.0
        patience_counter = 0
        best_model_state = None
        use_val_for_early_stopping = val_activations is not None

        logger.info(f"Starting linear probe training: {num_epochs} epochs, batch_size={actual_batch_size}, data size={len(activations)}")
        if val_activations is not None:
            logger.info(f"  Validation set size: {len(val_activations)}")
        if early_stopping_patience:
            stopping_metric = "validation accuracy" if use_val_for_early_stopping else "training accuracy"
            logger.info(f"Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}, using {stopping_metric}")

        # Output detailed logs every few epochs
        log_interval = max(1, num_epochs // 10) if num_epochs >= 10 else 1

        try:
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                correct = 0
                total = 0

                for batch_activations, batch_labels in dataloader:
                    batch_activations = batch_activations.to(device)
                    batch_labels = batch_labels.to(device)

                    optimizer.zero_grad()
                    logits = self(batch_activations)  # [batch, num_domains]

                    # Convert one-hot labels to binary labels (1-vs-rest)
                    # If domain_labels are class indices, convert them to binary
                    if batch_labels.dim() == 1:
                        # [batch] -> [batch, num_domains] (one-hot)
                        binary_labels = torch.zeros_like(logits)
                        binary_labels.scatter_(1, batch_labels.unsqueeze(1), 1.0)
                    else:
                        # Already binary labels
                        binary_labels = batch_labels

                    loss = self.criterion(logits, binary_labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    # Compute accuracy: the domain with the highest predicted probability
                    probs = torch.sigmoid(logits)
                    predicted = torch.argmax(probs, dim=1)
                    if batch_labels.dim() == 1:
                        # If the labels are class indices
                        correct += (predicted == batch_labels).sum().item()
                    else:
                        # If the labels are binary, use argmax
                        true_labels = torch.argmax(binary_labels, dim=1)
                        correct += (predicted == true_labels).sum().item()
                    total += batch_labels.size(0)

                # Prevent division-by-zero errors
                num_batches = max(len(dataloader), 1)
                avg_loss = epoch_loss / num_batches
                accuracy = correct / total if total > 0 else 0.0
                history['loss'].append(avg_loss)
                history['accuracy'].append(accuracy)
                
                # Validation set evaluation
                val_accuracy = None
                val_loss = None
                if val_activations is not None and val_domain_labels is not None:
                    self.eval()
                    with torch.no_grad():
                        val_activations_device = val_activations.to(device)
                        val_labels_device = val_domain_labels.to(device)
                        val_logits = self(val_activations_device)

                        # Compute validation loss
                        if val_labels_device.dim() == 1:
                            val_binary_labels = torch.zeros_like(val_logits)
                            val_binary_labels.scatter_(1, val_labels_device.unsqueeze(1), 1.0)
                        else:
                            val_binary_labels = val_labels_device
                        val_loss = self.criterion(val_logits, val_binary_labels).item()

                        # Compute validation accuracy
                        val_probs = torch.sigmoid(val_logits)
                        val_predicted = torch.argmax(val_probs, dim=1)
                        if val_labels_device.dim() == 1:
                            val_correct = (val_predicted == val_labels_device).sum().item()
                        else:
                            val_true_labels = torch.argmax(val_binary_labels, dim=1)
                            val_correct = (val_predicted == val_true_labels).sum().item()
                        val_accuracy = val_correct / len(val_labels_device) if len(val_labels_device) > 0 else 0.0
                        
                        history['val_loss'].append(val_loss)
                        history['val_accuracy'].append(val_accuracy)
                    self.train()

                    # Output detailed logs every log_interval epochs
                    if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == num_epochs - 1:
                        val_info = f", Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}" if val_accuracy is not None else ""
                        logger.info(f"  Epoch {epoch+1}/{num_epochs}: Train Loss={epoch_loss:.4f}, Train Acc={accuracy:.4f}{val_info}")
                else:
                    # Output detailed logs every log_interval epochs (no validation set)
                    if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == num_epochs - 1:
                        logger.info(f"  Epoch {epoch+1}/{num_epochs}: Train Loss={epoch_loss:.4f}, Train Acc={accuracy:.4f}")

                # Early stopping check (use validation accuracy if provided)
                if early_stopping_patience:
                    current_accuracy = val_accuracy if use_val_for_early_stopping and val_accuracy is not None else accuracy
                    if current_accuracy > best_accuracy + early_stopping_min_delta:
                        best_accuracy = current_accuracy
                        patience_counter = 0
                        # Save the best model state
                        best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                        logger.debug(f"Epoch {epoch + 1}: new best accuracy {best_accuracy:.4f}")
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping triggered: {patience_counter} epochs without improvement (best accuracy: {best_accuracy:.4f})")
                            # Restore the best model state
                            if best_model_state:
                                self.load_state_dict(best_model_state)
                                logger.info("Restored the best model state")
                            break

                # More frequent log output (every epoch or every 2 epochs)
                log_interval = max(1, min(5, num_epochs // 3))  # Output at least 3 times
                if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == num_epochs - 1:
                    early_stop_info = f" (patience: {patience_counter}/{early_stopping_patience})" if early_stopping_patience else ""
                    val_info = f", Val_Loss={val_loss:.4f}, Val_Acc={val_accuracy:.4f}" if val_accuracy is not None else ""
                    logger.info(f"Epoch {epoch + 1}/{num_epochs}: Train_Loss={avg_loss:.4f}, Train_Acc={accuracy:.4f}{val_info}{early_stop_info}")

            final_accuracy = best_accuracy if early_stopping_patience and best_model_state else history['accuracy'][-1]
            final_val_accuracy = history.get('val_accuracy', [None])[-1] if history.get('val_accuracy') else None
            val_info = f", validation accuracy={final_val_accuracy:.4f}" if final_val_accuracy is not None else ""
            logger.info(f"Training complete: final training accuracy={final_accuracy:.4f}{val_info} (trained {len(history['accuracy'])} epochs)")
        except Exception as e:
            logger.error(f"An error occurred during training: {e}", exc_info=True)
            raise
        
        return history

    def get_layer_domain_importance(
            self,
            activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the importance scores of a layer for each domain

        Following the paper's design: uses 1-vs-rest sigmoid, allowing multiple domains to be active simultaneously.
        This supports cross-domain scenarios and does not require the probabilities to sum to 1.

        Args:
            activations: Activation tensor [batch_size, input_dim]

        Returns:
            Importance scores [batch_size, num_domains] (sigmoid probabilities, not normalized)
        """
        self.eval()
        # Ensure activations and the model are on the same device with matching dtype
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        activations = activations.to(device=device, dtype=dtype)

        with torch.no_grad():
            logits = self(activations)
            # Use sigmoid to get 1-vs-rest probabilities (not normalized, supports multi-domain)
            importance = torch.sigmoid(logits)
        return importance

    def save(self, path: str):
        """Save the probe model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'num_domains': self.num_domains,
            'hidden_dim': getattr(self.classifier[0], 'out_features', None) if self.nonlinear else None,
            'nonlinear': self.nonlinear
        }, path)
        logger.info(f"Probe model saved: {path}")

    @classmethod
    def load(cls, path: str, device: torch.device = None):
        """Load the probe model"""
        # If CUDA_VISIBLE_DEVICES is set, first load to CPU, then move to the device
        # This avoids device index mismatch issues
        import os
        if device is not None and device.type == 'cuda' and "CUDA_VISIBLE_DEVICES" in os.environ:
            # First load to CPU, then move to cuda:0 (which maps to the specified GPU)
            checkpoint = torch.load(path, map_location='cpu')
            probe = cls(
                input_dim=checkpoint['input_dim'],
                num_domains=checkpoint['num_domains'],
                hidden_dim=checkpoint.get('hidden_dim', 256),
                nonlinear=checkpoint.get('nonlinear', False)
            )
            probe.load_state_dict(checkpoint['state_dict'])
            probe.to(torch.device('cuda:0'))  # Move to cuda:0 (which maps to the GPU specified by CUDA_VISIBLE_DEVICES)
        else:
            checkpoint = torch.load(path, map_location=device if device else 'cpu')
            probe = cls(
                input_dim=checkpoint['input_dim'],
                num_domains=checkpoint['num_domains'],
                hidden_dim=checkpoint.get('hidden_dim', 256),
                nonlinear=checkpoint.get('nonlinear', False)
            )
            probe.load_state_dict(checkpoint['state_dict'])
            if device:
                probe.to(device)
        logger.info(f"Probe model loaded: {path}")
        return probe


class MultiLayerProbe:
    """
    Multi-layer probe manager
    Trains an independent linear probe for each layer
    """

    def __init__(
            self,
            num_layers: int,
            activation_dims: Dict[int, int],  # {layer_idx: activation_dim}
            num_domains: int,
            hidden_dim: int = 256,
            nonlinear: bool = False
    ):
        """
        Initialize the multi-layer probe

        Args:
            num_layers: Number of layers
            activation_dims: Activation dimension of each layer
            num_domains: Number of domains
            hidden_dim: Hidden layer dimension
            nonlinear: Whether to use a nonlinear MLP
        """
        self.num_layers = num_layers
        self.num_domains = num_domains
        self.probes = {}

        for layer_idx in range(num_layers):
            if layer_idx in activation_dims:
                input_dim = activation_dims[layer_idx]
                self.probes[layer_idx] = LinearProbe(
                    input_dim=input_dim,
                    num_domains=num_domains,
                    hidden_dim=hidden_dim,
                    nonlinear=nonlinear
                )

        logger.info(f"Created multi-layer probe: {len(self.probes)} layers")

    def train_all_layers(
            self,
            layer_activations: Dict[int, torch.Tensor],  # {layer_idx: [batch, dim]}
            domain_labels: torch.Tensor,
            num_epochs: int = 10,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            device: torch.device = None,
            **kwargs  # Supports training_strategy and num_cycles parameters
    ) -> Dict[int, Dict[str, List[float]]]:
        """
        Train probes for all layers

        Args:
            layer_activations: Activations of each layer
            domain_labels: Domain labels
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            device: Compute device

        Returns:
            {layer_idx: training history}
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get training strategy parameters
        training_strategy = kwargs.get('training_strategy', 'sequential')
        num_cycles = kwargs.get('num_cycles', 1)

        histories = {}

        if training_strategy == "sequential":
            # Sequential training: train each layer independently for num_epochs epochs
            logger.info(f"Using sequential training strategy: training each layer for {num_epochs} epochs")

            for layer_idx, probe in self.probes.items():
                if layer_idx in layer_activations:
                    try:
                        logger.info(f"Training probe for layer {layer_idx}...")
                        activations = layer_activations[layer_idx]

                        # Ensure activations are on the correct device and dtype
                        activations = activations.to(device=device)
                        # Ensure dtype matches (if the model is FP16, activations should also be FP16)
                        probe_dtype = next(probe.parameters()).dtype
                        activations = activations.to(dtype=probe_dtype)

                        # Check the data
                        if len(activations) == 0:
                            logger.error(f"Activation data for layer {layer_idx} is empty, skipping")
                            continue

                        if len(activations.shape) != 2:
                            logger.error(f"Activation shape for layer {layer_idx} is incorrect: {activations.shape}, expected [batch, dim]")
                            continue

                        logger.debug(f"Layer {layer_idx}: activation shape={activations.shape}, label shape={domain_labels.shape}")

                        # Get validation data (if provided)
                        val_acts = None
                        val_labels = None
                        if 'val_layer_activations' in kwargs and layer_idx in kwargs['val_layer_activations']:
                            val_acts = kwargs['val_layer_activations'][layer_idx].to(device=device)
                            probe_dtype = next(probe.parameters()).dtype
                            val_acts = val_acts.to(dtype=probe_dtype)
                        if 'val_domain_labels' in kwargs:
                            val_labels = kwargs['val_domain_labels'].to(device=device)

                        history = probe.train_probe(
                            activations=activations,
                            domain_labels=domain_labels,  # Fix: use the domain_labels parameter
                            num_epochs=num_epochs,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            device=device,
                            val_activations=val_acts,
                            val_domain_labels=val_labels,
                            early_stopping_patience=kwargs.get('early_stopping_patience'),
                            early_stopping_min_delta=kwargs.get('early_stopping_min_delta')
                        )
                        histories[layer_idx] = history
                        train_acc = history['accuracy'][-1]
                        val_acc = history.get('val_accuracy', [None])[-1] if history.get('val_accuracy') else None
                        val_info = f", validation accuracy={val_acc:.4f}" if val_acc is not None else ""
                        logger.info(f"Probe training for layer {layer_idx} complete: training accuracy={train_acc:.4f}{val_info}")
                    except Exception as e:
                        logger.error(f"An error occurred while training the probe for layer {layer_idx}: {e}", exc_info=True)
                        continue

        elif training_strategy == "cyclic":
            # Cyclic training: one pass over all layers counts as one cycle, repeated num_cycles times
            logger.info(f"Using cyclic training strategy: {num_cycles} cycles, training each layer for {num_epochs // num_cycles if num_cycles > 0 else num_epochs} epochs per cycle")

            epochs_per_cycle = num_epochs // num_cycles if num_cycles > 0 else num_epochs
            if epochs_per_cycle < 1:
                epochs_per_cycle = 1

            for cycle in range(num_cycles):
                logger.info(f"Cycle {cycle + 1}/{num_cycles}...")

                for layer_idx, probe in self.probes.items():
                    if layer_idx not in layer_activations:
                        continue

                    try:
                        activations = layer_activations[layer_idx]
                        activations = activations.to(device=device)
                        probe_dtype = next(probe.parameters()).dtype
                        activations = activations.to(dtype=probe_dtype)

                        if len(activations) == 0 or len(activations.shape) != 2:
                            continue

                        # On the first cycle, initialize the history
                        if layer_idx not in histories:
                            histories[layer_idx] = {'loss': [], 'accuracy': []}

                        # Train this layer
                        # For cyclic training, the early stopping patience should be adjusted per cycle
                        cycle_early_stopping_patience = None
                        if kwargs.get('early_stopping_patience'):
                            # Within each cycle, if patience is too large it may lead to undertraining,
                            # so only use early stopping in the last cycle, or use a smaller patience
                            if cycle == num_cycles - 1:  # Last cycle
                                cycle_early_stopping_patience = kwargs.get('early_stopping_patience')

                        cycle_history = probe.train_probe(
                            activations=activations,
                            domain_labels=domain_labels,  # Fix: use the domain_labels parameter
                            num_epochs=epochs_per_cycle,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            device=device,
                            early_stopping_patience=cycle_early_stopping_patience,
                            early_stopping_min_delta=kwargs.get('early_stopping_min_delta')
                        )
                        
                        # Merge histories
                        histories[layer_idx]['loss'].extend(cycle_history.get('loss', []))
                        histories[layer_idx]['accuracy'].extend(cycle_history.get('accuracy', []))

                        if cycle == 0:
                            logger.info(f"  Layer {layer_idx}: training started")
                        elif (cycle + 1) % max(1, num_cycles // 4) == 0:
                            logger.info(f"  Layer {layer_idx}: completed {cycle + 1}/{num_cycles} cycles")

                    except Exception as e:
                        logger.error(f"An error occurred while training the probe for layer {layer_idx}: {e}", exc_info=True)
                        continue
        else:
            raise ValueError(f"Unknown training strategy: {training_strategy}")

        logger.info(f"Probe training for all layers complete: {len(histories)}/{len(self.probes)} layers succeeded")
        return histories

    def get_layer_domain_map(self, layer_activations: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Get the layer-domain mapping matrix

        Args:
            layer_activations: Activations of each layer

        Returns:
            {layer_idx: [batch, num_domains] importance scores}
        """
        layer_domain_map = {}

        if not layer_activations:
            return layer_domain_map

        for layer_idx, probe in self.probes.items():
            if layer_idx in layer_activations:
                activations = layer_activations[layer_idx]
                try:
                    importance = probe.get_layer_domain_importance(activations)
                    layer_domain_map[layer_idx] = importance
                except Exception as e:
                    # If the probe computation for a layer fails, log a warning but continue with other layers
                    import logging
                    logger = logging.getLogger(__name__)
                    if not hasattr(self, f'_probe_error_logged_{layer_idx}'):
                        logger.warning(f"⚠️  Probe computation for layer {layer_idx} failed: {e}")
                        setattr(self, f'_probe_error_logged_{layer_idx}', True)

        return layer_domain_map

    def save(self, path: str):
        """Save all probes"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for layer_idx, probe in self.probes.items():
            probe_path = path / f"probe_layer_{layer_idx}.pt"
            probe.save(str(probe_path))

        # Save metadata
        metadata = {
            'num_layers': self.num_layers,
            'num_domains': self.num_domains,
            'layer_indices': list(self.probes.keys())
        }
        import json
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f)

        logger.info(f"All probes saved: {path}")

    @classmethod
    def load(cls, path: str, device: torch.device = None):
        """Load all probes"""
        path = Path(path)
        import json

        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        # Rebuilding activation_dims would require fetching from the model; simplified here
        # In practice, activation_dims should be saved
        activation_dims = {}
        probes = {}

        # If CUDA_VISIBLE_DEVICES is set, the device needs to be mapped to cuda:0
        import os
        actual_device = device
        if device is not None and device.type == 'cuda' and "CUDA_VISIBLE_DEVICES" in os.environ:
            actual_device = torch.device('cuda:0')
            logger.info(f"Detected CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, mapping device to: {actual_device}")

        for layer_idx in metadata['layer_indices']:
            probe_path = path / f"probe_layer_{layer_idx}.pt"
            checkpoint = torch.load(probe_path, map_location='cpu')  # First load to CPU
            probe = LinearProbe.load(str(probe_path), actual_device)
            probes[layer_idx] = probe
            activation_dims[layer_idx] = checkpoint['input_dim']

        instance = cls.__new__(cls)
        instance.num_layers = metadata['num_layers']
        instance.num_domains = metadata['num_domains']
        instance.probes = probes

        logger.info(f"All probes loaded: {path}")
        return instance