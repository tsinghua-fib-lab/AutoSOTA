"""
Extends the BaseTrainer to handle graph-structured data with PyTorch Geometric.
Supports node classification tasks where the graph structure is shared across all clients
but each client trains on different nodes.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from appfl.algorithm.trainer import BaseTrainer


class PyGTrainer(BaseTrainer):
    """
    Trainer for PyTorch Geometric models in federated learning.

    This trainer handles the unique requirements of GNN training:
    - Uses the full graph structure for message passing
    - Trains only on client-specific nodes
    - Properly handles graph data in DataLoader
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        metric: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the PyG trainer.

        Args:
            model: PyTorch Geometric model (e.g., GCN, GAT)
            loss_fn: Loss function (e.g., NLLLoss)
            metric: Evaluation metric function
            train_dataset: Training dataset containing node subgraph
            val_dataset: Validation dataset containing node subgraph
            train_configs: Training configuration
            logger: Logger instance
        """
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            **kwargs,
        )

        # Extract the full graph data from the dataset
        if train_dataset is not None and hasattr(train_dataset, "graph_data"):
            self.graph_data = train_dataset.graph_data
        else:
            self.graph_data = None

        self.device = train_configs.get("device", "cpu")
        # Move graph data to device
        if self.graph_data is not None:
            self.graph_data = self.graph_data.to(self.device)

        # Create data loaders
        self.train_loader = (
            DataLoader(
                self.train_dataset,
                batch_size=self.train_configs.get("train_batch_size", 32),
                shuffle=self.train_configs.get("train_data_shuffle", True),
                num_workers=self.train_configs.get("num_workers", 0),
            )
            if self.train_dataset is not None
            else None
        )

        self.val_loader = (
            DataLoader(
                self.val_dataset,
                batch_size=self.train_configs.get("val_batch_size", 32),
                shuffle=False,
                num_workers=self.train_configs.get("num_workers", 0),
            )
            if self.val_dataset is not None
            else None
        )

        # Create optimizer
        if self.model is not None:
            optimizer_name = self.train_configs.get("optim", "Adam")
            learning_rate = self.train_configs.get("optim_args", {}).get("lr", 0.001)

            if hasattr(torch.optim, optimizer_name):
                optimizer_class = getattr(torch.optim, optimizer_name)
                optim_args = self.train_configs.get("optim_args", {})
                self.optimizer = optimizer_class(self.model.parameters(), **optim_args)
            else:
                # Default to Adam if optimizer not found
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=learning_rate
                )
        else:
            self.optimizer = None

    def train(self, **kwargs):
        """
        Train the model for the specified number of local steps/epochs.
        """
        # Initialize validation results dictionary
        if "round" in kwargs:
            self.round = kwargs["round"]
        self.val_results = {"round": self.round + 1}

        self.model.train()
        self.model.to(self.device)

        if self.loss_fn is not None:
            self.loss_fn.to(self.device)

        # Get training configuration
        mode = self.train_configs.get("mode", "epoch")
        num_local_steps = self.train_configs.get("num_local_steps", 100)
        num_local_epochs = self.train_configs.get("num_local_epochs", 1)

        # Pre-validation if enabled
        do_pre_validation = (
            self.train_configs.get("do_pre_validation", False)
            and self.val_loader is not None
        )
        if do_pre_validation:
            val_loss, val_accuracy = self.validate()
            self.val_results["pre_val_loss"] = val_loss
            self.val_results["pre_val_accuracy"] = val_accuracy
            if self.logger:
                self.logger.info(
                    f"Round {self.round} Pre-Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
                )

        if mode == "step":
            train_loss, train_accuracy = self._train_by_steps(num_local_steps)
        else:
            train_loss, train_accuracy = self._train_by_epochs(num_local_epochs)

        # Store training metrics
        self.val_results["train_loss"] = train_loss
        self.val_results["train_accuracy"] = train_accuracy

        # Post-validation after training
        do_validation = (
            self.train_configs.get("do_validation", False)
            and self.val_loader is not None
        )
        if do_validation:
            val_loss, val_accuracy = self.validate()
            self.val_results["val_loss"] = val_loss
            self.val_results["val_accuracy"] = val_accuracy
            if self.logger:
                self.logger.info(
                    f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
                )

    def _train_by_steps(self, num_steps):
        """Train for a fixed number of gradient steps."""
        step = 0
        epoch = 0
        total_loss = 0.0
        target_true, target_pred = [], []

        while step < num_steps:
            for batch_idx, batch in enumerate(self.train_loader):
                if step >= num_steps:
                    break

                self.optimizer.zero_grad()

                # For graph data, we need the full graph structure
                if self.graph_data is not None:
                    # Get node features, labels, and indices from batch
                    features, labels, node_indices = batch

                    # Forward pass on the entire graph
                    output = self.model(self.graph_data.x, self.graph_data.edge_index)

                    # Compute loss only on training nodes
                    loss = self.loss_fn(output[node_indices], labels.to(self.device))
                else:
                    raise ValueError("Graph data not found in dataset")

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Track metrics
                total_loss += loss.item()
                target_true.append(labels.detach().cpu().numpy())
                target_pred.append(output[node_indices].detach().cpu().numpy())

                step += 1

                if self.logger and step % 10 == 0:
                    self.logger.info(
                        f"Step [{step}/{num_steps}], Loss: {loss.item():.4f}"
                    )

            epoch += 1

        # Compute average metrics
        avg_loss = total_loss / num_steps
        if self.metric is not None and len(target_true) > 0:
            import numpy as np

            target_true = np.concatenate(target_true)
            target_pred = np.concatenate(target_pred)
            avg_accuracy = float(self.metric(target_pred, target_true))
        else:
            avg_accuracy = 0.0

        return avg_loss, avg_accuracy

    def _train_by_epochs(self, num_epochs):
        """Train for a fixed number of epochs."""
        final_loss = 0.0
        final_accuracy = 0.0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            target_true, target_pred = [], []

            for batch_idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                # For graph data, we need the full graph structure
                if self.graph_data is not None:
                    # Get node features, labels, and indices from batch
                    features, labels, node_indices = batch

                    # Forward pass on the entire graph
                    output = self.model(self.graph_data.x, self.graph_data.edge_index)

                    # Compute loss only on training nodes
                    loss = self.loss_fn(output[node_indices], labels.to(self.device))
                else:
                    raise ValueError("Graph data not found in dataset")

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Track predictions for metric computation
                target_true.append(labels.detach().cpu().numpy())
                target_pred.append(output[node_indices].detach().cpu().numpy())

            avg_loss = epoch_loss / num_batches

            # Compute accuracy for this epoch
            if self.metric is not None and len(target_true) > 0:
                import numpy as np

                target_true_np = np.concatenate(target_true)
                target_pred_np = np.concatenate(target_pred)
                epoch_accuracy = float(self.metric(target_pred_np, target_true_np))
            else:
                epoch_accuracy = 0.0

            if self.logger:
                self.logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
                )

            # Store final epoch metrics
            final_loss = avg_loss
            final_accuracy = epoch_accuracy

        return final_loss, final_accuracy

    def validate(self):
        """
        Validate the model on the validation dataset.

        Returns:
            val_loss: Average validation loss
            val_metric: Validation metric value
        """
        self.model.eval()

        val_loss = 0.0
        val_metric = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                features, labels, node_indices = batch

                # Move labels and node indices to the correct device
                labels = labels.to(self.device)
                node_indices = node_indices.to(self.device)

                # Forward pass
                output = self.model(self.graph_data.x, self.graph_data.edge_index)

                # Compute loss and metric on validation nodes
                loss = self.loss_fn(output[node_indices], labels)

                if self.metric is not None:
                    metric_val = self.metric(output[node_indices], labels)
                    val_metric += metric_val

                val_loss += loss.item()
                num_batches += 1

        val_loss /= num_batches
        val_metric /= num_batches

        return val_loss, val_metric

    def get_parameters(
        self,
    ) -> Union[Dict, OrderedDict, Tuple[Union[Dict, OrderedDict], Dict]]:
        """
        Return model parameters for communication to server.

        Returns:
            Model state dict or tuple of (state_dict, metadata)
        """
        model_state = self.model.state_dict()
        return (
            (model_state, self.val_results)
            if hasattr(self, "val_results")
            else model_state
        )

    def load_parameters(self, params: Union[Dict, OrderedDict]) -> None:
        """
        Load parameters from the server.

        Args:
            params: Model state dict from server
        """
        self.model.load_state_dict(params, strict=False)
