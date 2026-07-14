import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from networks.network_interface import Network
from networks.layers import BP_layer
from networks.activation_function import Softplus, Linear


class ReplayBuffer:
    """
    Replay buffer that stores inputs, labels, logits (dark knowledge), and task_id.

    Uses reservoir sampling to maintain a fixed-size buffer with uniform
    sampling probability across all seen samples.
    """

    def __init__(self, buffer_size, device="cpu"):
        self.buffer_size = buffer_size if buffer_size is not None else 500
        self.device = device

        self.buffer_x = None
        self.buffer_y = None
        self.buffer_logits = None
        self.buffer_task_id = None

        self.num_seen = 0
        self.current_size = 0

    def add(self, x, y, logits, task_id):
        """
        Add samples to buffer using reservoir sampling.

        Args:
            x: Input tensor [batch_size, ...]
            y: Labels [batch_size] (class indices, not one-hot)
            logits: Model outputs [batch_size, num_classes]
            task_id: Current task ID (for masking MSE on replay)
        """
        batch_size = x.size(0)

        # Initialize buffer on first add
        if self.buffer_x is None:
            x_shape = (self.buffer_size,) + x.shape[1:]
            logits_shape = (self.buffer_size,) + logits.shape[1:]

            self.buffer_x = torch.zeros(*x_shape, device=self.device)
            self.buffer_y = torch.zeros(
                self.buffer_size, dtype=torch.long, device=self.device
            )
            self.buffer_logits = torch.zeros(*logits_shape, device=self.device)
            self.buffer_task_id = torch.zeros(
                self.buffer_size, dtype=torch.long, device=self.device
            )

        for i in range(batch_size):
            self.num_seen += 1

            if self.current_size < self.buffer_size:
                idx = self.current_size
                self.current_size += 1
            else:
                idx = np.random.randint(0, self.num_seen)
                if idx >= self.buffer_size:
                    continue

            self.buffer_x[idx] = x[i].detach()
            self.buffer_y[idx] = (
                y[i].detach() if y.dim() == 1 else y[i].argmax().detach()
            )
            self.buffer_logits[idx] = logits[i].detach()
            self.buffer_task_id[idx] = task_id

    def sample(self, batch_size):
        """
        Sample a batch from the buffer.

        Returns:
            x, y, logits, task_ids tensors (or Nones if buffer is empty)
        """
        if self.current_size == 0:
            return None, None, None, None

        indices = np.random.choice(
            self.current_size, size=min(batch_size, self.current_size), replace=False
        )

        return (
            self.buffer_x[indices],
            self.buffer_y[indices],
            self.buffer_logits[indices],
            self.buffer_task_id[indices],
        )

    def __len__(self):
        return self.current_size


class DER_network(Network):
    """
    Dark Experience Replay (DER) network.

    Based on Buzzega et al. (2020) "Dark Experience for General Continual
    Learning: a Strong, Simple Baseline" (NeurIPS 2020).

    Stores samples along with their logits (dark knowledge) in a replay buffer.
    During training, replays old samples and uses MSE loss between current and
    stored logits for knowledge distillation. MSE is computed only on heads that
    were trained at storage time.

    Loss:
        L = L_CE(current_batch) + α * L_MSE(replay_logits, stored_logits)
    """

    def __init__(self, config, name="DER_network"):
        super().__init__(BP_layer, Softplus, Linear, config, name)

        self.alpha = getattr(config, "der_alpha", None) or 0.5
        self.buffer_size = getattr(config, "buffer_size", None) or 500
        self.replay_batch_size = (
            getattr(config, "replay_batch_size", None)
            or getattr(config, "batch_size", None)
            or 64
        )

        self.buffer = ReplayBuffer(self.buffer_size, device=self.device)

    def to(self, *args, **kwargs):
        """Override to() to also move buffer tensors to the correct device."""
        self = super().to(*args, **kwargs)

        device = next(self.parameters()).device
        self.buffer.device = device
        if self.buffer.buffer_x is not None:
            self.buffer.buffer_x = self.buffer.buffer_x.to(device)
        if self.buffer.buffer_y is not None:
            self.buffer.buffer_y = self.buffer.buffer_y.to(device)
        if self.buffer.buffer_logits is not None:
            self.buffer.buffer_logits = self.buffer.buffer_logits.to(device)
        if self.buffer.buffer_task_id is not None:
            self.buffer.buffer_task_id = self.buffer.buffer_task_id.to(device)

        return self

    def forward(self, x):
        """Standard forward pass, storing full output for replay."""
        self.input = x
        self.bzs = x.shape[0]

        full_output = x
        for layer in self.layers:
            full_output = layer(full_output)

        self._full_output = full_output
        x = full_output[:, self.task_masks[self.task_id]]
        self.y_hat = x
        return x

    def backward(self, y):
        """Compute loss with DER replay and backpropagate."""
        loss = self.loss_fn(self.y_hat, y)

        if self.buffer.current_size > 0:
            replay_loss = self._replay_loss()
            if replay_loss is not None:
                loss += replay_loss

        loss.backward()

        y_indices = y.argmax(dim=1) if y.dim() > 1 and y.size(1) > 1 else y
        self.buffer.add(self.input, y_indices, self._full_output, self.task_id)

    def _replay_loss(self):
        """Compute DER replay loss: MSE between current and stored logits."""
        x_replay, y_replay, logits_stored, task_ids = self.buffer.sample(
            self.replay_batch_size
        )

        if x_replay is None:
            return None

        replay_output = x_replay
        for layer in self.layers:
            replay_output = layer(replay_output)

        # Masked MSE: only on heads trained at storage time
        total_loss = 0.0
        count = 0

        for i in range(replay_output.size(0)):
            stored_task = task_ids[i].item()
            num_trained_heads = (stored_task + 1) * self.classes_per_task

            mse = F.mse_loss(
                replay_output[i, :num_trained_heads],
                logits_stored[i, :num_trained_heads],
            )
            total_loss += mse
            count += 1

        if count > 0:
            return self.alpha * (total_loss / count)
        return None

    def complete_task(self, dataloader=None):
        """Called after task completion (interface compatibility)."""
        pass

    def get_buffer_stats(self):
        """Return buffer statistics for monitoring."""
        return {
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer_size,
            "total_seen": self.buffer.num_seen,
        }


class DERpp_network(Network):
    """
    Dark Experience Replay++ (DER++) network.

    Extension of DER that adds a cross-entropy loss on replay samples.

    Loss:
        L = L_CE(current) + α * L_MSE(replay_logits, stored_logits)
                         + β * L_CE(replay_labels)
    """

    def __init__(self, config, name="DERpp_network"):
        super().__init__(BP_layer, Softplus, Linear, config, name)

        self.alpha = getattr(config, "der_alpha", None) or 0.5
        self.beta = getattr(config, "der_beta", None) or 0.5
        self.buffer_size = getattr(config, "buffer_size", None) or 500
        self.replay_batch_size = (
            getattr(config, "replay_batch_size", None)
            or getattr(config, "batch_size", None)
            or 64
        )

        self.buffer = ReplayBuffer(self.buffer_size, device=self.device)

    def to(self, *args, **kwargs):
        """Override to() to also move buffer tensors to the correct device."""
        self = super().to(*args, **kwargs)

        device = next(self.parameters()).device
        self.buffer.device = device
        if self.buffer.buffer_x is not None:
            self.buffer.buffer_x = self.buffer.buffer_x.to(device)
        if self.buffer.buffer_y is not None:
            self.buffer.buffer_y = self.buffer.buffer_y.to(device)
        if self.buffer.buffer_logits is not None:
            self.buffer.buffer_logits = self.buffer.buffer_logits.to(device)
        if self.buffer.buffer_task_id is not None:
            self.buffer.buffer_task_id = self.buffer.buffer_task_id.to(device)

        return self

    def forward(self, x):
        """Standard forward pass."""
        self.input = x
        self.bzs = x.shape[0]

        full_output = x
        for layer in self.layers:
            full_output = layer(full_output)

        self._full_output = full_output
        x = full_output[:, self.task_masks[self.task_id]]
        self.y_hat = x
        return x

    def backward(self, y):
        """Compute loss with DER++ replay and backpropagate."""
        loss = self.loss_fn(self.y_hat, y)

        if self.buffer.current_size > 0:
            replay_loss = self._replay_loss()
            if replay_loss is not None:
                loss += replay_loss

        loss.backward()

        y_indices = y.argmax(dim=1) if y.dim() > 1 and y.size(1) > 1 else y
        self.buffer.add(self.input, y_indices, self._full_output, self.task_id)

    def _replay_loss(self):
        """Compute DER++ replay loss: MSE on logits + CE on labels."""
        x_replay, y_replay, logits_stored, task_ids = self.buffer.sample(
            self.replay_batch_size
        )

        if x_replay is None:
            return None

        replay_output = x_replay
        for layer in self.layers:
            replay_output = layer(replay_output)

        # Masked MSE: only on heads trained at storage time
        total_mse = 0.0
        count = 0

        for i in range(replay_output.size(0)):
            stored_task = task_ids[i].item()
            num_trained_heads = (stored_task + 1) * self.classes_per_task

            mse = F.mse_loss(
                replay_output[i, :num_trained_heads],
                logits_stored[i, :num_trained_heads],
            )
            total_mse += mse
            count += 1

        loss_mse = self.alpha * (total_mse / count) if count > 0 else 0.0
        loss_ce = self.beta * F.cross_entropy(replay_output, y_replay)

        return loss_mse + loss_ce

    def complete_task(self, dataloader=None):
        """Called after task completion."""
        pass

    def get_buffer_stats(self):
        """Return buffer statistics for monitoring."""
        return {
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer_size,
            "total_seen": self.buffer.num_seen,
        }