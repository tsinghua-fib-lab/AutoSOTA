"""
Loss function for graph node classification using negative log-likelihood.

This loss function is suitable for multi-class classification tasks on graphs,
such as node classification in citation networks.
"""

import torch.nn as nn


class NLLLoss(nn.Module):
    """
    Negative Log-Likelihood Loss for node classification.

    This loss is used with models that output log probabilities (e.g., using log_softmax).
    It is equivalent to CrossEntropyLoss when the model outputs log probabilities.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.NLLLoss()

    def forward(self, output, target):
        """
        Compute the negative log-likelihood loss.

        Args:
            output: Log probabilities from the model, shape [num_nodes, num_classes]
            target: Ground truth labels, shape [num_nodes]

        Returns:
            loss: Scalar loss value
        """
        return self.loss_fn(output, target)
