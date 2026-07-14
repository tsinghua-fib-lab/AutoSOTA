"""
Custom loss functions for VDW models.

This module contains specialized loss functions including:
- MultiTaskLoss: Uncertainty-weighted multi-task regression loss
- SupConLoss: Supervised contrastive loss for embedding learning
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LossWrapper(nn.Module):
    """Wraps a functional loss (like F.mse_loss) in a module so we can store it."""
    def __init__(
        self,
        loss_fn: Callable,
    ) -> None:
        super().__init__()
        self.loss_fn = loss_fn
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fn(pred, target)


LossLike = Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | nn.Module


class MultiTaskLoss(nn.Module):
    """
    Adapts Uncertainty Weighting in a compound loss (Kendall et al. 2018,
    https://arxiv.org/abs/1705.07115) for multi-task regression.
    
    Each task's loss is weighted by a learned uncertainty parameter (log variance).
    The total loss is: sum_i [ exp(-log_var_i) * loss_i + log_var_i ]
    
    This allows the model to automatically balance multiple tasks during training.
    """
    
    def __init__(
        self,
        task_names: tuple[str, ...] | list[str],
        loss_fns: Optional[Dict[str, Callable]] = None,
    ) -> None:
        """
        Initialize multi-task loss with uncertainty weighting.
        
        Args:
            task_names: Names of the tasks, e.g. ['pos', 'vel']
            loss_fns: Dictionary mapping task name to loss function.
                Default: all tasks use euclidean_distance_loss (for 2D vector targets)
        """
        super().__init__()
        self.task_names = task_names
        
        # Import here to avoid circular dependency
        from models.base_module import euclidean_distance_loss
        
        # Default to Euclidean distance loss for all tasks (better for 2D vector targets)
        if loss_fns is None:
            loss_fns = {name: euclidean_distance_loss for name in task_names}
        else:
            # Fill in defaults if only some are provided
            for name in task_names:
                if name not in loss_fns:
                    loss_fns[name] = euclidean_distance_loss
                    
        self.loss_fns = nn.ModuleDict({
            name: self._build_loss_module(fn) for name, fn in loss_fns.items()
        })
        
        # Create learnable log variances for each task
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in task_names
        })

    def _build_loss_module(
        self,
        fn: LossLike,
    ) -> nn.Module:
        """
        Normalize arbitrary callables or nn.Module instances into nn.Module losses.
        """
        if isinstance(fn, nn.Module):
            return fn
        if callable(fn):
            return _LossWrapper(fn)
        raise TypeError(
            "Each entry in loss_fns must be a callable or nn.Module."
        )
    
    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute uncertainty-weighted multi-task loss.
        
        Args:
            preds: Dictionary mapping task name to predictions
            targets: Dictionary mapping task name to targets
            
        Returns:
            tuple of (total_loss, dict of per-task losses)
        """
        total_loss = 0.0
        loss_dict = {}

        for name in self.task_names:
            pred = preds[name]
            target = targets[name]
            base_loss = self.loss_fns[name](pred, target)
            log_var = self.log_vars[name]
            
            # Uncertainty-weighted loss
            weighted = base_loss * torch.exp(-log_var) + log_var
            total_loss = total_loss + weighted

            loss_dict[name] = base_loss.detach().item()
        
        return total_loss, loss_dict


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    https://arxiv.org/abs/2004.11362

    Github reference for a pytorch implementation: 
    https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    
    This loss encourages embeddings of samples with the same label to be similar
    while pushing apart embeddings of samples with different labels.
    
    For each anchor sample, it computes a contrastive loss where:
    - Positives: samples with the same label (excluding the anchor itself)
    - Negatives: samples with different labels
    
    The loss encourages the anchor to be close to all positives relative to
    all negatives in the normalized embedding space.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        learnable_temperature: bool = True,
        min_learnable_temperature: float = 1e-2,
    ) -> None:
        """
        Initialize supervised contrastive loss. Roughly speaking, this loss
        computes the negative mean log probability of the positive samples 
        for each anchor, where the probabilities are dot-product/similarity 
        scores, normalized by a temperature parameter and softmaxed (positive
        samples over all samples).

        Intuition: for positive samples (i.e., with same label),
        - If a similarity score is high, the softmax probability will approach 1,
        so the log probability will approach 0, and so will the loss, -log(p_ij).
        - If a similarity score is low, the softmax probability will 
        approach 0, so the log probability will approach -inf, and the loss, -log(p_ij), will approach inf.

        In this way, this loss encourages the upstream embeddings to have high similarity scores for positive samples.
        
        Args:
            temperature: Temperature scaling parameter for similarity scores.
                Lower values (e.g., 0.07-0.1) create sharper distributions.
            learnable_temperature: If True, optimize the temperature during
                training; otherwise keep it fixed at the provided value.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError("SupCon temperature must be positive.")
        self.learnable_temperature = bool(learnable_temperature)
        self.min_learnable_temperature = min_learnable_temperature
        log_temp = torch.log(torch.tensor(float(temperature)))
        if self.learnable_temperature:
            self.log_temperature = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temperature", log_temp)

    def _get_temperature(self) -> torch.Tensor:
        """
        Return a positive temperature scalar, whether fixed or learnable.
        """
        temperature = torch.exp(self.log_temperature)
        return torch.clamp(temperature, min=self.min_learnable_temperature)
    
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Supervised Contrastive Loss.
        
        Args:
            input: Embedding vectors of shape [batch_size, embedding_dim].
                These will be L2-normalized internally.
            target: Integer labels of shape [batch_size] for each sample.
            
        Returns:
            Scalar loss value.
            
        Example:
            >>> input = torch.randn(32, 128)  # 32 samples, 128-d embeddings
            >>> target = torch.randint(0, 7, (32,))  # 7 classes
            >>> loss_fn = SupConLoss(temperature=0.1)
            >>> loss = loss_fn(input, target)
        """
        device = input.device
        batch_size = input.shape[0]
        
        # Normalize feature vectors to unit sphere
        input = F.normalize(input, dim=-1)
        
        temperature = self._get_temperature()
        # Compute similarity matrix: [batch_size, batch_size]
        sim_matrix = torch.matmul(input, input.T) / temperature
        
        # For numerical stability: subtract max from similarity scores
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # Generate mask to exclude self-comparisons (diagonal elements)
        logits_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)

        # Compute exponentiated similarities excluding self
        exp_logits = torch.exp(logits) * logits_mask

        # Compute log probabilities
        # log(p_ij) = s_ij - log(sum_k(exp(s_ik))), for j != i
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        
        # Positives mask (i.e. same label, excluding self)
        # matches includes self, so we need to mask it out
        # Generate label mask: 1 if same label, 0 otherwise
        target = target.contiguous().view(-1, 1)  # [batch_size, 1]
        matches = torch.eq(target, target.T).to(device)  # [batch_size, batch_size]
        positive_mask = matches * logits_mask
        
        # Mean log probability over positives for each anchor
        # Add small epsilon to avoid division by zero when no positives exist
        num_positives = positive_mask.sum(1)
        log_prob_positives = positive_mask * log_prob
        mean_log_prob_pos = (log_prob_positives).sum(1) / (num_positives + 1e-9)
        
        # Final loss: negative mean of log probs of positives, over all anchors
        loss = -mean_log_prob_pos.mean()
        
        return loss


class SupConCrossEntropyLoss(nn.Module):
    """
    Composite loss that couples supervised contrastive loss with cross entropy
    using uncertainty-weighted multi-task balancing.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        learnable_temperature: bool = True,
        cross_entropy_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        ce_kwargs = cross_entropy_kwargs or {}
        loss_fns: Dict[str, LossLike] = {
            'supcon': SupConLoss(
                temperature=temperature,
                learnable_temperature=learnable_temperature,
            ),
            'cross_entropy': nn.CrossEntropyLoss(**ce_kwargs),
        }
        self._task_names: tuple[str, ...] = ('supcon', 'cross_entropy')
        self.multi_task_loss = MultiTaskLoss(
            task_names=self._task_names,
            loss_fns=loss_fns,
        )

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute uncertainty-weighted SupCon + CrossEntropy loss.
        """
        for key in self._task_names:
            if key not in preds:
                raise KeyError(
                    f"SupConCrossEntropyLoss missing '{key}' in preds dict."
                )
            if key not in targets:
                raise KeyError(
                    f"SupConCrossEntropyLoss missing '{key}' in targets dict."
                )

        return self.multi_task_loss(preds, targets)
