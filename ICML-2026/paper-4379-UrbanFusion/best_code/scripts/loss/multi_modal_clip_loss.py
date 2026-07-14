#!/usr/bin/env python3
"""
Description: Implements the CLIP loss function, a symmetric variant of the
InfoNCE loss for multi-modal contrastive learning. The loss containes a
learnable temperature parameter and supports different aggregation methods.
Applies the loss to each modality and aggregates them. Not used in final model.
"""

import torch

from scripts.loss.clip_loss import CLIPLoss


class MultiModalCLIPLoss(torch.nn.Module):
    def __init__(
        self,
        initial_temperature: float = 0.07,
        max_logit_scale: float = 100,
        fixed_temperature: float = None,
        normalize: float = True,
        reduction: str = "mean",
        modality_aggregation: str = "mean",
        modality_weights: list = None,
        label_smoothing_type: str = None,
        label_smoothing_epsilon: float = 0.1,
        temperature_label_smoothing: float = 0.07,
        use_js_divergence: bool = False,
        lmbd: float = 5e-3,
        u: float = 1,
        v: float = 1,
        epsilon: float = 1e-3,
        variance_covariance_reg_bool: bool = False,
        variance_covariance_reg_weight: float = 0.3,
    ) -> None:
        """
        Multi-modal CLIP loss function, a symmetric variant of the InfoNCE
        loss. Calculates the CLIP loss for each modality and aggregates them.

        Parameters
        ----------
        initial_temperature : float, optional
            Initial temperature for the logits, by default 0.07.
        max_logit_scale : float, optional
            Maximum logit scale, by default 100.
        fixed_temperature : float, optional
            Fixed temperature for the logits, by default None.
            If set, the temperature is fixed and not learnable.
        normalize : float, optional
            Whether to normalize the input features, by default True.
            Only set to False if the input features are already normalized.
        modality_reduction : str, optional
            Reduction method for individual modality losses, by default 'mean'.
        modality_aggregation : str, optional
            Aggregation method for the modality losses, by default 'mean'.
        modality_weights : list, optional
            List of weights for each modality, by default None.
        label_smoothing_type : str, optional
            Type of label smoothing, by default 'none'.
            Choose between 'none', 'uniform', 'fixed', 'learned'.
            "uniform" label smoothing is applied to all classes.
            "fixed" label smoothing is calculated based on cosine similarity
            of the input embeddings (predcited by modality encoders), with
            fixed temperature.
            "learned" label smoothing is calculated based on cosine similarity
            of the input embeddings (predcited by modality encoders), with
            learnable temperature.
        label_smoothing_epsilon : float, optional
            Label smoothing factor, by default 0.1. Only used if
            label_smoothing_type is not 'none'. The label smoothing factor
            weights the original cross entropy loss and the smoothed loss.
        temperature_label_smoothing : float, optional
            Temperature for label smoothing, by default 0.07.
            Only used if label_smoothing_type is 'fixed' or 'learned'.
        use_js_divergence : bool, optional
            Whether to use Jensen-Shannon divergence instead of cross entropy
            for the loss calculation, by default False. Only used if if the
            label_smoothing_type is not 'none'.
        lmbd : float
            Regularization parameter for the cross-covariance matrix.
        u : float
            Regularization parameter for the variance loss.
        v : float
            Regularization parameter for the invariance loss.
        epsilon : float
            Small value to avoid division by zero for variance and invariance
            loss.
        variance_covariance_reg_bool : bool, optional
            Whether to apply variance-covariance regularization, by default
            False.
        variance_covariance_reg_weight : float, optional
            Weight of the variance-covariance regularization, by default 0.3.
        """

        super().__init__()

        self.modality_weights = modality_weights
        self.modality_aggregation = modality_aggregation
        self.label_smoothing_type = label_smoothing_type
        self.reduction = reduction

        self.clip_loss = CLIPLoss(
            initial_temperature=initial_temperature,
            max_logit_scale=max_logit_scale,
            fixed_temperature=fixed_temperature,
            normalize=normalize,
            reduction=self.reduction,
            label_smoothing_type=label_smoothing_type,
            label_smoothing_epsilon=label_smoothing_epsilon,
            temperature_label_smoothing=temperature_label_smoothing,
            use_js_divergence=use_js_divergence,
            lmbd=lmbd,
            u=u,
            v=v,
            epsilon=epsilon,
            variance_covariance_reg_bool=variance_covariance_reg_bool,
            variance_covariance_reg_weight=variance_covariance_reg_weight,
        )

    def forward(
        self,
        output0: torch.Tensor,
        output1: torch.Tensor,
        input_similarity0: torch.Tensor = None,
        input_similarity1: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the CLIP loss function.

        Parameters
        ----------
        output0 : torch.Tensor
            Output tensor of the first augmentations.
        output1 : torch.Tensor
            Output tensor of the second augmentations.

        Returns
        -------
        torch.Tensor
            CLIP loss.
        """
        num_modalities = output0.size(1)
        losses = []

        for i in range(num_modalities):
            if self.label_smoothing_type not in [None, "uniform"]:
                loss = self.clip_loss(
                    output0[:, i],
                    output1[:, i],
                    input_similarity0[:, i],
                    input_similarity1[:, i],
                )
            else:
                loss = self.clip_loss(output0[:, i], output1[:, i])
            if self.modality_weights is not None:
                loss = loss * self.modality_weights[i]
            losses.append(loss)

        if self.modality_aggregation == "mean":
            return torch.mean(torch.stack(losses))
        elif self.modality_aggregation == "sum":
            return torch.sum(torch.stack(losses))
        else:
            raise ValueError(
                f"Invalid modality aggregation method: "
                f"{self.modality_aggregation}"
            )
