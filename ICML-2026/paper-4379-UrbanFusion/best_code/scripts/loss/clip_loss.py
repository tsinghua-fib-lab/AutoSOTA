#!/usr/bin/env python3
"""
Description: Implements the final SMF loss function, combining symmetric
InfoNCE loss with latent modality reconstruction.
"""

import numpy as np
import torch
import torch.nn.functional as F


class CLIPLoss(torch.nn.Module):
    def __init__(
        self,
        initial_temperature: float = 0.07,
        max_logit_scale: float = 100,
        fixed_temperature: float = None,
        normalize: float = True,
        reduction: str = "mean",
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
        reconstruction_loss_weight: float = 0.0,
        mean_vector_path: str = None,
        std_vector_path: str = None,
        scaling_vector_path: str = None,
        number_modalities: int = 5,
        ablation_masking: bool = False,
    ) -> None:
        """
        SMF loss function for UrbanFusion. CLIP style location alignment,
        and latent modality reconstruction.

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
        reduction : str, optional
            Reduction method for the loss, by default 'mean'.
            All PyTorch reduction methods are supported ('mean', 'sum',
            'none').
        label_smoothing_type : str, optional
            Type of label smoothing, by default 'none'.
            Choose between 'none', 'uniform', 'fixed', 'learned'.
            "uniform" label smoothing is applied to all classes.
            "fixed" label smoothing is calculated based on cosine similarity
            of the input embeddings (predcited by modality encoders), with
            fixed temperature.
            "learned" label smoothing is calculated based on cosine similarity
            of the input embeddings (predcited by modality encoders), with
            learnable temperature. Not used in final model.
        label_smoothing_epsilon : float, optional
            Label smoothing factor, by default 0.1. Only used if
            label_smoothing_type is not 'none'. The label smoothing factor
            weights the original cross entropy loss and the smoothed loss.
            Not used in final model.
        temperature_label_smoothing : float, optional
            Temperature for label smoothing, by default 0.07.
            Only used if label_smoothing_type is 'fixed' or 'learned'.
            Not used in final model.
        use_js_divergence : bool, optional
            Whether to use Jensen-Shannon divergence instead of cross entropy
            for the loss calculation, by default False. Only used if if the
            label_smoothing_type is not 'none'. Not used in final model.
        lmbd : float
            Regularization parameter for the cross-covariance matrix.
            Not used in final model.
        u : float
            Regularization parameter for the variance loss.
            Not used in final model.
        v : float
            Regularization parameter for the invariance loss.
            Not used in final model.
        epsilon : float
            Small value to avoid division by zero for variance and invariance
            loss. Not used in final model.
        variance_covariance_reg_bool : bool, optional
            Whether to apply the variance, invariance and cross-covariance
            regularization, by default False. Not used in final model.
        variance_covariance_reg_weight : float, optional
            Weight of the variance, invariance and cross-covariance
            regularization, by default 0.3. Not used in final model.
        reconstruction_loss_weight : float, optional
            Weight of the reconstruction loss, by default 0.0.
            If set to a value greater than 0, the reconstruction loss is
            calculated as the mean squared error between the input and the
            reconstructed output. Cannot be used together with
            variance_covariance_reg_bool.
        mean_vector_path : str, optional
            Path to the mean vector for normalization, by default None.
            If set, the mean vector is loaded from the specified path.
            Used for latent modality reconstruction.
        std_vector_path : str, optional
            Path to the standard deviation vector for normalization, by
            default None. If set, the standard deviation vector is loaded from
            the specified path. Used for latent modality reconstruction.
        scaling_vector_path : str, optional
            Path to the scaling vector for normalization, by default None.
            If set, the scaling vector is loaded from the specified path.
            Used for latent modality reconstruction.
        number_modalities : int
            Number of modalities in the input data.
        ablation_masking : bool, optional
            Whether to apply ablation masking, by default False. Used when
            missing labels for certain modalities during training.
        """

        super().__init__()

        # Store parameters
        self.normalize = normalize
        self.max_logit_scale = max_logit_scale
        self.reduction = reduction
        self.label_smoothing_type = label_smoothing_type
        self.label_smoothing_epsilon = label_smoothing_epsilon
        self.temperature_label_smoothing = temperature_label_smoothing
        self.use_js_divergence = use_js_divergence
        self.lmbd = lmbd
        self.u = u
        self.v = v
        self.epsilon = epsilon
        self.variance_covariance_reg_bool = variance_covariance_reg_bool
        self.variance_covariance_reg_weight = variance_covariance_reg_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.mean_vector_path = mean_vector_path
        self.std_vector_path = std_vector_path
        self.scaling_vector_path = scaling_vector_path
        self.number_modalities = number_modalities
        self.ablation_masking = ablation_masking

        # Initialize temperature
        if fixed_temperature is None:

            # Initialize temperature as learnable parameter
            self.logit_scale = torch.nn.Parameter(
                torch.log(torch.tensor(1 / initial_temperature))
            )
            self.learnable_temperature = True
        else:

            # Initialize temperature as fixed value
            self.logit_scale = torch.log(torch.tensor(1 / fixed_temperature))
            self.learnable_temperature = False

        if self.label_smoothing_type == "learned":
            self.temperature_label_smoothing = torch.nn.Parameter(
                torch.log(torch.tensor(1 / temperature_label_smoothing))
            )
        elif self.label_smoothing_type == "fixed":
            self.temperature_label_smoothing = torch.log(
                torch.tensor(1 / temperature_label_smoothing)
            )

        if self.reconstruction_loss_weight:
            # npy arrays for normalization
            mean_vector = np.load(mean_vector_path)
            std_vector = np.load(std_vector_path)
            scaling_vector = np.load(scaling_vector_path)
            self.register_buffer(
                "mean_vector", torch.tensor(mean_vector, dtype=torch.float32)
            )
            self.register_buffer(
                "std_vector", torch.tensor(std_vector, dtype=torch.float32)
            )
            self.register_buffer(
                "scaling_vector",
                torch.tensor(scaling_vector, dtype=torch.float32),
            )

    @staticmethod
    def linear_combination(x, y, epsilon):
        """
        Linearly combine two tensors, for calculating the label smoothing
        cross entropy loss.

        Parameters
        ----------
        x : torch.Tensor
            A tensor representing the smoothed loss.
        y : torch.Tensor
            A tensor representing the cross entropy loss.
        epsilon : float
            A float representing the smoothing factor.

        Returns
        -------
        torch.Tensor
            Smoothed loss.
        """
        return epsilon * x + (1 - epsilon) * y

    def variance_covariance_reg(
        self,
        output0: torch.Tensor,
        output1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the variance, invariance and cross-covariance loss between two
        tensors.

        Parameters
        ----------
        output0 : torch.Tensor
            Output tensor of the first augmentations.
        output1 : torch.Tensor
            Output tensor of the second augmentations.
        lmbd : float
            Regularization parameter for the cross-covariance matrix.
        u : float
            Regularization parameter for the variance loss.
        v : float
            Regularization parameter for the invariance loss.
        epsilon : float
            Small value to avoid division by zero.

        Returns
        -------
        torch.Tensor
            Variance, invariance and cross-covariance loss.
        """

        bs = output0.size(0)
        emb = output0.size(1)

        std_x = torch.sqrt(output0.var(dim=0) + self.epsilon)
        std_y = torch.sqrt(output1.var(dim=0) + self.epsilon)
        var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(
            F.relu(1 - std_y)
        )

        invar_loss = F.mse_loss(output0, output1)

        xNorm = (output0 - output0.mean(0)) / output0.std(0)
        yNorm = (output1 - output1.mean(0)) / output1.std(0)
        crossCorMat = (xNorm.T @ yNorm) / bs
        cross_loss = (
            (
                crossCorMat * self.lmbd
                - torch.eye(emb, device=output0.device) * self.lmbd
            )
            .pow(2)
            .sum()
        )

        loss = self.u * var_loss + self.v * invar_loss + cross_loss

        return loss

    def js_div_loss(self, pred_logits, target_probs):
        """
        Compute the Jensen-Shannon divergence between two probability
        distributions.

        Parameters
        ----------
        pred_logits : torch.Tensor
            Predicted logits.
        target_probs : torch.Tensor
            Target probabilities.

        Returns
        -------
        torch.Tensor
            Jensen-Shannon divergence.
        """
        # Convert logits to probabilities
        pred_probs = F.softmax(pred_logits, dim=-1)

        # Compute the mixture distribution M
        M = 0.5 * (pred_probs + target_probs)

        # Compute KL divergences
        kl_p_m = F.kl_div(M.log(), target_probs, reduction=self.reduction)
        kl_q_m = F.kl_div(M.log(), pred_probs, reduction=self.reduction)

        # Compute JS divergence
        js_div = 0.5 * (kl_p_m + kl_q_m)

        return js_div

    def get_number_modalities(self, rec_gt: torch.Tensor) -> int:
        """
        Count how many modalities are present (not entirely NaN) in rec_gt.

        Parameters
        ----------
        rec_gt : torch.Tensor
            The normalized concatenated ground truth embedding of shape
            (B, D_total), where missing modalities have been filled with NaNs
            in their respective blocks.
        modalities : list of str
            Names of the modalities, in the same order they were concatenated.
            E.g.: ["coords","SVI","sentinel2","OSM","POI"].

        Returns
        -------
        int
            Number of modalities whose block contains at least one non NaN
            entry.
        """
        # fixed per‑modality dimensions (must match your concatenation order!)
        modal_dims = {
            "coords": 2,
            "SVI": 768,
            "sentinel2": 384,
            "OSM": 2304,
            "POI": 384,
        }

        B, D = rec_gt.shape
        valid = ~torch.isnan(rec_gt)  # shape (B, D_total)

        count = 0
        offset = 0
        for mod in modal_dims.keys():
            dim = modal_dims[mod]
            # slice out this modality’s block and check if any entry is valid
            block = valid[:, offset : offset + dim]
            if block.any():
                count += 1
            offset += dim

        return count

    def prune_nan_columns_from_first(self, rec_gt, rec0, rec1):
        """
        Given one or more tensors of shape (B, D), remove every column j
        for which the first row is NaN.  Returns a tuple of pruned tensors.

        Parameters
        ----------
        rec_gt : torch.Tensor
            The normalized concatenated ground truth embedding of shape
            (B, D_total), where missing modalities have been filled with NaNs
            in their respective blocks.
        rec0 : torch.Tensor
            The normalized concatenated embedding of the first augmentations.
        rec1 : torch.Tensor
            The normalized concatenated embedding of the second augmentations.

        Returns
        -------
        tuple
            A tuple containing the pruned tensors (rec_gt, rec0, rec1).
        """
        keep = ~torch.isnan(rec_gt[0])
        return rec_gt[:, keep], rec0[:, keep], rec1[:, keep]

    def reconstruction_loss(
        self,
        rec_gt: torch.Tensor,
        rec0: torch.Tensor,
        rec1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the reconstruction loss between the ground truth and the
        reconstructed outputs.

        Parameters
        ----------
        rec_gt : torch.Tensor
            Ground truth tensor.
        rec0 : torch.Tensor
            Reconstructed output tensor of the first augmentations.
        rec1 : torch.Tensor
            Reconstructed output tensor of the second augmentations.

        Returns
        -------
        torch.Tensor
            Reconstruction loss.
        """
        # normalize the ground truth
        rec_gt = (
            (rec_gt - self.mean_vector) / self.std_vector
        ) * self.scaling_vector
        if self.ablation_masking is True:
            gt_sel, rec0_sel, rec1_sel = self.prune_nan_columns_from_first(
                rec_gt, rec0, rec1
            )
            number_dims = gt_sel.shape[1]
            number_modalities = self.get_number_modalities(rec_gt)
            rec_loss0 = (
                F.mse_loss(gt_sel, rec0_sel) * number_dims / number_modalities
            )
            rec_loss1 = (
                F.mse_loss(gt_sel, rec1_sel) * number_dims / number_modalities
            )

        else:
            rec_loss0 = (
                F.mse_loss(rec_gt, rec0)
                * self.mean_vector.shape[0]
                / self.number_modalities
            )
            rec_loss1 = (
                F.mse_loss(rec_gt, rec1)
                * self.mean_vector.shape[0]
                / self.number_modalities
            )
        return (rec_loss0 + rec_loss1) / 2

    def forward(
        self,
        output0: torch.Tensor,
        output1: torch.Tensor,
        input_similarity0: torch.Tensor = None,
        input_similarity1: torch.Tensor = None,
        rec_gt: torch.Tensor = None,
        rec0: torch.Tensor = None,
        rec1: torch.Tensor = None,
        queue_output: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the CLIP loss function.

        Parameters
        ----------
        output0 : torch.Tensor
            Output tensor of the first augmentations.
        output1 : torch.Tensor
            Output tensor of the second augmentations.
        input_similarity0 : torch.Tensor
            Similarity matrix of the first augmentations
            modality inputs.
        input_similarity1 : torch.Tensor
            Similarity matrix of the second augmentations
            modality inputs.
        rec_gt : torch.Tensor
            Ground truth tensor for reconstruction loss.
        rec0 : torch.Tensor
            Reconstructed output tensor of the first augmentations.
        rec1 : torch.Tensor
            Reconstructed output tensor of the second augmentations.
        queue_output : torch.Tensor, optional
            Output tensor of the queue (for negative sampling). Not
            used in final model.

        Returns
        -------
        torch.Tensor
            CLIP loss.
        """

        # Get number of samples
        n = output0.shape[0]

        # Optional normalization
        if self.normalize:
            output0 = F.normalize(output0, dim=1)
            output1 = F.normalize(output1, dim=1)
            if queue_output is not None:
                queue_output = F.normalize(queue_output, dim=1)

        # Compute cosine similarity matrix once
        cosine_sim = output0 @ output1.T  # [n, n]

        if queue_output is not None:
            # queue_output: [m, d]  (m = queue size)
            queue_output = queue_output.detach()
            cosine_q0 = output0 @ queue_output.T  # [n, m]
            cosine_q1 = output1 @ queue_output.T  # [n, m]
            # Concatenate current-batch positives + queue negatives
            cosine_sim_i = torch.cat([cosine_sim, cosine_q0], dim=1)
            cosine_sim_t = torch.cat([cosine_sim.T, cosine_q1], dim=1)

        # Compute scaled logits
        if self.learnable_temperature:
            logit_scale = torch.clamp(
                self.logit_scale.exp(), max=self.max_logit_scale
            )
        else:
            logit_scale = self.logit_scale.exp()
        if queue_output is not None:
            logits_i = logit_scale * cosine_sim_i
            logits_t = logit_scale * cosine_sim_t
            K = logits_i.shape[1]
        else:
            logits_i = logit_scale * cosine_sim
            logits_t = logits_i.T
            K = n

        # Apply label smoothing
        if self.label_smoothing_type not in [None, "uniform"]:
            scale_smoothing = torch.clamp(
                self.temperature_label_smoothing.exp(),
                max=self.max_logit_scale,
            )
            input_similarity0 = scale_smoothing * input_similarity0
            input_similarity1 = scale_smoothing * input_similarity1
            input_similarity0 = torch.softmax(input_similarity0, dim=0)
            input_similarity1 = torch.softmax(input_similarity1, dim=0)
        elif self.label_smoothing_type == "uniform":
            input_similarity0 = torch.ones_like(logits_i) / K
            input_similarity1 = torch.ones_like(logits_i) / K

        # Compute loss
        if self.label_smoothing_type is None:
            labels = torch.arange(n, device=output0.device)

            loss_i = F.cross_entropy(logits_i, labels, reduction="none")
            loss_t = F.cross_entropy(logits_t, labels, reduction="none")
            if self.reduction == "mean":
                loss_i = loss_i.mean()
                loss_t = loss_t.mean()
            elif self.reduction == "sum":
                loss_i = loss_i.sum()
                loss_t = loss_t.sum()
        else:
            if queue_output is not None:
                labels = F.one_hot(
                    torch.arange(n, device=output0.device), num_classes=K
                ).float()
            else:
                labels = torch.eye(n, device=output0.device)

            # Get smoothed labels
            labels0 = self.linear_combination(
                input_similarity0, labels, self.label_smoothing_epsilon
            )
            labels1 = self.linear_combination(
                input_similarity1, labels, self.label_smoothing_epsilon
            )

            if self.use_js_divergence:
                loss_i = self.js_div_loss(
                    logits_i, labels1, reduction=self.reduction
                )
                loss_t = self.js_div_loss(
                    logits_t, labels0, reduction=self.reduction
                )
            else:
                loss_i = F.cross_entropy(
                    logits_i, labels1, reduction=self.reduction
                )
                loss_t = F.cross_entropy(
                    logits_t, labels0, reduction=self.reduction
                )

        loss = (loss_i + loss_t) / 2

        if self.variance_covariance_reg_bool:
            loss = (
                1 - self.variance_covariance_reg_weight
            ) * loss + self.variance_covariance_reg_weight * self.variance_covariance_reg(
                output0, output1
            )

        if self.reconstruction_loss_weight > 0:
            rec_loss = self.reconstruction_loss(rec_gt, rec0, rec1)
            loss = (
                1 - self.reconstruction_loss_weight
            ) * loss + self.reconstruction_loss_weight * rec_loss

        return loss
