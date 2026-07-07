"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors"

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = t.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + t.log(1 + self.params[i] ** 2)
        return loss_sum


class GradientClippedMultiTaskLoss(nn.Module):
    def __init__(self, clip_ratio=1.0, clip_type='scale'):
        """
        Initialize the GradientClippedMultiTaskLoss class.
        
        Args:
            clip_ratio (float): The ratio for clipping auxiliary task gradients relative to main task.
                               For example, 1.0 means auxiliary task gradients can be at most equal to main task.
            clip_type (str): Type of gradient operation ('clip' or 'scale').
        """
        super(GradientClippedMultiTaskLoss, self).__init__()
        self.clip_ratio = clip_ratio
        self._named_params_cache = None
        if clip_type == 'cilp':
            self._forward = self.gradCilp
        else:
            self._forward = self.gradScale

    def _get_trainable_params(self, model):
        """
        Get list of parameters that require gradients (with caching mechanism).
        
        Args:
            model (torch.nn.Module): The model whose parameters we want to retrieve
            
        Returns:
            List of trainable parameters
        """
        if self._named_params_cache is not None:
            return self._named_params_cache

        params = [param for _, param in model.named_parameters() if param.requires_grad]

        self._named_params_cache = params
        return params

    def reset_cache(self):
        """
        Call when model structure changes to reset parameter cache.
        """
        self._named_params_cache = None

    def _compute_grad_norm(self, loss, params):
        """
        Compute gradient norm for specified loss with respect to parameters without breaking computation graph.
        
        Args:
            loss (torch.Tensor): The loss tensor to compute gradients for
            params (List[torch.Tensor]): List of parameters to compute gradients with respect to
            
        Returns:
            torch.Tensor: The L2 norm of gradients
        """
        grads = t.autograd.grad(
            loss, params, retain_graph=True, create_graph=False, allow_unused=True
        )

        valid_grads = [g.detach() for g in grads if g is not None]

        if not valid_grads:
            return t.tensor(0.0, device=loss.device)

        total_norm = t.norm(t.stack([t.norm(g) for g in valid_grads]))
        return total_norm

    def gradCilp(self, losses, model):
        """
        Forward function using gradient clipping approach.

        Args:
            losses (list or tensor): List containing loss terms for each task. losses[0] must be the main task.
            model (torch.nn.Module): Model used for computing gradients.

        Returns:
            torch.Tensor: Total computed loss after applying gradient clipping
        """
        main_loss = losses[0]
        aux_losses = losses[1:]
        params = self._get_trainable_params(model)

        main_grad_norm = self._compute_grad_norm(main_loss, params)

        target_norm = main_grad_norm * self.clip_ratio

        total_loss = main_loss

        for aux_loss in aux_losses:
            if aux_loss < 1e-6:
                continue

            aux_grad_norm = self._compute_grad_norm(aux_loss, params)

            if aux_grad_norm > target_norm:
                scale = target_norm / (aux_grad_norm + 1e-8)
                total_loss = total_loss + (scale * aux_loss)
            else:
                total_loss = total_loss + aux_loss
        return total_loss

    def gradScale(self, losses, model):
        """
        Forward function using gradient scaling approach.

        Args:
            losses (list or tensor): List containing loss terms for each task. losses[0] must be the main task.
            model (torch.nn.Module): Model used for computing gradients.

        Returns:
            torch.Tensor: Total computed loss after applying gradient scaling
        """
        main_loss = losses[0]
        aux_losses = losses[1:]
        params = self._get_trainable_params(model)

        main_grad_norm = self._compute_grad_norm(main_loss, params)

        if self.clip_ratio >= 1:
            target_norm = main_grad_norm
            total_loss = main_loss / self.clip_ratio
        else:
            target_norm = main_grad_norm * self.clip_ratio
            total_loss = main_loss

        for aux_loss in aux_losses:
            if aux_loss < 1e-6:
                continue
                
            aux_grad_norm = self._compute_grad_norm(aux_loss, params)
            
            scale = target_norm / (aux_grad_norm + 1e-8)
            total_loss = total_loss + (scale * aux_loss)
        return total_loss

    def forward(self, losses, model):
        """
        Forward pass for the module.
        
        Args:
            losses (list or tensor): List containing loss terms for each task
            model (torch.nn.Module): Model used for computing gradients
            
        Returns:
            torch.Tensor: Combined loss based on selected method
        """
        return self._forward(losses, model)

class HaarWaveletLayer(nn.Module):
    """
    Haar discrete wavelet transform (DWT) implemented using Conv1d.
    Properties: orthogonal, parameter-free, differentiable, energy-preserving.
    """

    def __init__(self, num_features):
        """
        Initialize the HaarWaveletLayer.

        Args:
            num_features (int): Number of input features/channels
        """
        super().__init__()
        self.num_features = num_features

        # Haar wavelet kernels
        # Low Pass (Approximation): [1/sqrt(2), 1/sqrt(2)]
        # High Pass (Detail):       [1/sqrt(2), -1/sqrt(2)]
        # Coefficient 0.70710678 ensures energy preservation (Unitary Transform)
        scale = 1.0 / t.sqrt(t.tensor(2.0))

        # Construct convolution kernels: [Out_Channels, In_Channels, Kernel_Size]
        # Using Group Conv to process each Feature independently
        # Low Pass Kernel
        dec_lo = t.tensor([1.0, 1.0]) * scale
        self.register_buffer('lo_filter',
                             dec_lo.view(1, 1, 2).repeat(num_features, 1, 1))

        # High Pass Kernel
        dec_hi = t.tensor([1.0, -1.0]) * scale
        self.register_buffer('hi_filter',
                             dec_hi.view(1, 1, 2).repeat(num_features, 1, 1))

    def forward(self, x):
        """
        Forward pass of Haar Wavelet Layer.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Seq_len, Features]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Approximation coefficients [Batch, Seq_len/2, Features]
                - Detail coefficients [Batch, Seq_len/2, Features]
                
        Note:
            Output sequence length is halved (Seq_len / 2)
        """
        # Permute dimensions: [B, F, S]
        x_in = x.permute(0, 2, 1)

        # Padding: add one zero if length is odd
        if x_in.shape[-1] % 2 != 0:
            x_in = F.pad(x_in, (0, 1), mode='reflect')

        # DWT (Stride=2 ensures orthogonality)
        # groups=num_features ensures each channel is processed independently
        approx = F.conv1d(x_in, self.lo_filter, stride=2, groups=self.num_features)
        detail = F.conv1d(x_in, self.hi_filter, stride=2, groups=self.num_features)

        # Permute back: [B, S/2, F]
        return approx.permute(0, 2, 1), detail.permute(0, 2, 1)

class GWNRLoss(nn.Module):

    def __init__(self,
                 bandwidths,
                 num_features: int,
                 alphaGR=1,
                 flat_threshold: float = 1e-5):
        """
        Initialize the Gradient-Weighted Noise Regularization Loss module.
        
        Args:
            bandwidths: Bandwidths for MMD calculation
            num_features (int): Number of input features
            alphaGR (float): Gradient clipping ratio for GradientClippedMultiTaskLoss
            flat_threshold (float): Threshold for determining activity in signals
        """
        super().__init__()

        self.num_features = num_features
        self.flat_threshold = flat_threshold
        self.GradClippedLoss = GradientClippedMultiTaskLoss(clip_ratio=alphaGR)
        self.wavelet = HaarWaveletLayer(num_features).to('cuda')

        if not isinstance(bandwidths, (list, tuple)):
            bandwidths = [bandwidths]
        self.bandwidths = [float(b) for b in bandwidths]

        self.eps = 1e-8
        self.max_mmd_samples = 1024

    @t.no_grad()
    def _get_activity_mask(self, x: t.Tensor) -> t.Tensor:
        """
        Generate an activity mask based on differences between consecutive time steps,
        then apply dilation to expand active regions around transitions.
        
        Args:
            x (t.Tensor): Input tensor of shape [Batch, Sequence, Features]
            
        Returns:
            t.Tensor: Dilated activity mask of same shape as input [Batch, Sequence, Features]
        """
        # Basic differential masking
        diff = x[:, 1:, :] - x[:, :-1, :]
        diff = t.cat([t.zeros_like(x[:, :1, :]), diff], dim=1)
        is_active = (diff.abs() > self.flat_threshold).float()

        # Mask dilation - spread active state forward and backward in time
        # This ensures regions at the beginning/end of fluctuations are still treated as active
        # [B, S, F] -> [B, F, S]
        mask_permuted = is_active.permute(0, 2, 1)

        # Use MaxPool1d for dilation (kernel_size=5 spreads 2 positions left/right)
        dilated_mask = F.max_pool1d(
            mask_permuted,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # [B, F, S] -> [B, S, F]
        return dilated_mask.permute(0, 2, 1)

    def _mmd_loss(self, detail: t.Tensor, mask: t.Tensor):
        """
        Compute MMD loss in parallel across all channels
        
        Args:
            detail (t.Tensor): Detail coefficients from wavelet decomposition [B, S_half, F]
            mask (t.Tensor): Activity mask [B, S_half, F]
            
        Returns:
            t.Tensor: Scalar MMD loss value
        """
        B, S_half, F = detail.shape
        K = self.max_mmd_samples

        # Flatten batch and time dimensions -> [N_total, F]
        flat_detail = detail.reshape(-1, F)
        flat_mask = mask.reshape(-1, F)

        # Vectorized random sampling
        # Add random noise to mask and use topk for selection
        # mask=1 positions get scores in range (1, 2], mask=0 get score 0
        rand_noise = t.rand_like(flat_detail)
        scores = flat_mask * (rand_noise + 1.0)

        # Get top K indices per column
        k_actual = min(K, flat_detail.shape[0])
        top_scores, indices = t.topk(scores, k=k_actual, dim=0)  # indices: [K, F]

        # Extract data using indices -> [K, F]
        sampled_detail = t.gather(flat_detail, 0, indices)

        # Generate corresponding mask -> [K, F]
        # Only scores > 1.0 are true active points
        sampled_mask = (top_scores > 1.0).float()

        # Filter out features with too few active points (< 10)
        valid_counts = sampled_mask.sum(dim=0)  # [F]
        feature_is_valid = valid_counts >= 10
        if feature_is_valid.sum() < 1:
            return t.tensor(0.0, device=detail.device)

        # Vectorized standardization
        # Calculate mean and variance using only mask=1 data
        safe_counts = valid_counts.clamp(min=1.0)

        # Mean: [F]
        mean = (sampled_detail * sampled_mask).sum(dim=0) / safe_counts

        # Var: [F]
        diff = (sampled_detail - mean.unsqueeze(0)) * sampled_mask
        diff = t.where(sampled_mask.bool(), diff, t.zeros_like(diff))
        var = diff.pow(2).sum(dim=0) / safe_counts
        std = t.sqrt(var) + 1e-6

        # Normalize -> [K, F]
        norm_detail = diff / std.unsqueeze(0)

        # Ensure padding areas are 0
        norm_detail = t.where(sampled_mask.bool(), norm_detail, t.zeros_like(norm_detail))

        # Generate target distribution (Standard Gaussian) -> [K, F]
        target = t.randn_like(norm_detail) * sampled_mask

        # Batch matrix operations for MMD computation
        X = norm_detail.t().unsqueeze(2)  # [F, K, 1]
        Y = target.t().unsqueeze(2)  # [F, K, 1]

        # Build joint mask matrix [F, K, K]
        M = sampled_mask.t().unsqueeze(2)  # [F, K, 1]
        joint_mask = M @ M.transpose(1, 2)  # [F, K, K]

        # Compute normalization factors (denominator: N^2)
        norm_factors = joint_mask.sum(dim=(1, 2)).clamp(min=1.0)

        # Compute squared Euclidean distances
        dist_xx = (X - X.transpose(1, 2)).pow(2)
        dist_yy = (Y - Y.transpose(1, 2)).pow(2)
        dist_xy = (X - Y.transpose(1, 2)).pow(2)

        loss_per_feature = 0.0

        for b in self.bandwidths:
            scale = 2 * (b ** 2)
            # RBF Kernel + Masking
            k_xx = (t.exp(-dist_xx / scale) * joint_mask).sum(dim=(1, 2))
            k_yy = (t.exp(-dist_yy / scale) * joint_mask).sum(dim=(1, 2))
            k_xy = (t.exp(-dist_xy / scale) * joint_mask).sum(dim=(1, 2))

            # MMD^2 = E[k(x,x)] + E[k(y,y)] - 2E[k(x,y)]
            loss_per_feature += (k_xx + k_yy - 2 * k_xy) / norm_factors

        # Final averaging over valid features
        final_loss = t.where(feature_is_valid, loss_per_feature, t.zeros_like(loss_per_feature)).sum() / (
                feature_is_valid.sum() + 1e-8)
        return final_loss

    def _spectral_flatness_loss(self, residuals: t.Tensor, active_mask: t.Tensor):
        """
        Compute spectral flatness loss using entropy of normalized power spectral density.
        
        Args:
            residuals (t.Tensor): Residual tensor [Batch, Sequence, Features]
            active_mask (t.Tensor): Activity mask [Batch, Sequence, Features]
            
        Returns:
            t.Tensor: Scalar spectral flatness loss value
        """

        valid_counts = active_mask.sum(dim=1)

        # Minimum points required for FFT (Nyquist-Shannon sampling theorem)
        min_spectral_points = 32

        # Validity mask: only compute spectral loss for sequences with sufficient active points
        is_valid = (valid_counts >= min_spectral_points)

        # Return 0 if no samples are valid in current batch
        if not is_valid.any():
            return t.tensor(0.0, device=residuals.device)

        B, seq_len, F = residuals.shape

        # Preprocessing: apply mask
        masked_residuals = residuals * active_mask

        # Remove mean (centering)
        # Only calculate mean of active points to avoid interference from dead zones
        active_counts = active_mask.sum(dim=1, keepdim=True) + 1e-8
        active_mean = masked_residuals.sum(dim=1, keepdim=True) / active_counts

        # Subtract mean and reapply mask (dead zones become negative mean, so zero them again)
        centered_residuals = (masked_residuals - active_mean) * active_mask

        # Compute FFT & Power Spectral Density (PSD)
        fft_res = t.fft.rfft(centered_residuals.permute(0, 2, 1), n=seq_len, dim=-1)
        psd = fft_res.abs().pow(2)

        # Safe normalization
        psd_sum = psd.sum(dim=-1, keepdim=True)
        # Prevent division by 0
        normalized_psd = psd / (psd_sum + 1e-8)

        # Create uniform distribution as substitute [1, 1, Freqs]
        num_freqs = normalized_psd.shape[-1]
        dummy_flat_psd = t.ones_like(normalized_psd) / num_freqs

        # Broadcast is_valid [B, F] -> [B, F, 1] to match PSD dimensions
        is_valid_expanded = is_valid.unsqueeze(-1)

        # Use real PSD if valid, otherwise use substitute PSD
        safe_psd = t.where(is_valid_expanded, normalized_psd, dummy_flat_psd)

        # Compute spectral entropy
        # Add eps to prevent log(0)
        log_psd = t.log(safe_psd + 1e-8)
        negative_entropy_val = (safe_psd * log_psd).sum(dim=-1)

        ideal_entropy = t.log(t.tensor(float(num_freqs), device=residuals.device))
        spectral_loss = negative_entropy_val + ideal_entropy

        # Average
        return spectral_loss.sum() / is_valid.sum()

    def forward(self, reconstruction: t.Tensor, original_data: t.Tensor, model) -> t.Tensor:
        """
        Forward pass to compute the combined loss.
        
        Args:
            reconstruction (t.Tensor): Reconstructed data from the model
            original_data (t.Tensor): Original input data
            model: Model being trained, passed to GradientClippedLoss
            
        Returns:
            t.Tensor: Combined loss value incorporating MSE, spectral flatness, and MMD losses
        """
        # 1. Compute raw residuals
        residuals = original_data - reconstruction
        # 2. Get dilated activity mask
        # active_mask=1 (active/edge), active_mask=0 (absolute dead zone)
        active_mask = self._get_activity_mask(original_data)

        loss_mse = residuals.pow(2).mean()

        loss_whiteness = self._spectral_flatness_loss(residuals, active_mask)

        # 2. Wavelet decomposition (DWT)
        # approx: low frequency (Trend), detail: high frequency (Noise)
        # Both have shapes [B, S/2, F]
        # Also downsample mask to match wavelet output dimensions
        # Mask is low frequency signal, downsample with MaxPool or AvgPool
        # [B, S, F] -> [B, F, S] -> Pool -> [B, F, S/2] -> [B, S/2, F]
        mask_downsampled = F.max_pool1d(active_mask.permute(0, 2, 1), kernel_size=2, stride=2).permute(0, 2, 1)

        _, res_noise = self.wavelet(residuals)
        # --- Loss B: Normalized Noise MMD (Shape Constraint) ---
        loss_noise_mmd = self._mmd_loss(res_noise, mask_downsampled)

        # Combine all losses using gradient-clipped combination
        total_loss = self.GradClippedLoss([loss_mse, loss_whiteness, loss_noise_mmd], model)

        return total_loss