"""DropoutTS: Sample-Adaptive Dropout for Time Series Forecasting

This module implements sample-adaptive dropout that dynamically adjusts dropout rates
based on noise levels estimated via frequency-domain analysis. The noise estimation
uses Spectral Flatness Measure (SFM) as a physical anchor to compute adaptive thresholds.
"""

import random
from typing import Optional, Tuple, Dict, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DropoutTSContext:
    """Context manager for storing and retrieving dropout rates during forward pass.
    
    This allows SampleAdaptiveDropout instances to automatically access
    the computed dropout rates without requiring explicit parameter passing.
    """
    _current_rates: Optional[torch.Tensor] = None
    
    @classmethod
    def set_rates(cls, rates: Optional[torch.Tensor]) -> None:
        """Set the current dropout rates for all SampleAdaptiveDropout layers.
        
        Args:
            rates: Dropout rates tensor of shape [batch_size] or None.
        """
        cls._current_rates = rates
    
    @classmethod
    def get_rates(cls) -> Optional[torch.Tensor]:
        """Get the current dropout rates.
        
        Returns:
            Dropout rates tensor or None if not set.
        """
        return cls._current_rates
    
    @classmethod
    def clear(cls) -> None:
        """Clear the current dropout rates."""
        cls._current_rates = None


class SampleAdaptiveDropout(nn.Module):
    """Dropout layer with sample-specific dropout rates.
    
    Supports per-sample dropout rates, allowing different dropout probabilities
    for different samples in the same batch. Falls back to standard dropout
    if sample-specific rates are not available.
    """
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """Initialize SampleAdaptiveDropout.
        
        Args:
            p: Default dropout probability (used when sample-specific rates unavailable).
            inplace: Whether to perform in-place operation.
        """
        super().__init__()
        self.p = p
        self.inplace = inplace
        
    def forward(self, inputs: torch.Tensor, sample_dropout_rates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply dropout with per-sample rates or default rate.
        
        Args:
            inputs: Input tensor of shape [batch_size, ...].
            sample_dropout_rates: Optional per-sample dropout rates [batch_size].
                If None, tries to get from DropoutTSContext.
                If still None, falls back to standard dropout with self.p.
        
        Returns:
            Output tensor with dropout applied.
        """
        if not self.training:
            return inputs
        
        # Try to get dropout_rates from context if not explicitly provided
        if sample_dropout_rates is None:
            sample_dropout_rates = DropoutTSContext.get_rates()
        
        # If still None, use standard dropout (compatible with nn.Dropout)
        if sample_dropout_rates is None:
            return F.dropout(inputs, p=self.p, training=self.training, inplace=self.inplace)
        
        batch_size = inputs.shape[0]
        if sample_dropout_rates.dim() == 0:
            sample_dropout_rates = sample_dropout_rates.unsqueeze(0)
        
        rates_batch_size = sample_dropout_rates.shape[0]
        
        # Handle batch size mismatch (e.g., when PatchEmbedding reshapes batch dimension)
        if rates_batch_size != batch_size:
            if batch_size % rates_batch_size == 0:
                # If batch_size is a multiple of rates_batch_size, repeat the rates
                # This handles cases like PatchEmbedding: [B, T, C] -> [B*C, ...]
                repeat_factor = batch_size // rates_batch_size
                sample_dropout_rates = sample_dropout_rates.repeat_interleave(repeat_factor, dim=0)
            elif rates_batch_size % batch_size == 0:
                # If rates_batch_size is a multiple of batch_size, take first batch_size elements
                sample_dropout_rates = sample_dropout_rates[:batch_size]
            else:
                # If neither is a multiple, use broadcasting with mean rate
                mean_rate = sample_dropout_rates.mean()
                sample_dropout_rates = torch.full(
                    (batch_size,), 
                    mean_rate.item(), 
                    device=sample_dropout_rates.device, 
                    dtype=sample_dropout_rates.dtype
                )
        
        # Reshape to broadcast across all dimensions except batch
        shape = [batch_size] + [1] * (inputs.dim() - 1)
        sample_dropout_rates = sample_dropout_rates.view(shape)
        
        # Use Straight-Through Estimator (STE) to maintain gradients
        rand_tensor = torch.rand_like(inputs)
        binary_mask = (rand_tensor > sample_dropout_rates).float()
        expect_mask = 1.0 - sample_dropout_rates
        mask = binary_mask + (expect_mask - expect_mask.detach())
        scale = 1.0 - sample_dropout_rates + 1e-6
        
        # Compute dropped_inputs, use inplace operation if inplace=True
        if self.inplace:
            dropped_inputs = inputs.clone()
            dropped_inputs.mul_(mask / scale)
        else:
            dropped_inputs = inputs * mask / scale
        
        return dropped_inputs


class SFMAnchoredRRF(nn.Module):
    """Frequency-domain filter using Spectral Flatness Measure (SFM) as dynamic anchor.
    
    This filter computes a dynamic threshold based on SFM, which serves as a physical
    anchor to prevent feature collapse. Higher SFM (more noise) leads to higher
    threshold, filtering more frequency components.
    
    The threshold is computed as: dynamic_threshold = sigmoid(SFM_Scale * SFM + SFM_Bias),
    where SFM_Scale and SFM_Bias are learnable parameters.
    """
    
    def __init__(self, n_freqs: int, num_features: int = 1, init_alpha: float = 10.0, use_sfm_anchor: bool = True):
        """Initialize SFMAnchoredRRF.
        
        Args:
            n_freqs: Number of frequency bins (unused, kept for compatibility).
            num_features: Number of features/channels.
            init_alpha: Initial value for sigmoid sharpness parameter.
            use_sfm_anchor: If True, use SFM as physical anchor. If False, use purely learnable threshold.
        """
        super().__init__()
        self.use_sfm_anchor = use_sfm_anchor
        self.sfm_scale = nn.Parameter(torch.ones(1, num_features, 1))
        self.sfm_bias = nn.Parameter(torch.zeros(1, num_features, 1))
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.softplus = nn.Softplus()
        
        # For ablation: purely learnable threshold (when use_sfm_anchor=False)
        if not self.use_sfm_anchor:
            self.learnable_threshold = nn.Parameter(torch.zeros(1, num_features, 1))

    def _compute_sfm(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """Compute Spectral Flatness Measure (SFM).
        
        SFM = Geometric Mean / Arithmetic Mean of power spectrum.
        Higher SFM indicates flatter spectrum (more noise-like).
        Lower SFM indicates peaky spectrum (more signal-like).
        
        Args:
            amplitudes: Frequency magnitude tensor [batch, features, freqs].
        
        Returns:
            SFM scores [batch, features, 1] in range [0, 1].
        """
        power_spec = amplitudes.pow(2) + 1e-8
        geo_mean = torch.exp(torch.mean(torch.log(power_spec), dim=-1, keepdim=True))
        ari_mean = torch.mean(power_spec, dim=-1, keepdim=True)
        return geo_mean / (ari_mean + 1e-8)

    def forward(
        self, 
        fft_magnitude: torch.Tensor, 
        norm_amp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate frequency mask based on SFM-anchored dynamic threshold.
        
        Args:
            fft_magnitude: FFT magnitude [batch, features, freqs] (for SFM computation).
            norm_amp: Normalized amplitude [batch, features, freqs] in [0, 1] (for mask comparison).
        
        Returns:
            Tuple of (mask, dynamic_threshold, sfm_score):
                - mask: Soft frequency mask [batch, features, freqs] in [0, 1]
                - dynamic_threshold: Threshold values [batch, features, 1] in [0, 1]
                - sfm_score: SFM scores [batch, features, 1] in [0, 1] (or None if not using SFM)
        """
        # Compute SFM score as physical anchor
        sfm_score = self._compute_sfm(fft_magnitude)
        
        # Ablation: Use SFM anchor vs purely learnable threshold
        if self.use_sfm_anchor:
            # Ours: SFM-guided dynamic threshold with learnable scale and bias
            dynamic_threshold = torch.sigmoid(self.sfm_scale * sfm_score + self.sfm_bias)
        else:
            # Ablation: Purely learnable threshold (no physical anchor)
            dynamic_threshold = torch.sigmoid(self.learnable_threshold)
            # Expand to match batch size
            batch_size = fft_magnitude.shape[0]
            dynamic_threshold = dynamic_threshold.expand(batch_size, -1, -1)
        
        # Generate soft mask using sigmoid
        alpha = self.softplus(self.alpha)  # Ensure positive sharpness
        mask = torch.sigmoid(alpha * (norm_amp - dynamic_threshold))
        
        return mask, dynamic_threshold, sfm_score


class NoiseScorer(nn.Module):
    """Compute noise scores via frequency-domain reconstruction error.
    
    Uses SFM-anchored RRF filter to separate signal from noise in frequency domain,
    then computes reconstruction error as noise score. The entire computation is
    differentiable, allowing gradients from task loss to flow back to RRF parameters.
    """
    
    def __init__(
        self, 
        seq_len: int, 
        num_features: int = 1, 
        init_alpha: float = 10.0, 
        detrend_method: str = 'robust_ols',
        use_instance_norm: bool = True,
        use_sfm_anchor: bool = True
    ):
        """Initialize NoiseScorer.
        
        Args:
            seq_len: Sequence length.
            num_features: Number of features/channels.
            init_alpha: Initial value for RRF sigmoid sharpness parameter.
            detrend_method: Detrending method. Options:
                - 'none': No detrending
                - 'simple': Simple end-to-end linear detrending
                - 'robust_ols': Robust OLS detrending (Ours)
            use_instance_norm: Whether to apply instance-level normalization to FFT amplitudes.
            use_sfm_anchor: Whether to use SFM as physical anchor in RRF filter.
        """
        super().__init__()
        self.n_freqs = seq_len // 2 + 1
        self.detrend_method = detrend_method
        self.use_instance_norm = use_instance_norm
        self.rrf_filter = SFMAnchoredRRF(
            self.n_freqs, 
            num_features=num_features, 
            init_alpha=init_alpha,
            use_sfm_anchor=use_sfm_anchor
        )
        
        # Prepare detrending buffers for robust_ols method
        if self.detrend_method == 'robust_ols':
            time_steps = torch.arange(seq_len).float()
            X = torch.stack([torch.ones(seq_len), time_steps], dim=1)
            self.register_buffer('pinv_X', torch.linalg.pinv(X))
            self.register_buffer('time_steps', time_steps)
        elif self.detrend_method == 'simple':
            # For simple detrending, we only need seq_len and time steps
            self.register_buffer('seq_len_tensor', torch.tensor(seq_len))
            self.register_buffer('time_steps', torch.arange(seq_len).float())
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        return_details: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute noise scores from input time series.
        
        Args:
            inputs: Input tensor of shape [batch, seq_len, features] or [batch, features, seq_len].
            return_details: If True, return detailed information including intermediate results.
        
        Returns:
            If return_details=False: noise_scores tensor [batch_size]
            If return_details=True: dict with keys:
                - 'noise_scores': [batch_size]
                - 'original': [batch, features, seq_len]
                - 'reconstructed': [batch, features, seq_len]
                - 'mask': [batch, features, freqs]
                - 'fft_magnitude_log1p': [batch, features, freqs]
                - 'norm_amp': [batch, features, freqs]
                - 'threshold': [batch, features, 1] (in log1p space)
                - 'sfm_score': [batch, features, 1]
                - 'dynamic_threshold': [batch, features, 1] (in normalized space)
        """
        # Handle input format: [B, T, C] -> [B, C, T]
        if inputs.dim() == 3 and inputs.shape[1] != inputs.shape[2]:
            inputs_permuted = inputs.transpose(1, 2)
        else:
            inputs_permuted = inputs
        
        batch_size, num_features, seq_len = inputs_permuted.shape
        
        # --- 1. Detrending (Ablation Study) ---
        trend = None
        process_input = inputs_permuted
        
        if self.detrend_method == 'robust_ols':
            # Ours: Robust OLS detrending to prevent spectral leakage
            # Reshape to [B*C, L] for batch matrix multiplication
            flat_input = inputs_permuted.reshape(-1, seq_len)
            
            # Solve Y = X*beta => beta = Y @ pinv(X)^T
            # flat_input: [N, L], pinv_X: [2, L] -> coeffs: [N, 2]
            coeffs = flat_input @ self.pinv_X.t()
            
            # Calculate Trend: y = intercept + slope * t
            # intercept: coeffs[:, 0], slope: coeffs[:, 1]
            slope = coeffs[:, 1].unsqueeze(1)      # [N, 1]
            intercept = coeffs[:, 0].unsqueeze(1)  # [N, 1]
            
            # [N, 1] * [1, L] + [N, 1] -> [N, L]
            trend_flat = slope * self.time_steps.unsqueeze(0) + intercept
            trend = trend_flat.reshape(batch_size, num_features, seq_len)
            
            # Detrend for FFT
            process_input = inputs_permuted - trend
        
        elif self.detrend_method == 'simple':
            # Ablation: Simple end-to-end detrending (connect first and last points)
            start_val = inputs_permuted[:, :, 0:1]  # [B, C, 1]
            end_val = inputs_permuted[:, :, -1:]    # [B, C, 1]
            
            # Linear slope from start to end
            slope = (end_val - start_val) / (seq_len - 1)  # [B, C, 1]
            
            # Build trend line: start + slope * t
            # time_steps: [L], slope: [B, C, 1] -> trend: [B, C, L]
            trend = start_val + slope * self.time_steps.view(1, 1, -1)
            
            # Detrend for FFT
            process_input = inputs_permuted - trend
        
        # elif self.detrend_method == 'none': No detrending, use original input

        # --- 2. FFT and Filter ---
        fft_result = torch.fft.rfft(process_input, dim=-1)
        fft_magnitude = torch.abs(fft_result)
        fft_log = torch.log1p(fft_magnitude)
        
        # --- 3. Instance Normalization (Ablation Study) ---
        if self.use_instance_norm:
            # Ours: Instance-level normalization to [0, 1] for handling amplitude heterogeneity
            amp_min = torch.amin(fft_log, dim=-1, keepdim=True)
            amp_max = torch.amax(fft_log, dim=-1, keepdim=True)
            amp_range = torch.clamp(amp_max - amp_min, min=1e-6)
            norm_amp = (fft_log - amp_min) / amp_range  # [batch, features, freqs] in [0, 1]
        else:
            # Ablation: No normalization, use raw log1p magnitude
            norm_amp = fft_log
            amp_min = torch.zeros_like(fft_log[:, :, 0:1])
            amp_range = torch.ones_like(fft_log[:, :, 0:1])
        
        # Apply SFM-anchored RRF filter
        mask, dynamic_threshold, sfm_score = self.rrf_filter(fft_log, norm_amp)
        
        # Reconstruct signal using filtered frequencies
        filtered_fft = fft_result * mask
        reconstructed_residual = torch.fft.irfft(filtered_fft, n=seq_len, dim=-1)
        
        # Add trend back (if detrended) to compare with original input
        if self.detrend_method in ['robust_ols', 'simple'] and trend is not None:
            reconstructed = reconstructed_residual + trend
        else:
            reconstructed = reconstructed_residual
        
        # Compute noise score as reconstruction error (MAE)
        noise_scores = (inputs_permuted - reconstructed).abs().mean(dim=(1, 2))
        
        if return_details:
            # Convert threshold from normalized space to original log1p magnitude space
            threshold_log1p = dynamic_threshold * amp_range + amp_min  # [batch, features, 1]
            
            return {
                'noise_scores': noise_scores,
                'original': inputs_permuted,
                'reconstructed': reconstructed,
                'mask': mask,
                'fft_magnitude_log1p': fft_log,
                'norm_amp': norm_amp,
                'threshold': threshold_log1p,
                'sfm_score': sfm_score,
                'dynamic_threshold': dynamic_threshold
            }
        
        return noise_scores


def visualize_noise_scoring(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    noise_scores: torch.Tensor,
    dropout_rates: Optional[torch.Tensor] = None,
    fft_magnitude_log1p: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    thresholds: Optional[torch.Tensor] = None,
    num_samples: int = 6
) -> None:
    """Visualize noise scoring results.
    
    Creates a grid visualization showing original vs reconstructed signals and
    frequency spectra with thresholds. Layout: 2 samples per row, 2 plots per sample.
    
    Args:
        original: Original signals [batch, features, seq_len].
        reconstructed: Reconstructed signals [batch, features, seq_len].
        noise_scores: Noise scores [batch] or [batch, features].
        dropout_rates: Optional dropout rates [batch] or [batch, features].
        fft_magnitude_log1p: Optional FFT log-magnitude [batch, features, freqs].
        mask: Optional frequency mask [batch, features, freqs].
        thresholds: Optional threshold values [batch, features, 1] (in log1p space).
        num_samples: Number of samples to visualize (default: 6).
    """
    original_np = original.detach().cpu().numpy()
    reconstructed_np = reconstructed.detach().cpu().numpy()
    noise_scores_np = noise_scores.detach().cpu().numpy()
    B, C, L = original_np.shape
    
    # Select random samples to visualize
    all_indices = [(b, c) for b in range(B) for c in range(C)]
    if not all_indices:
        return
    
    real_num_samples = min(len(all_indices), num_samples)
    selected_indices = sorted(random.sample(all_indices, real_num_samples))
    rows, cols = (real_num_samples + 1) // 2, 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(24, 4 * rows))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    time_steps, freqs = np.arange(L), np.arange(L // 2 + 1)
    
    for idx, (b, c) in enumerate(selected_indices):
        row, col_start = idx // 2, (idx % 2) * 2
        
        # Time domain plot: Original vs Reconstructed
        ax_time = axes[row, col_start]
        ax_time.plot(time_steps, original_np[b, c, :], label='Original', linewidth=1.5, alpha=0.8)
        ax_time.plot(time_steps, reconstructed_np[b, c, :], label='Recon', linewidth=1.5, alpha=0.8, linestyle='--')
        
        score_val = noise_scores_np[b, c] if noise_scores_np.ndim == 2 else noise_scores_np[b]
        title_text = f'B:{b} C:{c} | Score: {score_val:.4f}'
        
        if dropout_rates is not None:
            if isinstance(dropout_rates, torch.Tensor) and dropout_rates.dim() == 2:
                dr = dropout_rates[b, c].item()
            else:
                dr = dropout_rates[b].item() if isinstance(dropout_rates, torch.Tensor) else dropout_rates[b]
            title_text += f' | p: {dr:.2f}'
        
        ax_time.set_title(title_text, fontsize=10)
        ax_time.legend(fontsize=8)
        ax_time.grid(True, alpha=0.3)
        if row == rows - 1:
            ax_time.set_xlabel('Time')
        
        # Frequency domain plot: Spectrum with threshold
        ax_freq = axes[row, col_start + 1]
        if fft_magnitude_log1p is not None:
            fft_log = fft_magnitude_log1p[b, c, :].detach().cpu().numpy()
        else:
            fft_log = np.log1p(np.abs(np.fft.rfft(original_np[b, c, :])))
        
        ax_freq.plot(freqs, fft_log, label='Spectrum', color='purple', alpha=0.7)
        
        if thresholds is not None:
            thresh_val = thresholds[b, c, 0].item() if thresholds.dim() == 3 else thresholds[b, c].item()
            ax_freq.axhline(y=thresh_val, color='red', linestyle='--', linewidth=1.5, label='Threshold')
            ax_freq.fill_between(
                freqs, 0, fft_log, where=(fft_log < thresh_val),
                color='gray', alpha=0.2, label='Filtered', interpolate=True
            )
        elif mask is not None:
            # Fallback: show filtered regions based on mask
            mask_np = mask[b, c, :].detach().cpu().numpy()
            mask_threshold = 0.5
            filtered_mask = mask_np < mask_threshold
            if np.any(filtered_mask):
                ax_freq.fill_between(
                    freqs, 0, fft_log, where=filtered_mask,
                    color='gray', alpha=0.2, label='Filtered', interpolate=True
                )
        
        ax_freq.set_title('Freq Spectrum (Log1p)', fontsize=10)
        ax_freq.legend(fontsize=8)
        ax_freq.grid(True, alpha=0.3)
        if row == rows - 1:
            ax_freq.set_xlabel('Freq Index')
    
    # Hide unused subplots
    for i in range(real_num_samples * 2, rows * cols):
        axes[i // cols, i % cols].axis('off')
    
    plt.savefig('noise_scoring_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


class DropoutTS(nn.Module):
    """Sample-adaptive dropout optimized by task loss.
    
    Computes per-sample dropout rates based on noise scores estimated via
    frequency-domain analysis. Higher noise leads to higher dropout rates.
    """
    
    def __init__(
        self,
        seq_len: Optional[int] = None,
        p_min: float = 0.1,
        p_max: float = 0.5,
        init_alpha: float = 10.0,
        init_sensitivity: float = 5.0,
        detrend_method: str = 'robust_ols',
        use_instance_norm: bool = True,
        use_sfm_anchor: bool = True
    ):
        """Initialize DropoutTS.
        
        Args:
            seq_len: Sequence length (inferred from inputs if None).
            p_min: Minimum dropout rate for clean samples.
            p_max: Maximum dropout rate for noisy samples.
            init_alpha: Initial value for RRF sigmoid sharpness parameter.
            init_sensitivity: Initial value for noise score sensitivity parameter.
            detrend_method: Detrending method ('none', 'simple', 'robust_ols').
            use_instance_norm: Whether to use instance-level normalization.
            use_sfm_anchor: Whether to use SFM as physical anchor.
        """
        super().__init__()
        self.p_min = p_min
        self.p_max = p_max
        self.init_alpha = init_alpha
        self.noise_scorer = None
        
        # Ablation study parameters
        self.detrend_method = detrend_method
        self.use_instance_norm = use_instance_norm
        self.use_sfm_anchor = use_sfm_anchor
        
        # Sensitivity parameter with optional clamping
        self.sensitivity = nn.Parameter(torch.tensor(init_sensitivity), requires_grad=True)
    
    def compute_dropout_rates(
        self,
        inputs: torch.Tensor,
        visualize: bool = False,
        max_samples: int = 4
    ) -> torch.Tensor:
        """Compute per-sample dropout rates from input time series.
        
        Args:
            inputs: Input tensor [batch, seq_len, features] or [batch, features, seq_len].
            visualize: If True, generate visualization of noise scoring.
            max_samples: Number of samples to visualize (if visualize=True).
        
        Returns:
            Dropout rates tensor [batch_size] in range [p_min, p_max].
        """
        # Lazy initialization of noise scorer
        if self.noise_scorer is None:
            if inputs.dim() == 3:
                seq_len = inputs.shape[1]
                num_features = inputs.shape[2]
            else:
                seq_len = inputs.shape[-1]
                num_features = 1
            
            self.noise_scorer = NoiseScorer(
                seq_len=seq_len,
                num_features=num_features,
                init_alpha=self.init_alpha,
                detrend_method=self.detrend_method,
                use_instance_norm=self.use_instance_norm,
                use_sfm_anchor=self.use_sfm_anchor
            ).to(inputs.device)
        
        # Get details when we need visualization
        details = self.noise_scorer(inputs, return_details=visualize)
        
        if visualize:
            noise_scores = details['noise_scores']
            
            # Normalize noise scores to [0, 1] within batch to fully utilize dropout range
            min_score = noise_scores.min()
            max_score = noise_scores.max()
            score_range = max_score - min_score
            if score_range > 1e-6:
                normalized = (noise_scores - min_score) / score_range
            else:
                # All samples have same noise level, use middle of range
                normalized = torch.full_like(noise_scores, 0.5)
            
            # Apply sensitivity scaling and tanh for smooth mapping
            normalized = torch.tanh(normalized * F.softplus(self.sensitivity))
            dropout_rates = self.p_min + (self.p_max - self.p_min) * normalized
            
            visualize_noise_scoring(
                original=details['original'],
                reconstructed=details['reconstructed'],
                noise_scores=noise_scores,
                dropout_rates=dropout_rates,
                fft_magnitude_log1p=details.get('fft_magnitude_log1p'),
                mask=details.get('mask'),
                thresholds=details.get('threshold'),
                num_samples=max_samples
            )
            
            return dropout_rates
        else:
            noise_scores = details
            
            # Normalize noise scores to [0, 1] within batch to fully utilize dropout range
            min_score = noise_scores.min()
            max_score = noise_scores.max()
            score_range = max_score - min_score
            if score_range > 1e-6:
                normalized = (noise_scores - min_score) / score_range
            else:
                # All samples have same noise level, use middle of range
                normalized = torch.full_like(noise_scores, 0.5)
            
            # Apply sensitivity scaling and tanh for smooth mapping
            normalized = torch.tanh(normalized * F.softplus(self.sensitivity))
            dropout_rates = self.p_min + (self.p_max - self.p_min) * normalized
            return dropout_rates
    
    def forward(
        self,
        inputs: torch.Tensor,
        dropout_layer: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: compute dropout rates and optionally apply dropout.
        
        Args:
            inputs: Input tensor [batch, seq_len, features] or [batch, features, seq_len].
            dropout_layer: Optional dropout layer to apply. If None, returns inputs unchanged.
                Supports SampleAdaptiveDropout and nn.Dropout.
        
        Returns:
            Tuple of (outputs, dropout_rates):
                - outputs: Input tensor with dropout applied (if dropout_layer provided)
                - dropout_rates: Per-sample dropout rates [batch_size]
        """
        dropout_rates = self.compute_dropout_rates(inputs)
        
        if dropout_layer is None:
            return inputs, dropout_rates
        
        if isinstance(dropout_layer, SampleAdaptiveDropout):
            outputs = dropout_layer(inputs, dropout_rates)
        elif isinstance(dropout_layer, nn.Dropout):
            outputs = F.dropout(inputs, p=dropout_rates.mean().item(), training=self.training)
        else:
            outputs = inputs
        
        return outputs, dropout_rates
