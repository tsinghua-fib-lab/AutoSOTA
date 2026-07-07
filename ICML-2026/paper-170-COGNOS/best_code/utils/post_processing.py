import numpy as np
import pandas as pd
import torch
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import chi2


def apply_moving_average_filter(anomaly_scores):
    """
    Apply moving average filter to smooth anomaly scores.
    
    Args:
        anomaly_scores: Array of anomaly scores to be smoothed
        
    Returns:
        Filtered anomaly scores after applying moving average
    """
    s_full_residual = pd.Series(anomaly_scores.flatten())

    filtered_series = s_full_residual.rolling(window=20, center=True, min_periods=1).mean().values

    return filtered_series


def apply_lowpass_filter(anomaly_scores):
    """
    Apply low-pass filter using FFT to remove high frequency noise from anomaly scores.
    
    Args:
        anomaly_scores: Array of anomaly scores to be filtered
        
    Returns:
        Filtered anomaly scores after applying low-pass filter
    """
    mse_series = anomaly_scores

    N = len(mse_series)

    cutoff_freq_normalized = 0.05

    yf = fft(mse_series)

    xf = fftfreq(N, 1)

    filter_mask = np.abs(xf) < cutoff_freq_normalized

    yf_filtered = yf * filter_mask

    filtered_mse_series = ifft(yf_filtered).real

    return filtered_mse_series

class KalmanSmoothing():
    def __init__(self, args):
        """
        Initialize the Kalman Smoothing class.
        
        Args:
            args: Arguments containing configuration parameters
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = 1e-8  # Small constant for filling variance
        self.seq_noise = None
        # Initialize results
        self.autocovariance_1 = torch.zeros(self.args.c_out, dtype=torch.float32)
        self.autocovariance_2 = torch.zeros(self.args.c_out, dtype=torch.float32)
        self.threshold = chi2.ppf(self.args.KF_confidence, df=1)

    def process_residual_observations_per_channel(self, model, data_loader, total_original_timesteps,
                                                  is_train_set=True):
        """
        Process residual observations per channel using Kalman smoothing approach.
        
        Args:
            model: The trained model for prediction
            data_loader: DataLoader containing the test data
            total_original_timesteps: Total number of original timesteps
            is_train_set: Boolean indicating whether it's training set
            
        Returns:
            Processed results depending on is_train_set flag
        """
        model.eval()

        if self.args.use_Gaussian_regularization == False:
            # Vanilla implementation
            return self._process_vanilla(model, data_loader, is_train_set)

        # 1. Initialize GPU accumulators
        accumulators = self._init_gpu_accumulators(total_original_timesteps)

        # Labels container needed only in non-training mode
        final_point_labels_np_scalar = None
        if not is_train_set:
            final_point_labels_np_scalar = np.zeros(total_original_timesteps, dtype=np.int32)

        batch_size_dl = data_loader.batch_size

        # 2. Process batches in loop
        for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
            current_batch_size = batch_x.shape[0]
            batch_x = batch_x.float().to(self.device)

            with torch.no_grad():
                # Calculate model output and errors
                errors_per_element, abs_time_indices_in_batch = self._compute_batch_errors(
                    model, batch_x, batch_idx, batch_size_dl
                )

                # Accumulate Lag 0 (residual sum, square sum, count)
                self._accumulate_basic_residuals(
                    accumulators, errors_per_element, abs_time_indices_in_batch, total_original_timesteps
                )

                # Accumulate Lag 1 & Lag 2 (autocovariance related)
                self._accumulate_autocovariance_lags(
                    accumulators, errors_per_element, abs_time_indices_in_batch, total_original_timesteps
                )

                # Non-train set: collect labels
                if not is_train_set:
                    self._collect_batch_labels(
                        final_point_labels_np_scalar, batch_y, abs_time_indices_in_batch, total_original_timesteps
                    )

        # 3. Post-process (divert based on is_train_set)
        if is_train_set:
            return self._post_process_train_stats(accumulators, total_original_timesteps)
        else:
            return self._post_process_inference_results(accumulators, final_point_labels_np_scalar,
                                                        total_original_timesteps)

    def apply_adaptive_kalman_filter(self, residual_series):
        """
        Apply adaptive Kalman filter based on AR(1) mean reversion model
        Parameters are automatically estimated through autocovariances (gamma0, gamma1, gamma2) of observation sequence.
        
        Args:
            residual_series: Series of residuals to be filtered
            
        Returns:
            Anomaly scores after Kalman filtering
        """

        # --- 1. Set device ---
        # Assuming residual_series is numpy array
        device = torch.device("cpu")  # Or according to your self.device

        # [Time_idx, Channels]
        noisy_data = torch.from_numpy(residual_series).float().to(device)
        num_points, num_channels = noisy_data.shape

        print('Filtering residuals with AR(1) Mean Reverting Model...')

        # CODE-02: Check for NaN/Inf in input residuals
        if torch.isnan(noisy_data).any() or torch.isinf(noisy_data).any():
            print('[Kalman] WARNING: NaN/Inf in input, replacing with zeros')
            noisy_data = torch.nan_to_num(noisy_data)

        # --- 2. Hybrid Parameter Configuration ---

        gamma_0 = torch.tensor(self.seq_noise, dtype=torch.float32, device=device)
        gamma_1 = torch.tensor(self.autocovariance_1, dtype=torch.float32, device=device)
        gamma_2 = torch.tensor(self.autocovariance_2, dtype=torch.float32, device=device)

        # 1. Determine channel types
        # Valid: Has good autocorrelation, suitable for AR(1) mean reversion
        # Invalid: High-frequency noise (gamma_1 <= 0) or abnormal correlation decay
        valid_mask = (gamma_1 > 1e-9) & (gamma_2 > 1e-9)

        # --- Branch A: For channels with [good correlation] (AR(1) estimation) ---
        # Calculate phi, sigma_x using moment estimation method
        phi_est = gamma_2 / (gamma_1 + 1e-9)
        a_valid = torch.clamp(phi_est, 0.0, 0.99)

        denom_a = torch.where(a_valid > 1e-3, a_valid, torch.ones_like(a_valid))
        sigma_x_est = gamma_1 / denom_a
        sigma_x_valid = torch.clamp(sigma_x_est, min=torch.zeros_like(gamma_0), max=0.99 * gamma_0)

        r_valid = torch.clamp(gamma_0 - sigma_x_valid, min=1e-6)
        q_valid = torch.clamp(sigma_x_valid * (1 - a_valid ** 2), min=1e-10)

        # --- Branch B: For channels with [high-frequency noise] (forced smoothing) ---
        # Strategy: Random walk model
        # Purpose: Filter out high-frequency jitter, preserve low-frequency trend

        # 1. State transition a set to 1.0 (or 0.995 to avoid numerical drift)
        # This allows state to remember past values, forming trends
        a_smooth = torch.ones_like(gamma_0) * 0.995

        # 2. Observation noise R set to total variance gamma_0
        # Assume all fluctuations are mainly noise
        r_smooth = gamma_0

        # 3. Process noise Q set to a small proportion of R (hyperparameter)
        # This proportion determines smoothing degree. Smaller = more smoothing, larger lag.
        # Suggested value: 0.01 ~ 0.001
        smooth_factor = 0.1
        q_smooth = r_smooth * smooth_factor

        # --- 3. Parameter merging ---

        # Select parameters based on mask
        a = torch.where(valid_mask, a_valid, a_smooth)
        r = torch.where(valid_mask, r_valid, r_smooth)
        q_normal = torch.where(valid_mask, q_valid, q_smooth)

        # Perform safety clamping again
        r = torch.clamp(r, min=1e-6)
        q_normal = torch.clamp(q_normal, min=1e-9)

        # Q for anomalies (effective for both modes)
        # When trend changes occur, NIS increases, switch to q_anomaly allowing rapid tracking
        q_anomaly = self.args.anomaly_QR_ratio * r

        h = 1.0

        # --- 4. Initialization ---
        x_hat = torch.zeros(num_points, num_channels, device=device)
        p = torch.zeros(num_points, num_channels, device=device)

        # Initialize state
        x_hat[0] = noisy_data[0]
        # Initialize covariance:
        # Valid channels use estimated sigma_x
        # Smooth channels use r (since uncertainty is large)
        p[0] = torch.where(valid_mask, sigma_x_valid, r)

        # --- 5. Loop (no longer need to force K=1) ---
        for k in range(1, num_points):
            # ... (standard Kalman filter process, no need to modify K) ...
            # Because we set reasonable Q and R for Invalid channels,
            # K will automatically calculate a smaller value (e.g., 0.05), acting as low-pass filtering

            x_hat_prev = x_hat[k - 1]
            p_prev = p[k - 1]

            x_pred_normal = a * x_hat_prev
            p_pred_normal = a * torch.clamp(p_prev, min=0.0) * a + q_normal

            y_innovation = noisy_data[k] - h * x_pred_normal
            s_normal = h * p_pred_normal * h + r

            # NIS detects anomalies
            epsilon = (y_innovation ** 2) / (s_normal + 1e-9)

            is_anomaly = (epsilon > self.threshold)
            q_to_use = torch.where(is_anomaly, q_anomaly, q_normal)

            # Update
            p_pred = a * p_prev * a + q_to_use
            s = h * p_pred * h + r
            K = (p_pred * h) / (s + 1e-9)

            x_hat_k = x_pred_normal + K * y_innovation
            # Replace NaN/Inf with zero (stable fallback)
            nan_mask = torch.isnan(x_hat_k) | torch.isinf(x_hat_k)
            x_hat_k = torch.where(nan_mask, torch.zeros_like(x_hat_k), x_hat_k)
            x_hat[k] = x_hat_k
            # 6. Update covariance (using Joseph Form for numerical stability)
            p_k = p_pred * ((1 - K * h)**2) + r * (K**2)
            p[k] = torch.clamp(p_k, min=1e-12)

        # MSE shape is [Time_idx, Channels], which is the square of the residual after filtering (Kalman state estimate) for each channel
        MSE = x_hat ** 2

        # As per docstring requirement, return [Time_idx * Channels] numpy array
        anomaly_scores = MSE.flatten().cpu().numpy()

        return anomaly_scores

    # =========================================================================
    #  Internal Helper Methods
    # =========================================================================
    def _init_gpu_accumulators(self, total_timesteps):
        """
        Initialize all accumulator tensors that need scatter_add on GPU.
        
        Args:
            total_timesteps: Total number of timesteps
            
        Returns:
            Dictionary of accumulator tensors
        """
        shape = (total_timesteps, self.args.c_out)
        return {
            # Lag 0
            'res_sum': torch.zeros(shape, dtype=torch.float32, device=self.device),
            'res_sq': torch.zeros(shape, dtype=torch.float32, device=self.device),
            'res_cnt': torch.zeros(shape, dtype=torch.int32, device=self.device),
            # Lag 1
            'lag1_sum': torch.zeros(shape, dtype=torch.float32, device=self.device),
            'lag1_cnt': torch.zeros(shape, dtype=torch.int32, device=self.device),
            # Lag 2
            'lag2_sum': torch.zeros(shape, dtype=torch.float32, device=self.device),
            'lag2_cnt': torch.zeros(shape, dtype=torch.int32, device=self.device),
        }

    def _compute_batch_errors(self, model, batch_x, batch_idx, batch_size_dl):
        """
        Forward propagation of model and calculation of errors.
        
        Args:
            model: The model used for prediction
            batch_x: Input batch tensor
            batch_idx: Current batch index
            batch_size_dl: Batch size from dataloader
            
        Returns:
            Tuple of errors per element and absolute time indices in batch
        """
        input_x = batch_x
        means = input_x.mean(1, keepdim=True).detach()
        input_x = input_x.sub(means)
        stdev = torch.sqrt(
            torch.var(input_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        input_x = input_x.div(stdev)

        if self.args.model == 'CrossAD':
            ms_x_dec, _ = model(input_x, None, None, None)
            outputs = model.ms_interpolate(ms_x_dec)
        else:
            outputs = model(input_x, None, None, None)

        # De-Normalization from Non-stationary Transformer
        outputs = outputs.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.args.seq_len, 1)))
        outputs = outputs.add(
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.args.seq_len, 1)))

        errors_per_element = batch_x - outputs  # [B, T, C]

        # Calculate absolute time indices
        current_batch_size = batch_x.shape[0]
        window_start_indices = torch.arange(current_batch_size, device=self.device) + batch_idx * batch_size_dl
        abs_time_indices_in_batch = window_start_indices.unsqueeze(1) + \
                                    torch.arange(self.args.seq_len, device=self.device).unsqueeze(0)

        return errors_per_element, abs_time_indices_in_batch

    def _accumulate_basic_residuals(self, acc, errors, abs_indices, total_timesteps):
        """
        Process basic residual accumulation (Lag 0).
        
        Args:
            acc: Accumulator dictionary
            errors: Error tensors
            abs_indices: Absolute indices
            total_timesteps: Total number of timesteps
        """
        flat_errors = errors.reshape(-1, self.args.c_out)
        flat_indices = abs_indices.flatten()

        valid_mask = (flat_indices < total_timesteps)
        valid_idx = flat_indices[valid_mask]
        valid_err = flat_errors[valid_mask]

        if valid_idx.numel() > 0:
            scatter_idx = valid_idx.unsqueeze(1).expand(-1, self.args.c_out)
            acc['res_sum'].scatter_add_(0, scatter_idx, valid_err)
            acc['res_sq'].scatter_add_(0, scatter_idx, valid_err ** 2)
            acc['res_cnt'].scatter_add_(0, scatter_idx, torch.ones_like(valid_err, dtype=torch.int32))

    def _accumulate_autocovariance_lags(self, acc, errors, abs_indices, total_timesteps):
        """
        Process autocovariance accumulation for Lag 1 and Lag 2 (Scheme B: Keep explicit logic).
        
        Args:
            acc: Accumulator dictionary
            errors: Error tensors
            abs_indices: Absolute indices
            total_timesteps: Total number of timesteps
        """

        # --- Lag 1 ---
        lag1_curr = errors[:, 1:, :]
        lag1_prev = errors[:, :-1, :]
        lag1_prod = lag1_curr * lag1_prev
        lag1_idx_map = abs_indices[:, 1:]

        flat_lag1_idx = lag1_idx_map.flatten()
        flat_lag1_prod = lag1_prod.reshape(-1, self.args.c_out)
        mask_lag1 = (flat_lag1_idx < total_timesteps)

        valid_idx_1 = flat_lag1_idx[mask_lag1]
        valid_prod_1 = flat_lag1_prod[mask_lag1]

        if valid_idx_1.numel() > 0:
            s_idx_1 = valid_idx_1.unsqueeze(1).expand(-1, self.args.c_out)
            acc['lag1_sum'].scatter_add_(0, s_idx_1, valid_prod_1)
            acc['lag1_cnt'].scatter_add_(0, s_idx_1, torch.ones_like(valid_prod_1, dtype=torch.int32))

        # --- Lag 2 ---
        lag2_curr = errors[:, 2:, :]
        lag2_prev = errors[:, :-2, :]
        lag2_prod = lag2_curr * lag2_prev
        lag2_idx_map = abs_indices[:, 2:]

        flat_lag2_idx = lag2_idx_map.flatten()
        flat_lag2_prod = lag2_prod.reshape(-1, self.args.c_out)
        mask_lag2 = (flat_lag2_idx < total_timesteps)

        valid_idx_2 = flat_lag2_idx[mask_lag2]
        valid_prod_2 = flat_lag2_prod[mask_lag2]

        if valid_idx_2.numel() > 0:
            s_idx_2 = valid_idx_2.unsqueeze(1).expand(-1, self.args.c_out)
            acc['lag2_sum'].scatter_add_(0, s_idx_2, valid_prod_2)
            acc['lag2_cnt'].scatter_add_(0, s_idx_2, torch.ones_like(valid_prod_2, dtype=torch.int32))

    def _collect_batch_labels(self, labels_container, batch_y, abs_indices, total_timesteps):
        """
        Collect labels in non-training mode.
        
        Args:
            labels_container: Container to store labels
            batch_y: Batch labels tensor
            abs_indices: Absolute indices
            total_timesteps: Total number of timesteps
        """
        batch_labels = batch_y.squeeze().cpu().numpy().flatten()
        flat_indices = abs_indices.flatten()
        valid_mask = (flat_indices < total_timesteps)

        valid_idx = flat_indices[valid_mask]

        if valid_idx.numel() > 0:
            valid_idx_np = valid_idx.cpu().numpy()
            valid_mask_np = valid_mask.cpu().numpy()
            labels_container[valid_idx_np] = batch_labels[valid_mask_np]

    # =========================================================================
    #  Post-Processing Methods
    # =========================================================================
    def _compute_mean_residuals(self, acc):
        """
        Compute basic mean residuals, common step.
        
        Args:
            acc: Accumulator dictionary
            
        Returns:
            Average residuals and counts as floats
        """
        counts_float = acc['res_cnt'].float()
        valid_any = (counts_float > 0)

        avg_residuals = torch.full_like(acc['res_sum'], float('nan'))
        avg_residuals[valid_any] = acc['res_sum'][valid_any] / counts_float[valid_any]
        return avg_residuals, counts_float

    def _post_process_train_stats(self, acc, total_timesteps):
        """
        Post-process for training set: compute variance, update seq_noise and autocovariance.
        
        Args:
            acc: Accumulator dictionary
            total_timesteps: Total number of timesteps
            
        Returns:
            Average residuals as numpy array
        """
        avg_residuals, counts_float = self._compute_mean_residuals(acc)

        # 1. Compute variance
        avg_variance = torch.full_like(avg_residuals, float('nan'))

        single_obs = (counts_float == 1)
        multi_obs = (counts_float > 1)

        # Multiple observation points: E[x^2] - (E[x])^2
        if torch.any(multi_obs):
            mean_sq = acc['res_sq'][multi_obs] / counts_float[multi_obs]
            sq_mean = avg_residuals[multi_obs] ** 2
            var = mean_sq - sq_mean
            var[var < 0] = self.eps  # Clamp
            avg_variance[multi_obs] = var

        # Single observation points: fill with eps
        if torch.any(single_obs):
            avg_variance[single_obs] = self.eps

        # Update global noise
        self.seq_noise = avg_variance.cpu().mean(dim=0)

        # 2. Update Autocovariance
        self._update_autocovariance(acc)

        return avg_residuals.cpu().numpy()

    def _update_autocovariance(self, acc):
        """
        Compute global average autocovariance for Lag 1 and Lag 2 and update class attributes.
        
        Args:
            acc: Accumulator dictionary
        """
        # Prevent division by zero
        safe_cnt1 = acc['lag1_cnt'].float().clamp(min=1.0)
        safe_cnt2 = acc['lag2_cnt'].float().clamp(min=1.0)

        ac1_per_step = acc['lag1_sum'] / safe_cnt1
        ac2_per_step = acc['lag2_sum'] / safe_cnt2

        # Move to CPU aggregation
        ac1_cpu = ac1_per_step.cpu()
        ac2_cpu = ac2_per_step.cpu()
        mask1_cpu = (acc['lag1_cnt'].cpu() > 0)
        mask2_cpu = (acc['lag2_cnt'].cpu() > 0)

        for c in range(self.args.c_out):
            if mask1_cpu[:, c].any():
                self.autocovariance_1[c] = ac1_cpu[mask1_cpu[:, c], c].mean()
            if mask2_cpu[:, c].any():
                self.autocovariance_2[c] = ac2_cpu[mask2_cpu[:, c], c].mean()

    def _post_process_inference_results(self, acc, labels_scalar, total_timesteps):
        """
        Post-process for inference set: generate label matrix.
        
        Args:
            acc: Accumulator dictionary
            labels_scalar: Scalar labels
            total_timesteps: Total number of timesteps
            
        Returns:
            Tuple of average residuals and label matrix
        """
        avg_residuals, _ = self._compute_mean_residuals(acc)

        kf_output_labels = np.full((total_timesteps, self.args.c_out), -1, dtype=np.int32)

        # Use count mask on GPU to quickly determine which points have data
        obs_mask_np = (acc['res_cnt'][:, 0] > 0).cpu().numpy()

        kf_output_labels[obs_mask_np, :] = labels_scalar[obs_mask_np][:, np.newaxis]

        return avg_residuals.cpu().numpy(), kf_output_labels

    # =========================================================================
    #  New: Vanilla Mode
    # =========================================================================

    def _process_vanilla(self, model, data_loader, is_train_set):
        """
        Process vanilla implementation without Gaussian regularization.
        
        Args:
            model: The trained model
            data_loader: DataLoader containing the data
            is_train_set: Boolean indicating whether it's training set
            
        Returns:
            Processed results depending on is_train_set flag
        """
        # Container: collect all residuals whether training or inference
        all_residuals_list = []
        # Label container: needed only in inference mode
        all_labels_list = []

        # Statistic accumulator (used only during training phase to compute global Gamma)
        stats_acc = {
            'count': 0,
            'sum_x': torch.zeros(self.args.c_out, device=self.device),
            'sum_x2': torch.zeros(self.args.c_out, device=self.device),
            'sum_lag1': torch.zeros(self.args.c_out, device=self.device),
            'sum_lag2': torch.zeros(self.args.c_out, device=self.device),
            'lag1_count': 0,
            'lag2_count': 0
        }

        # Cache: used to connect Lag calculations between batches
        prev_tail = None

        batch_size_dl = data_loader.batch_size

        for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.float().to(self.device)

            with torch.no_grad():
                # Reuse existing error calculation
                errors_per_element, _ = self._compute_batch_errors(
                    model, batch_x, batch_idx, batch_size_dl
                )

                # [B, T, C] -> [B*T, C]
                flat_errors = errors_per_element.reshape(-1, self.args.c_out)

                # --- Common operation: collect residual data ---
                # Save regardless of training or testing to return at the end
                all_residuals_list.append(flat_errors.cpu().numpy())

                # --- Branch: Training set (calculate statistics) ---
                if is_train_set:
                    self._accumulate_vanilla_stats(stats_acc, flat_errors, prev_tail)
                    # Update tail cache
                    prev_tail = flat_errors[-2:].clone() if flat_errors.shape[0] >= 2 else flat_errors.clone()

                # --- Branch: Inference set (collect labels) ---
                else:
                    # Collect and align labels [B, T] -> [B*T, 1]
                    all_labels_list.append(batch_y)

        # === Post-processing and return ===

        # 1. Concatenate all residuals [Total_Len, C]
        final_residuals = np.concatenate(all_residuals_list, axis=0)

        if is_train_set:
            # Training mode: compute and update internal parameters (seq_noise etc.), then return residuals
            self._finalize_vanilla_stats(stats_acc)
            return final_residuals
        else:
            # Inference mode: return residuals and labels
            # Generate label matrix [Total_Len, C]
            final_labels = np.concatenate(all_labels_list, axis=0).reshape(-1)

            return final_residuals, final_labels

    def _accumulate_vanilla_stats(self, acc, current_flat, prev_tail):
        """
        Stream accumulation of variance and autocovariance (optimized version)
        Strategy: If batch data is too long, randomly sample a continuous sequence to accelerate parameter estimation.
        
        Args:
            acc: Statistics accumulator
            current_flat: Current flattened errors
            prev_tail: Previous tail values for continuity
        """
        # Get sampling limit, default to 5000 if not set (adjust based on memory capacity)
        sample_limit = getattr(self.args, 'stats_sample_len', 5000)

        total_len = current_flat.shape[0]


        # Randomly select a starting position and take a continuous sample_limit length
        # Use randint to generate random offset
        processed_data = current_flat[0:  sample_limit]

        # Since we're randomly sampling a middle section, we can't physically connect with prev_tail from previous batch
        # Ignore loss of 1-2 connection points, has almost no effect on statistical results
        combined_seq = processed_data

        num_points = processed_data.shape[0]

        # 1. Basic statistics (Lag 0) - based only on current sampled data
        acc['count'] += num_points
        acc['sum_x'] += processed_data.sum(dim=0)
        acc['sum_x2'] += (processed_data ** 2).sum(dim=0)

        # 2. Lag statistics - based on combined_seq (may include tail)
        if combined_seq.shape[0] > 1:
            # Calculate Lag 1
            # Sequence: [x0, x1, x2, ...]
            # curr: [x1, x2, ...]
            # prev: [x0, x1, ...]
            lag1_prod = combined_seq[1:] * combined_seq[:-1]
            acc['sum_lag1'] += lag1_prod.sum(dim=0)
            acc['lag1_count'] += lag1_prod.shape[0]

        if combined_seq.shape[0] > 2:
            # Calculate Lag 2
            # curr: [x2, x3, ...]
            # prev: [x0, x1, ...]
            lag2_prod = combined_seq[2:] * combined_seq[:-2]
            acc['sum_lag2'] += lag2_prod.sum(dim=0)
            acc['lag2_count'] += lag2_prod.shape[0]
            
    def _finalize_vanilla_stats(self, acc):
        """
        Calculate final Noise and Covariance and update class attributes, no longer responsible for returning data.
        
        Args:
            acc: Statistics accumulator
        """
        count = max(acc['count'], 1)
        mean = acc['sum_x'] / count
        mean_sq = acc['sum_x2'] / count

        # Var = E[x^2] - (E[x])^2
        variance = mean_sq - mean ** 2
        variance[variance < 0] = self.eps

        # Update global noise
        self.seq_noise = variance.cpu()

        # Calculate Autocovariance
        cnt1 = max(acc['lag1_count'], 1)
        cnt2 = max(acc['lag2_count'], 1)

        term1_lag1 = acc['sum_lag1'] / cnt1
        term1_lag2 = acc['sum_lag2'] / cnt2

        self.autocovariance_1 = (term1_lag1 - mean ** 2).cpu()
        self.autocovariance_2 = (term1_lag2 - mean ** 2).cpu()