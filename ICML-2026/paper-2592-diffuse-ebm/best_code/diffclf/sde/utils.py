# Various useful functions
import torch

def binary_search(f, low, high, target_value, n_attemps):
    """Binary search function (vectorized)"""
    for _ in range(n_attemps):
        # Get the middle point
        mid = (low + high) / 2.
        ret = f(mid)
        # Check the different conditions
        low = torch.where(ret > target_value, mid, low)
        high = torch.where(ret <= target_value, mid, high)
    return (low + high) / 2.

class TimeSampler(torch.nn.Module):
    """Helper to sample the times for a diffusion SDE"""

    def __init__(self, times=None, sde=None, t_limits=None, log_snr_dist=(2.4,2.4)):
        """Constructor for the TimeSampler object

        Args:
            * times (torch.Tensor of shape (n_levels,)): Discrete levels (only when sde is not defined)
            * sde (LinearSDE): SDE (only is times is not defined)
            * t_limits (tuple of length 1): Minimum and maximum times allowed (only when sde is defined)
            * log_snr_dist (tuple of length 2): Mean and standard deviation of the log-SNR dist
                (default is (2.4, 2.4) based on (Karras et al. 2022))

        """
        super().__init__()
        if (sde is None) and (times is None):
            raise ValueError('Either of sde or times must be defined.')
        if (sde is not None) and (times is not None):
            raise ValueError('sde and times can\'t be defined together.')
        self.use_continuous_time = sde is not None
        if self.use_continuous_time and (t_limits is None):
            raise ValueError('t_limits must be defined if sde is defined.')
        self.sde = sde
        if times is not None:
            self.register_buffer('times', times.flatten())
            self.n_levels = self.times.shape[0]
        self.log_snr_dist = log_snr_dist
        if t_limits is not None:
            log_snr_limits = []
            for t in t_limits:
                if not isinstance(t, torch.Tensor):
                    t_ = torch.tensor(t)
                else:
                    t_ = t
                log_snr_limits.append(self.sde.log_snr_inv(t_))
            self.register_buffer('log_snr_limits', torch.FloatTensor(sorted(log_snr_limits)))

    def sample(self, sample_shape, return_idx=False, unique=False, exclude_first_level=False,
            exclude_last_level=False):
        """Sample times

        Args:
            * sample_shape (tuple): Shape of the sample (we recommend not using data_shape_ones in)
            * return_idx (bool): Whether to return the idx (only when self.times is defined)
            * unique (bool): Whether to return unique values on the rows (defaul is False)
            * exclude_first_level (bool): Whether to exclude time 0 (default is False)
            * exclude_last_level (bool): Whether to exclude time T (default is False)

        Returns:
            * ts (torch.Tensor of shape sample_shape): Times
            if return_idx:
                * idx (torch.Tensor of shape sample_shape): Different time indexes (None if times is None)
        """
        if self.use_continuous_time:
            # Sample the log-SNR
            log_snrs = torch.empty(sample_shape, device=self.log_snr_limits.device)
            torch.nn.init.trunc_normal_(log_snrs, mean=self.log_snr_dist[0], std=self.log_snr_dist[1],
                a=self.log_snr_limits[0], b=self.log_snr_limits[1])
            # Convert to times and return 
            ts = self.sde.log_snr_inv(log_snrs)
            idx = None
        else:
            # Sample the indexes
            if unique:
                # Check the length of the shape
                if len(sample_shape) > 2:
                    raise ValueError('Can\'t use unique when the shape if more than 2')
                # Get the number of considered levels
                n_levels = self.n_levels
                if exclude_first_level:
                    n_levels -= 1
                if exclude_last_level:
                    n_levels -= 1
                # Case of a 1D shape
                if len(sample_shape) == 1:
                    n_samples = sample_shape[0]
                    if n_samples > n_levels:
                        raise ValueError('Number of samples (= {}) is larger than the number of levels (= {})'.format(
                            n_samples, n_levels
                        ))
                    if (not exclude_first_level) and exclude_last_level:
                        idx_ = torch.randperm(self.n_levels-1)
                    elif exclude_first_level and (not exclude_last_level):
                        idx_ = 1+torch.randperm(self.n_levels-1)
                    elif exclude_first_level and exclude_last_level:
                        idx_ = 1+torch.randperm(self.n_levels-2)
                    else:
                        idx_ = torch.randperm(self.n_levels)
                    idx = idx_[:n_samples]
                # Case of a 2D shape
                if len(sample_shape) == 2:
                    if sample_shape[1] > n_levels:
                        raise ValueError('Number of samples (= {}) is larger than the number of levels (= {})'.format(
                            sample_shape[1], n_levels
                        ))
                    probs = torch.ones(sample_shape[0], n_levels, device=self.times.device)
                    idx = torch.multinomial(probs, sample_shape[1], replacement=False)
                    if exclude_first_level:
                        idx += 1
            else:
                idx = torch.randint(
                    low=1 if exclude_first_level else 0,
                    high=self.n_levels-1 if exclude_last_level else self.n_levels,
                    size=sample_shape,
                    device=self.times.device
                )
            # Convert indexes to times
            ts = self.times[idx]
        # Return indexes is needed
        if return_idx:
            return ts, idx
        else:
            return ts



