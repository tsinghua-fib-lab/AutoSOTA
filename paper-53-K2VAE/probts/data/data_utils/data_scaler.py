# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
# - License: MIT, Apache-2.0 license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

import torch
import torch.nn as nn

class Scaler:
    def __init__(self):
        super().__init__()

    def fit(self, values):
        raise NotImplementedError

    def transform(self, values):
        raise NotImplementedError

    def fit_transform(self, values):
        raise NotImplementedError

    def inverse_transform(self, values):
        raise NotImplementedError


class StandardScaler(Scaler):
    def __init__(
        self,
        mean: float = None,
        std: float = None,
        epsilon: float = 1e-9,
        var_specific: bool = True
    ):
        """
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        
        Args:
            mean: The mean of the features. The property will be set after a call to fit.
            std: The standard deviation of the features. The property will be set after a call to fit.
            epsilon: Used to avoid a Division-By-Zero exception.
            var_specific: If True, the mean and standard deviation will be computed per variate.
        """
        self.mean = mean
        self.scale = std
        self.epsilon = epsilon
        self.var_specific = var_specific

    def fit(self, values):
        """
        Args:
            values: Input values should be a PyTorch tensor of shape (T, C) or (N, T, C), 
                where N is the batch size, T is the timesteps and C is the number of variates.
        """
        dims = list(range(values.dim() - 1))
        if not self.var_specific:
            self.mean = torch.mean(values)
            self.scale = torch.std(values)
        else:
            self.mean = torch.mean(values, dim=dims)
            self.scale = torch.std(values, dim=dims)

    def transform(self, values):
        if self.mean is None:
            return values

        values = (values - self.mean.to(values.device)) / (self.scale.to(values.device) + self.epsilon)
        return values.to(torch.float32)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        if self.mean is None:
            return values
        
        values = values * (self.scale.to(values.device) + self.epsilon)
        values = values + self.mean.to(values.device)
        return values


class TemporalScaler(Scaler):
    def __init__(
        self,
        minimum_scale:float = 1e-10,
        time_first: bool = True
    ):
        """
        The ``TemporalScaler`` computes a per-item scale according to the average
        absolute value over time of each item. The average is computed only among
        the observed values in the data tensor, as indicated by the second
        argument. Items with no observed data are assigned a scale based on the
        global average.

        Args:
            minimum_scale: default scale that is used if the time series has only zeros.
            time_first: if True, the input tensor has shape (N, T, C), otherwise (N, C, T).
        """
        super().__init__()
        self.scale = None
        self.minimum_scale = torch.tensor(minimum_scale)
        self.time_first = time_first

    def fit(
        self,
        data: torch.Tensor,
        observed_indicator: torch.Tensor = None
    ):
        """
        Fit the scaler to the data.
        
        Args:
            data: tensor of shape (N, T, C) if ``time_first == True`` or (N, C, T)
                if ``time_first == False`` containing the data to be scaled

            observed_indicator: observed_indicator: binary tensor with the same shape as
                ``data``, that has 1 in correspondence of observed data points,
                and 0 in correspondence of missing data points.

        Note:
            Tensor containing the scale, of shape (N, 1, C) or (N, C, 1).
        """
        if self.time_first:
            dim = -2
        else:
            dim = -1

        if observed_indicator is None:
            observed_indicator = torch.ones_like(data)

        # These will have shape (N, C)
        num_observed = observed_indicator.sum(dim=dim)
        sum_observed = (data.abs() * observed_indicator).sum(dim=dim)

        # First compute a global scale per-dimension
        total_observed = num_observed.sum(dim=0)
        denominator = torch.max(total_observed, torch.ones_like(total_observed))
        default_scale = sum_observed.sum(dim=0) / denominator

        # Then compute a per-item, per-dimension scale
        denominator = torch.max(num_observed, torch.ones_like(num_observed))
        scale = sum_observed / denominator

        # Use per-batch scale when no element is observed
        # or when the sequence contains only zeros
        scale = torch.where(
            sum_observed > torch.zeros_like(sum_observed),
            scale,
            default_scale * torch.ones_like(num_observed),
        )

        self.scale = torch.max(scale, self.minimum_scale).unsqueeze(dim=dim).detach()

    def transform(self, data):
        return data / self.scale.to(data.device)

    def fit_transform(self, data, observed_indicator=None):
        self.fit(data, observed_indicator)
        return self.transform(data)

    def inverse_transform(self, data):
        return data * self.scale.to(data.device)


class IdentityScaler(Scaler):
    """
    No scaling is applied upon calling the ``IdentityScaler``.
    """
    def __init__(self, time_first: bool = True):
        super().__init__()
        self.scale = None
        
    def fit(self, data):
        pass

    def transform(self, data):
        return data
    
    def inverse_transform(self, data):
        return data
    
class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        """
        :param eps: a value added for numerical stability
        """
        super(InstanceNorm, self).__init__()
        self.eps = eps

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):
        x = x * self.stdev
        x = x + self.mean
        return x

