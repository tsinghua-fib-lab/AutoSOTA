import abc
import torch
import numpy as np
from .utils.weight_utils import get_weight_fn


class Scheduler(abc.ABC):
    """
    prior -> data
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.eps = kwargs.get("eps", 1e-6)
        self.beta_0 = kwargs.get("beta_0")
        self.beta_f = kwargs.get("beta_f")

        self.rbeta = self.beta_0 / self.beta_f if self.beta_f>0 else self.beta_0 / self.beta(self.eps)
        assert np.isfinite(self.rbeta)
        self._beta = self.beta_f - self.beta_0

        self.weight_type = kwargs.get("weight_type", "default")
        self.weight_fn = get_weight_fn(self.weight_type)(**kwargs)

    @property
    def T(self):
        return 1

    @abc.abstractmethod
    def beta(self, t):
        pass
    
    @abc.abstractmethod
    def int_beta(self, t):
        # integrate beta from time t to T
        pass
        
    @abc.abstractmethod
    def drift_coeff(self, t):
        # return self.beta(t) / self.int_beta(t)
        pass
        
    def importance_weight(self, t, train):
        if train:
            return self.weight_fn.norm_const / self.weight_fn(t)
        else:
            return 1.

    def importance_weighted_time(self, shape, device, steps=100):
        quantile = torch.rand(shape, device=device) * self.weight_fn.norm_const
        lb = torch.zeros_like(quantile)
        ub = torch.ones_like(quantile) * (1 - self.eps)

        for _ in range(steps):
            mid = (lb + ub) / 2.
            value = self.weight_fn.cum_weight_fn(mid)
            lb = torch.where(value <= quantile, mid, lb)
            ub = torch.where(value <= quantile, ub, mid)

        return (lb + ub) / 2.
    

class Linear(Scheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def beta(self, t):
        return self.beta_0 + t * self._beta
    
    def int_beta(self, t):
        return (1-t) * (self.beta_0 + 0.5 * (1+t) * self._beta)
        
    def drift_coeff(self, t):
        numer = (1-t) * self.rbeta + t
        denom = (1-t) * (self.rbeta + 0.5 * (1+t) * (1-self.rbeta))
        return numer / denom


class Geometric(Scheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def beta(self, t):
        return self.beta_0**(1-t) * self.beta_f**t
    
    def int_beta(self, t):
        if self.beta_0 == self.beta_f:
            return self.beta_0 * (1-t)
        else:
            r = self.beta_f / self.beta_0
            return self.beta_0 * (r - r**t) / np.log(r)
        
    def drift_coeff(self, t):
        if self.beta_0 == self.beta_f:
            return 1 / (1-t)
        else:
            r = self.beta_f / self.beta_0
            return np.log(r) / (r**(1-t) - 1)
        

class Cosine(Scheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def beta(self, t):
        cos_t = (np.pi/2 * (1-t)).cos()
        return self.beta_0 * (1 - cos_t) + self.beta_f *  cos_t
    
    def int_beta(self, t):
        return self.beta_0 * (1-t) + 2 / np.pi * self._beta * (np.pi/2 * (1-t)).sin()
        
    def drift_coeff(self, t):
        return self.beta(t) / self.int_beta(t)