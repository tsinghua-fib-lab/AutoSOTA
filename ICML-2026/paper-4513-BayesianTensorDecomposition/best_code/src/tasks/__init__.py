'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''
import torch
from abc import ABC, abstractmethod


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)

    def likelihood_gradient(self, data, y, sigma, **kwargs):
        return self.transpose(self.forward(data, **kwargs) - y).reshape(*data.shape) / sigma**2


class LinearSVDOperator(LinearOperator):
    def A(self, vec):
        """
        Multiplies the input vector by A
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])
    
    def At(self, vec):
        """
        Multiplies the input vector by A transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
    
    def forward(self, vec, **kwargs):
        return self.A(vec)
    
    def transpose(self, vec):
        return self.At(vec)

    def A_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of A
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        
        factors = 1. / singulars
        factors[singulars == 0] = 0.
        
#         temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))
    
    def A_pinv_eta(self, vec, eta):
        """
        Multiplies the input vector by the pseudo inverse of A with factor eta
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        factors = singulars / (singulars*singulars+eta)
#         print(temp.size(), factors.size(), singulars.size())
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))
    
    def proximal_generator(self, x, y, sigma, rho):
        singulars = self.add_zeros(self.singulars().unsqueeze(0))
        Qx_inv_eigvals = 1 / (singulars**2 / sigma**2 + 1 / rho**2)
        noise = self.V(torch.sqrt(Qx_inv_eigvals) * torch.randn_like(x).reshape(x.shape[0], -1))
        mu_x = self.V(Qx_inv_eigvals * self.Vt(self.At(y / sigma**2) + (x.reshape(x.shape[0], -1) / rho**2)))
        return (mu_x + noise).reshape(*x.shape)

    def proximal_for_admm(self, x, y, rho):
        singulars = self.add_zeros(self.singulars().unsqueeze(0))
        inv_eigvals = 1 / (singulars**2 + rho)
        ret = self.V(inv_eigvals * self.Vt(self.At(y) + rho * x.reshape(x.shape[0], -1)))
        return ret.reshape(*x.shape)

    def initialize(self, gt, y):
        return self.A_pinv(y).reshape(y.shape[0], self.channels, self.img_dim, self.img_dim)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 


# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper


def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='no_noise')
class NoNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma

    @property
    def display_name(self):
        return f'no_noise_sigma={self.sigma}'
    
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma, input_snr=None):
        self.sigma = sigma
        self.input_snr = input_snr
    
    @property
    def display_name(self):
        if self.input_snr is None:
            return f'sigma={self.sigma}'
        else:
            return f'isnr={self.input_snr}_sigma={self.sigma}'
    
    def forward(self, data):
        if self.input_snr is None:
            return data + torch.randn_like(data, device=data.device) * self.sigma
        else:
            noise = torch.randn_like(data)
            noise_norm = torch.norm(data) * 10 ** (-self.input_snr / 20)
            scale = noise_norm / torch.norm(noise)
            scaled_noise = noise * scale
            print(f'input snr mode of gaussian noise: {scale} in input sigma')
            return data + scaled_noise

from .completion import Completion
