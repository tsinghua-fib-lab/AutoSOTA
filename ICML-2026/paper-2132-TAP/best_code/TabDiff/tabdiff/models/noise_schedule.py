import abc

import torch
import torch.nn as nn


class Noise(abc.ABC, nn.Module):
  """
  Baseline forward method to get the total + rate of noise at a timestep
  """
  def forward(self, t):
    # Assume time goes from 0 to 1
    return self.total_noise(t), self.rate_noise(t)

  @abc.abstractmethod
  def total_noise(self, t):
    """
    Total noise ie \int_0^t g(t) dt + g(0)
    """
    pass
  

class LogLinearNoise(Noise):
  """Log Linear noise schedule.
    
  """
  def __init__(self, eps_max=1e-3, eps_min=1e-5, **kwargs):
    super().__init__()
    self.eps_max = eps_max
    self.eps_min = eps_min
    self.sigma_max = self.total_noise(torch.tensor(1.0))
    self.sigma_min = self.total_noise(torch.tensor(0.0))
    
  def k(self):
    return torch.tensor(1)

  def rate_noise(self, t):
    return (1 - self.eps_max - self.eps_min) / (1 - ((1 - self.eps_max - self.eps_min) * t + self.eps_min))

  def total_noise(self, t):
    """
    sigma_min=-log(1-eps_min), when t=0
    sigma_max=-log(eps_max), when t=1
    """
    return -torch.log1p(-((1 - self.eps_max - self.eps_min) * t + self.eps_min))
  
class PowerMeanNoise(Noise):
  """The noise schedule using the power mean interpolation function.
  
  This is the schedule used in EDM
  """
  def __init__(self, sigma_min=0.002, sigma_max=80, rho=7, **kwargs):
    super().__init__()
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.raw_rho = rho
    
  def rho(self):
    # Return the softplus-transformed rho for all num_numerical values
    return torch.tensor(self.raw_rho)

  def total_noise(self, t):
    sigma = (self.sigma_min ** (1/self.rho()) + t * (
                self.sigma_max ** (1/self.rho()) - self.sigma_min ** (1/self.rho()))).pow(self.rho())
    return sigma
  
  def inverse_to_t(self, sigma):
    t = (sigma.pow(1/self.rho()) - self.sigma_min ** (1/self.rho())) / (self.sigma_max ** (1/self.rho()) - self.sigma_min ** (1/self.rho()))
    return t


class PowerMeanNoise_PerColumn(nn.Module):

  def __init__(self, num_numerical, sigma_min=0.002, sigma_max=80, rho_init=1, rho_offset=2, **kwargs):
    super().__init__()
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.num_numerical = num_numerical
    self.rho_offset = rho_offset
    self.rho_raw = nn.Parameter(torch.tensor([rho_init] * self.num_numerical, dtype=torch.float32))

  def rho(self):
    # Return the softplus-transformed rho for all num_numerical values
    return nn.functional.softplus(self.rho_raw) + self.rho_offset

  def total_noise(self, t):
    """
    Compute total noise for each t in the batch for all num_numerical rhos.
    t: [batch_size]
    Returns: [batch_size, num_numerical]
    """
    batch_size = t.size(0)

    rho = self.rho()

    sigma_min_pow = self.sigma_min ** (1 / rho)  # Shape: [num_numerical]
    sigma_max_pow = self.sigma_max ** (1 / rho)  # Shape: [num_numerical]

    sigma = (sigma_min_pow + t * (sigma_max_pow - sigma_min_pow)).pow(rho)  # Shape: [batch_size, num_numerical]

    return sigma

  def rate_noise(self, t):
    return None

  def inverse_to_t(self, sigma):
    """
    Inverse function to map sigma back to t, with proper broadcasting support.
    sigma: [batch_size, num_numerical] or [batch_size, 1]
    Returns: t: [batch_size, num_numerical]
    """
    rho = self.rho()

    sigma_min_pow = self.sigma_min ** (1 / rho)  # Shape: [num_numerical]
    sigma_max_pow = self.sigma_max ** (1 / rho)  # Shape: [num_numerical]

    # To enable broadcasting between sigma and the per-column rho values, expand rho where needed.
    t = (sigma.pow(1 / rho) - sigma_min_pow) / (sigma_max_pow - sigma_min_pow)

    return t


class LogLinearNoise_PerColumn(nn.Module):

  def __init__(self, num_categories, eps_max=1e-3, eps_min=1e-5, k_init=-6, k_offset=1, **kwargs):

    super().__init__()
    self.eps_max = eps_max
    self.eps_min = eps_min
    # Use softplus to ensure k is positive
    self.num_categories = num_categories
    self.k_offset = k_offset
    self.k_raw = nn.Parameter(torch.tensor([k_init] * self.num_categories, dtype=torch.float32))

  def k(self):
    return torch.nn.functional.softplus(self.k_raw) + self.k_offset

  def rate_noise(self, t, noise_fn=None):
    """
    Compute rate noise for all categories with broadcasting.
    t: [batch_size]
    Returns: [batch_size, num_categories]
    """
    k = self.k()  # Shape: [num_categories]

    numerator = (1 - self.eps_max - self.eps_min) * k * t.pow(k - 1)
    denominator = 1 - ((1 - self.eps_max - self.eps_min) * t.pow(k) + self.eps_min)
    rate = numerator / denominator  # Shape: [batch_size, num_categories]

    return rate

  def total_noise(self, t, noise_fn=None):
    k = self.k()  # Shape: [num_categories]
    
    total_noise = -torch.log1p(-((1 - self.eps_max - self.eps_min) * t.pow(k) + self.eps_min))

    return total_noise
