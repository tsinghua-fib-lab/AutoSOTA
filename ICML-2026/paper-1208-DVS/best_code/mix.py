import abc
import torch

class DiffusionMixture(object):
  def __init__(self, bridge, drift_coeff, sigma_0, sigma_1, N=1000):
    """Construct a Mixture of Diffusion Bridges.
    Args:
      bridge: type of bridge
      sigma_0, simga_1: hyperparameters for the noise schedule
      N: number of discretization steps
    """
    super().__init__()
    self.bridge_type = bridge
    self.drift_coeff = drift_coeff
    self.sigma_0 = sigma_0
    self.sigma_1 = sigma_1
    self.N = N

  def diffusion(self, t):
    sigma_t = torch.sqrt((1-t) * self.sigma_0**2 + t * self.sigma_1**2)
    return sigma_t 

  def bridge(self, destination):
    bridge_args = {'drift_coeff': self.drift_coeff, 'sigma_0': self.sigma_0, 
                    'sigma_1': self.sigma_1, 'destination': destination, 'N': self.N}
    if 'BB' in self.bridge_type:
      bridge = BrownianBridge(**bridge_args)
    elif 'OU' in self.bridge_type:
      bridge = OUBridge(**bridge_args)
    else:
      raise NotImplementedError(f'Bridge type {self.bridge_type} not implemented.')
    return bridge


class Bridge(abc.ABC):
  """Diffusion Bridge abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, drift_coeff, sigma_0, sigma_1, destination, N):
    """Construct an Diffusion Bridge.
    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.drift_coeff = drift_coeff
    self.sigma_0 = sigma_0
    self.sigma_1 = sigma_1
    self.dest = destination
    self.N = N

  # -------- Do not scale the bridge here --------
  @property
  def T(self):
    return 1.

  def diffusion(self, t):
    sigma_t = torch.sqrt((1-t) * self.sigma_0**2 + t * self.sigma_1**2)
    return sigma_t

  # -------- sigma_t**2 --------
  def variance(self, t):
    variance = (1-t) * self.sigma_0**2 + t * self.sigma_1**2
    return variance

  # -------- Integrate sigma_t ** 2 from time 0 to t --------
  def beta_t(self, t):
    beta_t = t * self.sigma_0**2 - 0.5 * t**2 * (self.sigma_0**2 - self.sigma_1**2)
    return beta_t

  @abc.abstractmethod
  def sde(self, z, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, z0, t):
    """Parameters to determine the marginal distribution of the Bridge, $p_t(z)$."""
    pass


class BrownianBridge(Bridge):
  def __init__(self, drift_coeff, sigma_0, sigma_1, destination, N=1000):
    """Construct a Brownian Bridge.
    Args:
      destination : terminal point
      N: number of discretization steps
    """
    super().__init__(drift_coeff, sigma_0, sigma_1, destination, N)

  # -------- sigma_t**2 / (beta_1 - beta_t) --------
  def drift_time_scaled(self, t):
    seps = 1 - (self.sigma_1 / self.sigma_0)**2 
    drift_time_scaled = (1 - t * seps) / (1 - t - 0.5 * seps * (1-t**2))
    return drift_time_scaled

  def sde(self, z, t):
    drift = (self.dest - z) * self.drift_time_scaled(t)[:, None, None]
    diffusion = self.diffusion(t)
    return drift, diffusion

  # -------- mean, std of the perturbation kernel --------
  def marginal_prob(self, z0, t):
    beta_1 = self.beta_t(torch.ones_like(t))
    beta = self.beta_t(t)
    mean = (z0 * (beta_1 - beta)[:, None, None] + self.dest * (beta)[:, None, None]) / (beta_1)[:, None, None] 
    std = torch.sqrt((beta_1 - beta) * beta / beta_1)
    return mean, std

  def prior_sampling(self, shape, device='cpu'):
    return torch.randn(*shape, device=device)

  def prior_sampling_sym(self, shape, device='cpu'):
    z = torch.randn(*shape, device=device).triu(1)
    return z + z.transpose(-1,-2)

  # Compute sigma_t / (beta_1 - beta_t)
  def loss_coeff(self, t):
    loss_coeff = self.drift_time_scaled(t) / self.diffusion(t)
    return loss_coeff


class OUBridge(Bridge):
  def __init__(self, drift_coeff, sigma_0, sigma_1, destination, N=1000):
    """Construct a Ornstein–Uhlenbeck Bridge.
    Args:
      destination : terminal point
      N: number of discretization steps
    """
    super().__init__(drift_coeff, sigma_0, sigma_1, destination, N)

  def alpha_t(self, t):
    alpha_t = -0.5 * self.drift_coeff 
    return alpha_t

  # -------- Integration of alpha_t from time s to t --------
  def alpha_sum(self, s, t):
    alpha_sum = -0.5 * self.drift_coeff * (t - s)
    return alpha_sum

  def a_ou(self, s, t):
    log_coeff = self.alpha_sum(self.beta_t(s), self.beta_t(t))
    return torch.exp(log_coeff)

  def v_ou(self, s, t):
    v_ou = 0.5 * ( self.a_ou(s,t)**2 / self.alpha_t(self.beta_t(s)) - 1. /  self.alpha_t(self.beta_t(t)))
    return v_ou 

  # Compute a_t1 / v_t1
  def a_over_v(self, t):
    ones = torch.ones_like(t)
    a_t1 = self.a_ou(t, ones)
    result = self.drift_coeff / (1./a_t1 - a_t1)
    return result

  # -------- \sigma_t**2 * \nabla_{z} \log p_{1|t}(x|z) --------
  def drift_adjustment(self, z, t):
    ones = torch.ones_like(t)
    a_t1 = self.a_ou(t, ones)
    gamma = a_t1 * self.a_over_v(t)
    adjustment = (self.dest / a_t1[:, None, None] - z ) * (self.variance(t) * gamma)[:, None, None] 
    return adjustment

  def sde(self, z, t):
    diffusion = self.diffusion(t)
    drift = (self.alpha_t(t) * self.variance(t))[:, None, None] * z + self.drift_adjustment(z, t)
    return drift, diffusion

  # -------- mean, std of the perturbation kernel using Eq. (55) --------
  # def marginal_prob(self, z0, t):
  #   zeros = torch.zeros_like(t)
  #   ones = torch.ones_like(t)
  #   a_0t = self.a_ou(zeros, t)
  #   a_t1 = self.a_ou(t, ones)
  #   v_0t = self.v_ou(zeros, t)
  #   v_t1 = self.v_ou(t, ones)
  #   denom = v_t1 + v_0t * a_t1**2
  #   # -------- std --------
  #   std = torch.sqrt(v_0t * v_t1 / denom)
  #   # -------- mean --------
  #   coeff0 = v_t1 * a_0t / denom
  #   coeff1 = v_0t * a_t1 / denom
  #   mean = coeff0[:, None, None] * z0 + coeff1[:, None, None] * self.dest 
  #   return mean, std

  # -------- Used for Eq. (56) --------
  def phi(self, t):
    beta_t = t * self.sigma_0**2 - 0.5 * t**2 * (self.sigma_0**2 - self.sigma_1**2)
    return self.alpha_t(t) * beta_t

  # -------- mean, std of the perturbation kernel using Eq. (56) --------
  def marginal_prob(self, z0, t):
    ones = torch.ones_like(t)
    sinh_t = torch.sinh(self.phi(t))
    sinh_Tt = torch.sinh(self.phi(ones) - self.phi(t))
    sinh_T = torch.sinh(self.phi(ones))
    coeff0 = sinh_Tt / sinh_T
    coeff1 = sinh_t / sinh_T
    var = sinh_Tt * sinh_t / sinh_T / self.alpha_t(t)
    std = torch.sqrt(var)
    mean = coeff0[:, None, None] * z0 + coeff1[:, None, None] * self.dest 
    return mean, std

  def prior_sampling(self, shape, device='cpu'):
    return torch.randn(*shape, device=device)

  def prior_sampling_sym(self, shape, device='cpu'):
    z = torch.randn(*shape, device=device).triu(1) 
    return z + z.transpose(-1,-2)

  # Compute sigma_t * a_t1 / v_t1
  def loss_coeff(self, t):
    return self.diffusion(t) * self.a_over_v(t)

