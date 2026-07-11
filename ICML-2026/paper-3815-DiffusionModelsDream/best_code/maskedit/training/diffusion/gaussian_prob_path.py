import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable
from abc import ABC, abstractmethod
from jax import vmap, jacrev

Array = jax.Array

class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert jnp.allclose(self(jnp.zeros((1, 1))), jnp.zeros((1, 1)))
        # Check alpha_1 = 1
        assert jnp.allclose(self(jnp.ones((1, 1))), jnp.ones((1, 1)))

    @abstractmethod
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - alpha_t (num_samples, 1)
        """
        raise NotImplementedError

    def dt(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """
        dt = vmap(jacrev(self))(t)
        return dt.reshape(-1, 1)


class Beta(ABC):
    def __init__(self):
        assert jnp.allclose(self(jnp.zeros((1, 1))), jnp.ones((1, 1)))
        assert jnp.allclose(self(jnp.ones((1, 1))), jnp.zeros((1, 1)))

    @abstractmethod
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluates beta_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """
        raise NotImplementedError

    def dt(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt beta_t (num_samples, 1)
        """
        dt = vmap(jacrev(self))(t)
        return dt.reshape(-1, 1)


class LinearAlpha(Alpha):
    """
    Implements alpha_t = t
    """
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        return t

    def dt(self, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones_like(t)


class SquareRootBeta(Beta):
    """
    Implements beta_t = sqrt(1 - t)
    """
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(1.0 - t)

    def dt(self, t: jnp.ndarray) -> jnp.ndarray:
        """d/dt sqrt(1 - t) = -1 / (2 * sqrt(1 - t))"""
        eps = 1e-8
        return -0.5 / (jnp.sqrt(1.0 - t) + eps)


class BetaVP(Beta):
    """
    Implements beta_t = sqrt(1 - t**2)
    """
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(1.0 - t**2)

    def dt(self, t: jnp.ndarray) -> jnp.ndarray:
        """d/dt sqrt(1 - t) = -1 / (2 * sqrt(1 - t))"""
        eps = 1e-12
        return -t / (jnp.sqrt(1.0 - t**2) + eps)

class LinearBeta(Beta):
    """
    Implements beta_t = sqrt(1 - t)
    """
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        return 1.0 - t

    def dt(self, t: jnp.ndarray) -> jnp.ndarray:
        return - jnp.ones_like(t)


@dataclass
class GaussianConditionalPath:
    """
    Gaussian probability path: x_t = alpha(t) * z + beta(t) * eps
    """
    alpha: Alpha
    beta:  Beta
    sigma: float = 2.5

    def __init__(self, sigma=2.5, alpha=None, beta=None):
        beta = SquareRootBeta()
        self.sigma = (lambda t: sigma) if isinstance(sigma, float) else (lambda t: beta(t))
        self.alpha = alpha if alpha is not None else LinearAlpha()
        self.beta  = beta  if beta  is not None else BetaVP()

    def _b(self, v: Array, x: Array) -> Array:
        while v.ndim < x.ndim:
            v = v[..., None]
        return v

    def add_noise(self, rng, z0: Array, t: Array) -> tuple[Array, Array, Array]:
        """
        x_t = alpha(t) * z0 + beta(t) * eps, z0 is the sampled data
        Returns (x_t, eps, std) with std = beta(t) (marginal std)
        """
        # t can be (B,) or (B,1). z0 is (B, ..., C)
        eps = jax.random.normal(rng, shape=z0.shape, dtype=z0.dtype)
        a = self.alpha(t)         # (B,) or (B,1)
        b = self.beta(t)
        a_b = self._b(a, z0)
        b_b = self._b(b, z0)
        x_t = a_b * z0 + b_b * eps
        return x_t, eps, b_b

    def prior_sample(self, rng, shape: tuple[int, ...]) -> Array:
        """
        Simple prior p_simple = N(0, I) at t=0 (alpha(0)=0, beta(0)=1).
        """
        return jax.random.normal(rng, shape)

    def step(self, rng, eps_pred, x_t, t, dt, shape,
         t_eps: float = 1e-4, beta_min: float = 1e-6, alpha_min: float = 1e-12):
        """
        One Euler–Maruyama step, numerically safe at t=0.
        """
        # Broadcast helpers
        def B(v): 
            v = jnp.asarray(v)
            while v.ndim < x_t.ndim: v = v[..., None]
            return v
    
        # Clamp t away from endpoints for the "general" branch
        t_clamped = jnp.clip(t, t_eps, 1.0 - t_eps)
    
        # Values at current t (clamped) for the general expression
        a     = self.alpha(t_clamped)
        b     = self.beta(t_clamped)
        a_dt  = self.alpha.dt(t_clamped)
        b_dt  = self.beta.dt(t_clamped)
    
        # Safe denominators
        a_safe = jnp.maximum(a, alpha_min)
        b_safe = jnp.maximum(b, beta_min)
    
        # Score from epsilon network: s = -eps / beta
        score = - eps_pred / B(b_safe)
    
        # General drift (matches eq. (56))
        r = a_dt / a_safe  # this is d/dt log alpha for t>0
        drift_general = ((b * b * r - b_dt * b + 0.5 * self.sigma(t)**2) * score
                         + B(r) * x_t)
    
        # Exact t→0 drift: drift0 = (-beta'(0) - sigma^2/2) * x
        b_dt_at_0 = self.beta.dt(jnp.zeros_like(t))
        drift_t0  = B(-b_dt_at_0 - 0.5 * self.sigma(t)**2) * x_t
    
        # Blend: use limit near t=0; otherwise use general form
        use_t0 = (t <= t_eps).astype(x_t.dtype)
        drift  = B(use_t0) * drift_t0 + B(1.0 - use_t0) * drift_general
    
        # Diffusion term
        noise = jax.random.normal(rng, shape=shape, dtype=x_t.dtype)
        return x_t + drift * dt + self.sigma(t) * jnp.sqrt(dt) * noise
    
    def weight_fn(self, t):
        return t

    def output_scale_fn(self, t, x):
        return x
    