import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

# Add root path to import tesseracts solver
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../'))
if root_path not in sys.path:
    sys.path.append(root_path)

from tesseracts.ks1d.solver import ks_spectral_step

class KSSolverJAX:
    """Wrapper that provides a persistent JAX solver state and single-step updates."""
    
    def __init__(self, N, L, dt, sigma, centers):
        self.N = N
        self.L = L
        self.dt = dt
        self.sigma = sigma
        self.centers = jnp.array(centers)
        
        self.dx = L / N
        self.k = 2 * jnp.pi * jnp.fft.rfftfreq(N, d=self.dx)
        self.L_linear = self.k**2 - self.k**4
        
        # Jit the step function for speed
        @jit
        def _step(u_curr, u_hat_curr, u_control):
            u_hat_next, u_next = ks_spectral_step(
                u_hat_curr, u_curr, self.centers, u_control, 
                self.k, self.L_linear,
                N=self.N, L=self.L, dt=self.dt, sigma=self.sigma
            )
            return u_next, u_hat_next
            
        self._jitted_step = _step
        
    def step(self, u_curr, u_hat_curr, u_control):
        """Perform a single integration step."""
        u_next, u_hat_next = self._jitted_step(
            jnp.array(u_curr), 
            jnp.array(u_hat_curr), 
            jnp.array(u_control)
        )
        return np.array(u_next), np.array(u_hat_next)

    def generate_chaotic_ic(self, seed=42, steps=2000):
        """Burn-in from a sine wave perturbation to reach the chaotic attractor."""
        np.random.seed(seed)
        x = np.linspace(0, self.L, self.N, endpoint=False)
        u0 = (np.sin(2 * np.pi * x / self.L) + 
              0.5 * np.sin(4 * np.pi * x / self.L) + 
              0.1 * np.random.randn(self.N))
        
        u_curr = jnp.array(u0)
        u_hat_curr = jnp.fft.rfft(u_curr)
        u_control = jnp.zeros(len(self.centers))
        
        # Unroll explicitly or use scan if it's too slow, but normally compiling a loop is okay.
        # jax.lax.scan is better.
        def _scan_step(carry, _):
            u, u_hat = carry
            return self._jitted_step(u, u_hat, u_control), None
            
        (u_final, u_hat_final), _ = jax.lax.scan(_scan_step, (u_curr, u_hat_curr), None, length=steps)
            
        return np.array(u_final), np.array(u_hat_final)
