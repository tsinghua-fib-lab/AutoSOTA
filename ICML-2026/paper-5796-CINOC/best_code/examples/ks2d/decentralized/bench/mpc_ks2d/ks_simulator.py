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

from tesseracts.ks2d.solver import precompute_etdrk4_coeffs, ks_spectral_step_etdrk4, generate_random_noise_2d

class KSSolverJAX2D:
    """Wrapper that provides a persistent JAX solver state and single-step updates for 2D KS."""
    
    def __init__(self, N, L, dt, sigma, centers):
        self.N = N
        self.L = L
        self.dt = dt
        self.sigma = sigma
        self.centers = jnp.array(centers)
        
        self.dx = L / N
        self.kx = 2 * jnp.pi * jnp.fft.fftfreq(N, d=self.dx)
        self.ky = 2 * jnp.pi * jnp.fft.rfftfreq(N, d=self.dx)
        
        KX, KY = jnp.meshgrid(self.kx, self.ky, indexing='ij')
        
        # Save KX, KY to pass to the solver
        self.KX = KX
        self.KY = KY
        
        q_sq = KX**2 + KY**2
        self.L_linear = q_sq - q_sq**2
        
        k_max_x = jnp.max(jnp.abs(self.kx))
        k_max_y = jnp.max(jnp.abs(self.ky))
        mask_x = jnp.abs(KX) < (2.0/3.0 * k_max_x)
        mask_y = jnp.abs(KY) < (2.0/3.0 * k_max_y)
        self.dealias_mask = (mask_x & mask_y).astype(jnp.float32)

        self.etdrk4_coeffs = precompute_etdrk4_coeffs(self.L_linear, self.dt)
        
        # Jit the step function for speed
        @jit
        def _step(u_curr, u_hat_curr, u_control):
            u_hat_next, u_next = ks_spectral_step_etdrk4(
                u_hat_curr, u_curr, self.centers, u_control, 
                self.KX, self.KY, self.etdrk4_coeffs, self.dealias_mask,
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
        """Burn-in from a smooth random perturbation to reach the chaotic attractor."""
        key = jax.random.PRNGKey(seed)
        u0 = generate_random_noise_2d(key, self.N, self.L, scale=1.0)
        
        u_curr = jnp.array(u0)
        u_hat_curr = jnp.fft.rfftn(u_curr)
        u_control = jnp.zeros(len(self.centers))
        
        # Unroll explicitly or use scan
        def _scan_step(carry, _):
            u, u_hat = carry
            u_next, u_hat_next = self._jitted_step(u, u_hat, u_control)
            return (u_next, u_hat_next), None
            
        (u_final, u_hat_final), _ = jax.lax.scan(_scan_step, (u_curr, u_hat_curr), None, length=steps)
            
        return np.array(u_final), np.array(u_hat_final)