"""
Differentiable Wrapper for 2D Turbulence (Vorticity-Streamfunction).
Uses jax.lax.scan to allow backpropagation through the RK4 physics steps.
"""
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from tesseracts.turbulence2d import solver

class PDEDynamics2D:
    def __init__(self, policy_apply_fn=None):
        """
        Args:
            policy_apply_fn: Function (params, obs) -> actions.
        """
        self.policy_apply_fn = policy_apply_fn

    def unroll_controlled(
        self, 
        w_hat_init, 
        xi_fixed,    # Fixed policy params (e.g. actuator positions if embedded)
        params,      # Trainable params (Neural Net weights)
        t_steps,     # Number of Control Steps
        # Physics Config
        N_grid=128,
        L=1.0,
        dt=0.02,
        substeps=16, # Physics steps per control step
        viscosity=5e-5,
        actuator_grid_shape=(8, 8),
        sigma=0.05#0.2
    ):
        """
        Differentiable Unroll for Training.
        Returns:
            w_phys_traj: Physical vorticity history (for loss calculation).
            u_ctrl_traj: Control action history (for effort loss).
        """
        # 1. Setup Grid & Constants
        kx, ky, k_sq, k_inv = solver.get_spectral_grid(N_grid, L)
        dt_phys = dt / substeps

        # 2. Precompute Actuator Forcing Profiles (Batch, N, N)
        # We calculate centers once here to ensure differentiation works if needed
        nx_act, ny_act = actuator_grid_shape
        x_c = jnp.linspace(0, L, nx_act, endpoint=False) + L/(2*nx_act)
        y_c = jnp.linspace(0, L, ny_act, endpoint=False) + L/(2*ny_act)
        xv, yv = jnp.meshgrid(x_c, y_c)
        centers_flat = jnp.stack([xv.flatten(), yv.flatten()], axis=1)
        
        # Helper to compute Gaussian blobs in spectral space
        forcing_hat = solver.compute_forcing_profile(
            centers_flat[:, 0], centers_flat[:, 1], N_grid, L, sigma
        )

        # 3. Define the Control Step (Scan Body)
        def control_step(carry, _):
            w_hat_curr, t_curr = carry
            
            # --- A. Observation ---
            # Convert Spectral -> Physical (Observation is Real Vorticity)
            w_phys = jnp.fft.ifft2(w_hat_curr).real
            
            # Add batch dim: (N, N) -> (1, N, N)
            obs = w_phys[None, :, :]
            
            # --- B. Policy Inference ---
            # Call the Flax/Haiku apply function
            # Output shape: (n_actuators,)
            actions = self.policy_apply_fn(params, xi_fixed, obs)
            actions = actions.squeeze() 
            
            # --- C. Physics Evolution (Sub-stepping) ---
            def physics_loop(i, w):
                return solver.rk4_step(
                    w, dt_phys, 
                    kx, ky, k_sq, k_inv, 
                    viscosity, forcing_hat, actions
                )
            
            # Run physics bursts
            w_hat_next = jax.lax.fori_loop(0, substeps, physics_loop, w_hat_curr)
            
            t_next = t_curr + dt
            
            # Return: (Carry), (Output: State for Loss, Actions for Loss)
            return (w_hat_next, t_next), (w_phys, actions)

        # 4. Execute Scan
        _, (w_phys_traj, u_ctrl_traj) = jax.lax.scan(
            control_step,
            (w_hat_init, 0.0),
            None,
            length=t_steps
        )
        
        return w_phys_traj, u_ctrl_traj