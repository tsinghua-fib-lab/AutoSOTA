"""
Wrapper for Decentralized 1D Fisher-KPP Dynamics.
Enables controlled simulations via a ControlNet policy using native JAX.
""" 
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import tesseracts.solverFKPP_decentralized.solver as solver

class PDEDynamics:
    def __init__(self, policy_apply_fn):
        """
        Initializes the dynamics wrapper for Decentralized 1D Fisher-KPP.
        
        Args:
            policy_apply_fn: The .apply method of your ControlNet.
        """
        self.policy_apply_fn = policy_apply_fn

    def unroll_controlled(self, z_init, xi_init, z_target, params, t_steps, key=jax.random.PRNGKey(0), noise_u=0.0, noise_z=0.0, nu=0.005, rho=3.0):
        """
        Performs a FULL controlled FKPP simulation in ONE call using the native JAX solver.
        The policy dictates agent movement and forcing intensities at each step.
        """
        return solver.solve_with_policy(
            z_init, 
            xi_init, 
            z_target, 
            params, 
            self.policy_apply_fn, 
            t_steps,
            key=key,         # Pass the key
            noise_u=noise_u, # Pass actuator noise
            noise_z=noise_z,  # Pass sensor noise
            nu=nu,           # diffusion coefficient
            rho=rho          # population growth rate
        )