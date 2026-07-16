"""
Dynamics module for Decentralized 2D Heat Equation Control using native JAX.
Enables controlled simulations via a DecentralizedControlNet policy.
"""
import jax
import jax.numpy as jnp
from tesseracts.solverHeat2D_decentralized import solver

class PDEDynamics:
    def __init__(self, policy_apply_fn):
        """
        Initializes the dynamics with a control policy.
        
        Args:
            policy_apply_fn: The function to apply the DecentralizedControlNet policy.
        """
        self.policy_apply_fn = policy_apply_fn

    def unroll_controlled(self, z_init, xi_init, z_target, params, t_steps):
        """
        Performs a FULL controlled simulation in ONE call using native JAX.
        
        Args:
            z_init: Initial state of the system.
            xi_init: Initial internal controller state.
            z_target: Target state/setpoint.
            params: Policy parameters (PyTree).
            t_steps: Number of time steps to simulate.
            
        Returns:
            A tuple of (z_trajectory, xi_trajectory, u_trajectory, v_trajectory).
        """
        # Native JAX handles the dict 'params' directly without flattening
        return solver.solve_with_policy(
            z_init, 
            xi_init, 
            z_target, 
            params,
            self.policy_apply_fn, 
            t_steps
        )