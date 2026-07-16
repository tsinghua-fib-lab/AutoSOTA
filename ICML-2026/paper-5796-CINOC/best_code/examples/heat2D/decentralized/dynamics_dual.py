"""
Dynamics module for Decentralized 2D Heat Equation Control using Native JAX.
Enables controlled simulations via a ControlNet policy.
"""
import jax
import jax.numpy as jnp
from tesseracts.solverHeat2D_decentralized import solver 

class PDEDynamics:
    def __init__(self, policy_apply_fn):
        """
        Initializes the dynamics module.
        
        Args:
            policy_apply_fn: The JAX-compatible apply function for the policy network.
        """
        self.policy_apply_fn = policy_apply_fn

    def unroll_controlled(self, z_init, xi_init, z_target, params, t_steps):
        """
        Performs a FULL controlled simulation in ONE call using native JAX.
        
        Args:
            z_init: Initial state.
            xi_init: Initial auxiliary state (if any).
            z_target: Target state.
            params: Policy network parameters (PyTree).
            t_steps: Number of time steps to simulate.
            
        Returns:
            Tuple of trajectories: (z, xi, u, v)
        """
        # Native JAX handles the dict/PyTree 'params' directly without flattening
        return solver.solve_with_policy(
            z_init, xi_init, z_target, params,
            self.policy_apply_fn, t_steps
        )