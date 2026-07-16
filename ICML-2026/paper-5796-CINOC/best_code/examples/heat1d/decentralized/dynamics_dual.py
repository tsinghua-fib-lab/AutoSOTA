"""
Wrapper for Decentralized 1D Heat Equation Dynamics using Tesseract-JAX
Enables controlled simulations via a ControlNet policy, either through Tesseract runtime or native JAX (for fast prototyping).
""" 
import jax
import jax.numpy as jnp
from tesseracts.solverHeat_decentralized import solver

class PDEDynamics:
    def __init__(self, policy_apply_fn):
        self.policy_apply_fn = policy_apply_fn

    def unroll_controlled(self, z_init, xi_init, z_target, params, t_steps):
        """Performs a FULL controlled simulation in ONE call using native JAX."""
        return solver.solve_with_policy(
            z_init, xi_init, z_target, params, 
            self.policy_apply_fn, t_steps
        )