"""
Data Utilities for 2D Kuramoto-Sivashinsky (KS)
Generates chaotic initial conditions by starting from random noise and 
evolving the system autonomously (no control) for a warm-up period.
"""
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt

# Enable 64-bit precision to prevent numerical drift in spectral integration
jax.config.update("jax_enable_x64", True)

import tesseracts.ks2d.solver as solver 

class PDEDynamics2D:
    def __init__(self, policy_apply_fn):
        """
        Initializes the dynamics wrapper for Centralized 2D KS.
        
        Args:
            policy_apply_fn: The .apply method of your ControlNet (JAX/Flax).
        """
        self.policy_apply_fn = policy_apply_fn

    def unroll_controlled(
        self, 
        u_init, 
        xi_fixed, 
        u_target, 
        params, 
        t_steps,
        # Exposing length, resolution, and physics params
        N_grid=128,
        L=64.0,
        dt=0.05,
        sigma=1.0,
        substeps=1
    ):
        """
        Performs a FULL controlled KS simulation in ONE call (2D).
        
        Args:
            t_steps: Number of CONTROL steps.
            substeps: Number of PHYSICS steps per control step.
            dt: Time step size for a SINGLE physics substep.
            
            Total physical time = t_steps * substeps * dt
        """
        # Ensure inputs are JAX arrays
        u_init = jax.numpy.array(u_init)
        xi_fixed = jax.numpy.array(xi_fixed)
        u_target = jax.numpy.array(u_target)

        # Call the 2D solver function
        return solver.solve_with_policy(
            u_init, 
            xi_fixed, 
            u_target, 
            params, 
            self.policy_apply_fn, 
            t_steps,
            substeps=substeps,  # Pass it down
            N_grid=N_grid,
            L=L,
            dt=dt,
            sigma=sigma
        )