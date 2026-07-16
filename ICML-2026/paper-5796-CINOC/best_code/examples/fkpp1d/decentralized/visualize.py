"""Visualization script for Decentralized 1D Fisher-KPP Control using native JAX and DecentralizedControlNet policy."""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
import flax.serialization
import os
from pathlib import Path

jax.config.update("jax_platform_name", "cpu")

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from dynamics_dual import PDEDynamics 
from models.policy import DecentralizedControlNet
import data_utils

def load_params(model, filepath, n_pde=100, n_agents=8):
    """Loads the weights from a msgpack file into the DecentralizedControlNet PyTree."""
    with open(filepath, 'rb') as f:
        serialized_bytes = f.read()
    key = jax.random.PRNGKey(0)
    # Match the 3-arg signature of DecentralizedControlNet
    init_params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))
    return flax.serialization.from_bytes(init_params, serialized_bytes)

def main():
    # --- 1. Setup ---
    n_pde, n_agents, T_steps = 100, 20, 300 
    
    # Create output directory
    output_dir = Path("figures/images/vanilla")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = DecentralizedControlNet(features=(64, 64))
    dynamics = PDEDynamics(policy_apply_fn=model.apply)
    
    try:
        params = load_params(model, 'decentralized_params.msgpack', n_pde, n_agents)
    except FileNotFoundError:
        print("Error: params file not found. Please run training script first.")
        return

    # --- 2. Generation & Rollout ---
    key = jax.random.PRNGKey(42) # Different seed for testing
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    
    for i in range(2): 
        key, subkey1, subkey2 = jax.random.split(key, 3)
        _, z_init = data_utils.generate_grf(subkey1, n_points=n_pde, length_scale=0.2)
        _, z_target = data_utils.generate_grf(subkey2, n_points=n_pde, length_scale=0.4)
        xi_init = jnp.linspace(0.2, 0.8, n_agents) 
        
        # REPLACED: No more manual scan loop. One call handles the full physics + policy rollout.
        z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
            z_init, xi_init, z_target, params, T_steps
        )
        
        x_grid = jnp.linspace(0, 1, n_pde)
        
        # --- Column 1: State Evolution ---
        ax = axes[i, 0]
        ax.plot(x_grid, z_target, 'k--', label='Target', linewidth=2)
        ax.plot(x_grid, z_init, 'b:', label='Initial', alpha=0.6)
        
        # Ghosting effect: intermediate time steps
        for t in range(0, T_steps, 15): 
            ax.plot(x_grid, z_traj[t], 'g-', alpha=0.1)
            
        ax.plot(x_grid, z_traj[-1], 'r-', label='Final Output', linewidth=2)
        
        # Scatter plot of final actuator positions
        act_idx = jnp.clip((xi_traj[-1] * n_pde).astype(int), 0, n_pde-1)
        ax.scatter(xi_traj[-1], z_traj[-1, act_idx], color='red', zorder=5, label='Actuators')
        
        ax.set_title(f"Ex {i+1}: FKPP State Evolution")
        ax.set_ylim([0, 1.1]) # FKPP is typically bounded between 0 and 1
        ax.legend(fontsize='x-small')
        
        # --- Column 2: Forcing Intensity (u) ---
        ax2 = axes[i, 1]
        ax2.plot(u_traj)
        ax2.set_title("Control: Intensity (u)")
        ax2.set_xlabel("Time")
        
        # --- Column 3: Control Velocity (v) ---
        ax3 = axes[i, 2]
        ax3.plot(v_traj)
        ax3.set_title("Control: Velocity (v)")
        ax3.set_xlabel("Time")

        # --- Column 4: Actuator Paths (xi) ---
        ax4 = axes[i, 3]
        for j in range(n_agents):
            ax4.plot(xi_traj[:, j], alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.3)
        ax4.set_title("Agent Trajectories ($\\xi$)")
        ax4.set_ylim([-0.05, 1.05])
        ax4.set_xlabel("Time")
        
    plt.tight_layout()
    save_path = output_dir / 'fkpp_decentralized_visual.png'
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    main()