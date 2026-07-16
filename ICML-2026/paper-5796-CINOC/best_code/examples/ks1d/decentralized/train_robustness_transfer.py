# We're not using this script actually.
"""
Robustness Transfer Experiment - KS 1D (L=32)
Trains three variants of the ControlNet (Clean, Low Noise, High Noise)
on a highly chaotic domain.
"""
from pathlib import Path
import sys

try:
    from train_utils import train
except ImportError:
    print("Error: Could not import 'train' from 'train_ks.py'.")
    print("Make sure the training script is in the same folder.")
    sys.exit(1)

# Experiment Output Directory
EXPERIMENT_DIR = Path("figures/ks_noise_experiments/robustness_transfer_L32")

# --- Experiment Constants (Tuned for KS L=32) ---
EXPERIMENT_CONSTANTS = {
    # Physics Settings
    "L_domain": 32.0,  
    "N_grid": 256,          
    "n_agents": 30,         
    
    # Training Settings
    "n_pde": 1024,         # pool size for initial conditions
    "batch_size": 32,
    "T_steps": 300,         
    "epochs": 500,         
}

# Define the noise configurations
# Note: KS state values are roughly [-3, 3], so noise 0.5 is significant.
NOISE_CONFIGS = {
    "baseline":       {"noise_u": 0.0,  "noise_z": 0.0},
    "low_noise":  {"noise_u": 0.05, "noise_z": 0.025},
    "medium_noise": {"noise_u": 0.1,  "noise_z": 0.05},
    "high_noise": {"noise_u": 0.5,  "noise_z": 0.25},
}

def run_all():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--- Starting KS-1D (L=32) Robustness Experiments ---")
    print(f"Output Directory: {EXPERIMENT_DIR.resolve()}\n")

    for config_name, noise_vals in NOISE_CONFIGS.items():
        print(f"=== Running Configuration: {config_name} ===")
        print(f"Noise U: {noise_vals['noise_u']} | Noise Z: {noise_vals['noise_z']}")
        
        # Run the training
        train(
            # Pass experiment constants
            **EXPERIMENT_CONSTANTS,
            
            # Pass noise specific to this run
            noise_u=noise_vals['noise_u'],
            noise_z=noise_vals['noise_z'],
            
            # Output settings
            save_repo=str(EXPERIMENT_DIR),
            net_params_filename=f"{config_name}_params",
            plot_filename=f"{config_name}_training_plot",
            plot_metrics=True
        )
        print(f"Completed {config_name}\n")
        print("-" * 50)

if __name__ == "__main__":
    run_all()