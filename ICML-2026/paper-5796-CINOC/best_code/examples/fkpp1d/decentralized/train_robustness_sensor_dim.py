"""
Sensor Dimension Experiment - Runner
Trains 5 variants of the Decentralized ControlNet with varying patch sizes:
[10, 12, 14, 16, 18]

Uses the fixed 'Low Noise' configuration for all runs.
"""
from pathlib import Path
from train_utils import train

# --- Experiment Configuration ---

# Experiment Output Directory
EXPERIMENT_DIR = Path("figures/sensor_dim")

# Ensure the directory exists
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

# Standard Experiment Constants
EXPERIMENT_CONSTANTS = {
    "n_pde": 100,
    "n_agents": 30,
    "batch_size": 32,
    "T_steps": 300,
    "epochs": 500,
    "R_safe": 0.05
}

# Fixed Noise Setup (Low Noise)
# We use this for all sensor dimension tests to isolate the variable.
FIXED_NOISE_CONFIG = {
    "noise_u": 0.02, 
    "noise_z": 0.01
}

# Variable: Sensor Dimensions (Patch Sizes)
SENSOR_DIMS = [0.04, 0.06, 0.08, 0.12, 0.2, 0.3, 0.5, 1.0]

def run_sensor_experiments():
    print(f"Starting Sensor Dimension Sensitivity Experiment...")
    print(f"Saving results to: {EXPERIMENT_DIR.resolve()}\n")

    for patch_size in SENSOR_DIMS:
        print(f"=== Running Training for Patch Size: {patch_size}x{patch_size} ===")
        
        # Define filenames based on the variable
        run_name = f"sensor_dim_{patch_size}"
        
        train(
            # 1. Pass standard constants
            **EXPERIMENT_CONSTANTS,
            
            # 2. Pass fixed noise settings
            **FIXED_NOISE_CONFIG,
            
            # 3. Pass Model Configuration
            sensor_range=patch_size,
            
            # 4. Output settings
            save_repo=str(EXPERIMENT_DIR),
            net_params_filename=f"{run_name}_params",
            plot_filename=f"{run_name}_training_plot",
            plot_metrics=True
        )
        print(f"Completed {run_name}\n")

if __name__ == "__main__":
    run_sensor_experiments()