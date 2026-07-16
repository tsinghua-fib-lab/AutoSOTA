"""
Robustness Transfer Experiment - Runner
Decoupled Ablation Study: Isolates the effects of Actuator Noise (u) vs State Noise (z).
"""
from pathlib import Path
from train_utils import train

# Experiment Output Directory
EXPERIMENT_DIR = Path("figures/noise_experiments/decoupled_robustness")

# --- Experiment Constants ---
EXPERIMENT_CONSTANTS = {
    "n_pde": 100,
    "n_agents": 30,
    "batch_size": 32,
    "T_steps": 300,
    "epochs": 500,
    "R_safe": 0.05
}

# --- Define Noise Levels ---
# We define the specific magnitudes we want to test for each channel
ACTUATOR_LEVELS = [0.02, 0.1, 0.5] # Low, Med, High for u
STATE_LEVELS    = [0.01, 0.05, 0.25] # Low, Med, High for z

# --- Generate Decoupled Configurations ---
NOISE_CONFIGS = {}

# 1. Baseline (Clean)
NOISE_CONFIGS["baseline_clean"] = {"noise_u": 0.0, "noise_z": 0.0}

# 2. Actuator Noise Study (Fix z=0, Vary u)
for level in ACTUATOR_LEVELS:
    name = f"actuator_only_{str(level).replace('.', 'p')}"
    NOISE_CONFIGS[name] = {"noise_u": level, "noise_z": 0.0}

# 3. State Noise Study (Fix u=0, Vary z)
for level in STATE_LEVELS:
    name = f"state_only_{str(level).replace('.', 'p')}"
    NOISE_CONFIGS[name] = {"noise_u": 0.0, "noise_z": level}

# 4. Coupled Noise (u=z) 
# NOISE_CONFIGS["coupled_high"] = {"noise_u": 0.5, "noise_z": 0.25}


def run_all():
    print(f"Starting Decoupled Robustness Experiments...")
    print(f"Total configurations to run: {len(NOISE_CONFIGS)}")
    print(f"Saving results to: {EXPERIMENT_DIR.resolve()}\n")

    # Sort keys to run in a logical order (Clean -> Actuator -> State)
    for config_name in sorted(NOISE_CONFIGS.keys()):
        noise_vals = NOISE_CONFIGS[config_name]
        
        print(f"=== Running Configuration: {config_name} ===")
        print(f"   > Noise U: {noise_vals['noise_u']}")
        print(f"   > Noise Z: {noise_vals['noise_z']}")
        
        train(
            **EXPERIMENT_CONSTANTS,
            
            # Pass noise specific to this run
            noise_u=noise_vals['noise_u'],
            noise_z=noise_vals['noise_z'],
            
            # Output settings
            save_repo=str(EXPERIMENT_DIR),
            net_params_filename=f"{config_name}_params_0.001",
            plot_filename=f"{config_name}_training_plot_0.001",
            plot_metrics=True,
            
            lambda_u=0.001
        )
        print(f"Completed {config_name}\n")
        print("-" * 30)
        
if __name__ == "__main__":
    run_all()