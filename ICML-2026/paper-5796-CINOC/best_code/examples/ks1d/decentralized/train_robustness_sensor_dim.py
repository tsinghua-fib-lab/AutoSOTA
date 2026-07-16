# We're not using this script actually.
"""
Sensor Dimension Experiment - Multi-Domain Runner (Smart Skip)
Trains Decentralized ControlNet across multiple physical configurations (L, N).

Updates:
- Added `should_skip` logic to avoid retraining existing models.
- Uses corrected "Resonance Bracketing" sensor lists.
- Uses corrected Noise levels (0.1/0.05).
"""
from pathlib import Path
from train_utils import train

# --- Experiment Configuration ---

# Base Output Directory
BASE_EXPERIMENT_DIR = Path("figures/sensor_dim_ablation")
BASE_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Define the Physics Configurations
DOMAIN_CONFIGS = [
    {
        # Target Wavelength: ~52 pixels
        "name": "L22_N128_Original",
        "L_domain": 22.0,
        "N_grid": 128,
        "n_agents": 30,
        # Bracket: [0.5x, 0.6x, 0.8x, 1.0x, 1.2x, 1.5x]
        "sensor_dims": [25, 30, 40, 52, 65, 80] 
    },
    # {
    #     # Target Wavelength: ~36 pixels
    #     "name": "L64_N256_HighRes",
    #     "L_domain": 64.0,
    #     "N_grid": 256,
    #     "n_agents": 88,  # Scaled density (approx 1.36)
    #     # Bracket: [0.5x, 0.7x, 0.8x, 1.0x, 1.2x, 2.0x]
    #     "sensor_dims": [18, 25, 30, 36, 45, 72]
    # },
    # {
    #     # Target Wavelength: ~19 pixels
    #     "name": "L124_N256_Coarse",
    #     "L_domain": 124.0, # Corrected L
    #     "N_grid": 256,     # Corrected N
    #     "n_agents": 170,   # Scaled density (approx 1.36)
    #     # Bracket: [0.5x, 1.0x, 1.2x, 1.5x, 2.0x]
    #     "sensor_dims": [10, 19, 24, 30, 40]
    # }
]

# Fixed Training Constants 
COMMON_TRAIN_ARGS = {
    "n_pde": 128,
    "batch_size": 32,
    "T_steps": 300,
    "epochs": 500,
    "noise_u": 0.1,     # Corrected: Robustness Noise
    "noise_z": 0.05    
}

def run_multi_domain_experiments():
    print(f"Starting Targeted Multi-Domain Experiment (Smart Resume)...")
    print(f"Base Output Directory: {BASE_EXPERIMENT_DIR.resolve()}\n")

    # --- Outer Loop: Physical Domains ---
    for config in DOMAIN_CONFIGS:
        config_name = config["name"]
        L = config["L_domain"]
        N = config["N_grid"]
        n_agents = config["n_agents"]
        sensors = config["sensor_dims"]
        
        # Create directory
        current_save_dir = BASE_EXPERIMENT_DIR / config_name
        current_save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"==========================================================")
        print(f" CONFIG: {config_name}")
        print(f" Physics: L={L}, N={N} | Density: {n_agents} Agents")
        print(f" Targeted Sensors: {sensors}")
        print(f"==========================================================\n")

        # --- Inner Loop: Sensor Sizes ---
        for patch_size in sensors:
            run_name = f"sensor_{patch_size}"
            
            # --- SKIP LOGIC ---
            # We check if the final parameter file already exists.
            # The train function saves params as f"{net_params_filename}.msgpack"
            expected_param_file = current_save_dir / f"{run_name}_params.msgpack"
            
            if expected_param_file.exists():
                print(f"   > [SKIPPING] {run_name} (File exists: {expected_param_file.name})")
                continue
            
            # --- RUN TRAINING ---
            print(f"   > Training Sensor Size: {patch_size}x{patch_size}...")
            
            train(
                # 1. Physics & Agent Config
                L_domain=L,
                N_grid=N,
                n_agents=n_agents,
                
                # 2. Variable: Sensor Range
                sensor_range=patch_size,
                
                # 3. Output settings
                save_repo=str(current_save_dir),
                net_params_filename=f"{run_name}_params",
                plot_filename=f"{run_name}_training",
                plot_metrics=True,
                
                # 4. Common Fixed Args
                **COMMON_TRAIN_ARGS
            )
            print(f"     [Done] {run_name}\n")
            
        print(f"--- Completed {config_name} ---\n")

if __name__ == "__main__":
    run_multi_domain_experiments()