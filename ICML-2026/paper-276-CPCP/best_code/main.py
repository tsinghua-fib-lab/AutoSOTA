import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from utils import seed_everything, DEVICE
from data_utils import *  
from methods import * 


# Suppress warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

# Data split
def rcp_protocol_split(X, Y, cal_size=0.2, seed=42):
    """Splits data into Training, Calibration, and Test sets."""
    # Convert cal_size fraction to int if needed, here simplified
    n_cal = int(len(X) * cal_size)    
    X_rem, X_cal, Y_rem, Y_cal = train_test_split(X, Y, test_size=n_cal, random_state=seed)
    X_tr, X_te, Y_tr, Y_te = train_test_split(X_rem, Y_rem, test_size=0.25, random_state=seed)
    return X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te


# Main function
def run_benchmark_suite():
    seed_everything(42)
    print(f"Using Device: {DEVICE}")

    # Dataset registry
    dataset_loaders = {    
        "bike": load_bike,
        "diamond": load_diamonds,            
        "gas_turbine":load_gas_turbine,        
        "naval": load_naval,    
        "SGEMM": load_sgemm_product,                               
        "superconduct": load_superconductivity,                
        "Transcoding": load_transcoding, 
        "WEC": load_wec,                 
    }
    
    alpha = 0.1
    n_seeds = 20
    
    # Method registry 
    methods = [
        ('Split', run_split),
        ('PLCP-Pin-G20', lambda *a: run_plcp(*a, n_groups=20, score_type='pinball')),
        ('PLCP-Pin-G50', lambda *a: run_plcp(*a, n_groups=50, score_type='pinball')),
        ('Gaussian-Scoring', run_gaussian_scoring),
        ('CQR-Pinball', lambda *a: run_cqr(*a, 'pinball')),
        ('CQR-ALD', lambda *a: run_cqr(*a, 'ald')),        
        ('RCP-Pinball', run_rcp),        
        ('RCP-ALD', lambda *a: run_rcp(*a, 'ald')),
        ('RCP-MultiHead', run_rcp_multi_head),                
        
        # Colorful Pinball Variants (CPCP)
        ('CPCP-Split-0.02', lambda *a, **k: run_rcp_density_improved(*a, epsilon=0.02, mode='vanilla', **k)),                
        ('CPCP-Clip-0.02', lambda *a, **k: run_rcp_density_improved(*a, epsilon=0.02, mode='clip', clip_max=5.0, **k)),            
        ('CPCP-Mix-0.02', lambda *a, **k: run_rcp_density_improved(*a, epsilon=0.02, mode='mix', mix_ratio=0.5, **k)),
        ('CPCP-Clip+Mix-0.02', lambda *a, **k: run_rcp_density_improved(*a, epsilon=0.02, mode='clip', clip_max=5.0, mix_ratio=0.5, **k)),
        ## Ablation study on delta
        # ('CPCP-Split-0.01', lambda *a, **k: run_rcp_density_improved(*a, epsilon=0.01, mode='vanilla', **k)),          
        # ('CPCP-Clip+Mix-0.01', lambda *a, **k: run_rcp_density_improved(*a, epsilon=0.01, mode='clip', clip_max=5.0, mix_ratio=0.5, **k)),
        # ('CPCP-Split-0.05', lambda *a, **k: run_rcp_density_improved(*a, epsilon=0.05, mode='vanilla', **k)),  
        # ('CPCP-Clip+Mix-0.05', lambda *a, **k: run_rcp_density_improved(*a, epsilon=0.05, mode='clip', clip_max=5.0, mix_ratio=0.5, **k)),
    ]
    
    for ds_name, loader in dataset_loaders.items():
        print(f"\n>>>>>> Running {ds_name} <<<<<<")
        try: 
            X, Y = loader()
            print(f"Data Shape: {X.shape}, {Y.shape}")
        except Exception as e: 
            print(f"Error loading {ds_name}: {e}")
            continue
            
        results = {m[0]: [] for m in methods}
        
        total_start_time = time.time()

        for seed in range(n_seeds):
            seed_start_time = time.time()
            print(f"Seed {seed}...", end="", flush=True)
            X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te = rcp_protocol_split(X, Y, seed=42+seed)
            
            for name, func in methods:
                try:
                    # Pass extra args for CPCP methods
                    if 'CPCP' in name or 'RCP-Density' in name:
                        res = func(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha, dataset_name=ds_name, seed=seed)
                    else:
                        res = func(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha)
                    results[name].append(res)
                except Exception as e:
                    print(f" Err({name}:{e})", end="")

            seed_duration = time.time() - seed_start_time
            total_elapsed = time.time() - total_start_time
            # print(" Done")
            print(f" Done (Seed Time: {seed_duration/60:.2f}m | Total Time: {total_elapsed/60:.2f}m)")

        # Summary & Save
        summary_rows = []
        for name, mets in results.items():
            if not mets: continue
            row = {'Method': name}
            for k in ['Cov', 'Size', 'WSC', 'MSCE_10', 'MSCE_30', "L1-ERT", "L2-ERT"]:
                vals = [m[k] for m in mets]
                row[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
            summary_rows.append(row)
        
        print("\nSummary:")
        df_res = pd.DataFrame(summary_rows)
        print(df_res)
        
        if not os.path.exists("./results"): os.makedirs("./results")
        df_res.to_csv(f"./results/{ds_name}_results.csv")

if __name__ == "__main__":
    run_benchmark_suite()