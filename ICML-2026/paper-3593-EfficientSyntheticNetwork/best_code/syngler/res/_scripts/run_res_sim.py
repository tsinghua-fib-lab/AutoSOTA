import numpy as np
import pickle
import os
import sys
from syngler.utils.source import bootstrap_alpha_Z
from tqdm import tqdm

N_RANGE = [500, 1000, 1500]
R_RANGE = [2, 3, 4]
sparse_level = 0.0
SEED_RANGE = range(0, 200)
REP_RANGE = range(0, 200)

def main():
    for n in N_RANGE:
        for r in R_RANGE:
            input_base_path = f'../../datasets/simulation/run/n={n}_r={r}_sparse={sparse_level}/'
            output_base_path = f'../../synthetic/simulation/Res-sample/n={n}_r={r}_sparse={sparse_level}/'

            for seed in SEED_RANGE:
                input_file_path = os.path.join(input_base_path, f'seed={seed}.pkl')
                seed_output_path = os.path.join(output_base_path, f'seed={seed}')
                os.makedirs(seed_output_path, exist_ok=True)

                try:
                    with open(input_file_path, 'rb') as f:
                        result = pickle.load(f)
                
                    model_alpha = np.array(result["model_alpha"]).reshape(-1, 1)
                    model_Z = np.array(result["model_Z"])

                    print(f"Processing seed={seed}, with data shape alpha={model_alpha.shape}, Z={model_Z.shape}")
                
                    for rep in REP_RANGE:
                        np.random.seed(seed+rep) 
                        alpha_bootstrap, Z_bootstrap = bootstrap_alpha_Z(model_alpha, model_Z, batch=1)
                        Z_processed = Z_bootstrap.squeeze()
                        alpha_processed = alpha_bootstrap.squeeze()
                        output_file_path = os.path.join(seed_output_path, f'rep{rep}.npz')
                        np.savez(output_file_path, alpha=alpha_processed, Z=Z_processed)
                    
                        if (rep + 1) % 10 == 0 or rep == REP_RANGE[-1]:
                            print(f"Saved rep={rep} to {output_file_path}")

                except FileNotFoundError:
                    print(f"Warning: File not found at {input_file_path}. Skipping this seed.")
                except KeyError:
                    print(f"Error: The pkl file for seed={seed} does not contain the expected keys 'model_Z' or 'model_alpha'. Skipping.")
                except Exception as e:
                    print(f"An unexpected error occurred while processing seed={seed}: {e}. Skipping.")

    print("\nBootstrap process completed.")


if __name__ == "__main__":
    main()
