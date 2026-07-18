import numpy as np
import pickle
import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm


from syngler.utils.source import bootstrap_alpha_Z

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["dblp", "youtube", "yelp", "polblogs"])
    args = parser.parse_args()

    R_RANGE = [2, 3, 4, 5, 6]
    REP_RANGE = range(0, 2)

    for r in R_RANGE:
        input_base_path = f'../../datasets/{args.dataset}/run/r={r}/'
        output_base_path = f'../../synthetic/{args.dataset}/Res-sample/r={r}'

        seed = 0
        input_file_path = os.path.join(input_base_path, f'seed=0.pkl')
        seed_output_path = os.path.join(output_base_path, f'seed=0')
        os.makedirs(seed_output_path, exist_ok=True)

        try:
            with open(input_file_path, 'rb') as f:
                result = pickle.load(f)
            
            model_alpha = np.array(result["model_alpha"]).reshape(-1, 1)
            model_Z = np.array(result["model_Z"])

            print(f"Processing dataset={args.dataset}, seed=0, alpha={model_alpha.shape}, Z={model_Z.shape}")
            
            for rep in REP_RANGE:
                np.random.seed(seed + rep + 10000) 
                alpha_bootstrap, Z_bootstrap = bootstrap_alpha_Z(model_alpha, model_Z, batch=1)
                Z_processed = Z_bootstrap.squeeze()
                alpha_processed = alpha_bootstrap.squeeze()
                output_file_path = os.path.join(seed_output_path, f'rep{rep}.npz')
                np.savez(output_file_path, alpha=alpha_processed, Z=Z_processed)
                
                if (rep + 1) % 10 == 0 or rep == REP_RANGE[-1]:
                    print(f"  - Saved rep={rep} to {output_file_path}")

        except FileNotFoundError:
            print(f"Warning: File not found at {input_file_path}. Skipping.")
        except KeyError:
            print(f"Error: The pkl file for seed=0 does not contain 'model_Z' or 'model_alpha'. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing seed=0: {e}. Skipping.")

    print("\nBootstrap process completed.")

if __name__ == "__main__":
    main()
