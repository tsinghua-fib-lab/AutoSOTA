import numpy as np
import jax
import jax.numpy as jnp
import neural_tangents as nt
import os

from neural_tangents import stax
from itertools import product
from tqdm import tqdm

from templates import dfs
from templates import utils

from initialization.initialize_dfs import verify_dfs_equivalence, GraphInitializer, generate_random_connected_graph, two_hop_coloring

import json  
from pathlib import Path

eps = 1e-10

jax.config.update("jax_enable_x64", True)


if __name__ == "__main__":

    print("Validating DFS")

    trials = 100
    results = []  # List to store results for each trial

    # Create main directory for all outputs
    main_directory = "dfs_results"
    os.makedirs(main_directory, exist_ok=True)

    for trial in tqdm(range(trials), desc="Overall Trials"):
        print(f"\n=== Starting Trial {trial + 1}/{trials} ===")

        # Generate graph and parameters
        n, E, D, d_G, A = generate_random_connected_graph(max_D=3, max_nodes=7)

        colors = two_hop_coloring(A, D)
        l = (2*E).bit_length() + 1

        print(f"Generated graph: n={n}, D={D}, d_G={d_G}, l={l}")

        template = dfs.get_sample(l,D)[1]
        target_key = 'u_stamp_slot'
        message_idx = utils.get_key_start_index(target_key, template)
        X, Y_train = dfs.get_dataset(l,D)

        X_train = jnp.eye(len(X))
        Y_train = jnp.array(Y_train, dtype=jnp.float64)
        n0 = len(X)
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(1024),   # First dense layer with 128 units
            stax.Relu(),       # ReLU activation
            stax.Dense(n0)      # Output dense layer with 1 unit
        )
        predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train, Y_train)

        # Initialize the graph
        initializer = GraphInitializer(A, colors, D, l)
        X_test_list = initializer.initialize_all_nodes()

        # Calculate iterations
        iterations= (2*E)*(5*l+D**2+3*D+12)

        print(f"Total iterations to simulate: {iterations}")

        # Create trial-specific directory
        trial_directory = os.path.join(main_directory, f"trial_{trial}")
        os.makedirs(trial_directory, exist_ok=True)

        # Simulation log file
        simulation_log_path = os.path.join(trial_directory, f"dfs_simulation_log_{trial}.txt")

        # Open the file to write the output
        with open(simulation_log_path, 'w') as f:
            # Log graph parameters at the beginning of the file
            f.write("GRAPH PARAMETERS:\n")
            f.write(f"n (nodes): {n}\n")
            f.write(f"D (max degree): {D}\n")
            f.write(f"d_G (diameter): {d_G}\n")
            f.write(f"l (bit length): {l}\n")
            f.write(f"Adjacency Matrix:\n{A}\n\n")
            
            # --- Log the initial state ---
            f.write("INITIAL STATE ====================================================\n")
            
            if not X_test_list:
                f.write("X_test_list is empty. No initial state to log.\n")
            else:
                ordered_keys = list(X_test_list[0].keys())
            
                for e, row in enumerate(X_test_list):
                    node_state_parts = [f"'{k}': {repr(row[k])}" for k in ordered_keys]
                    node_state_str = ' '.join(node_state_parts)
                    f.write(f"node{e}=>      {node_state_str}\n")
            
            # --- Main iteration loop with progress bar ---
            progress_bar = tqdm(total=iterations, desc=f"Trial {trial} Iterations", leave=False)
            
            for i in range(iterations):
                f.write(f"iteration #{i+1}# ====================================================\n")                
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({'current_iter': i+1})
                
                # --- computational logic ---
                X_test_encoded_l = []
                for x_test in X_test_list:
                    _, x_enc = utils.encode_data(x_test, X)
                    X_test_encoded_l.append(x_enc)

                X_test_list = [] # Clear the list to hold the new states for the next iteration

                X_test = jnp.array(X_test_encoded_l, dtype=jnp.float64)
                Y_pred = predict_fn(x_test=X_test, get='ntk', compute_cov=True)
                
                Y_pred_round = np.where(Y_pred.mean > eps, 1.0, 0.0).astype(int)

                n_nodes = Y_pred_round.shape[0]
                n_features = Y_pred_round.shape[1]

                P_c = np.zeros((n_features, n_features))
                P_c[:message_idx, :message_idx] = np.eye(message_idx)

                P_m = np.zeros((n_features, n_features))
                P_m[message_idx:, message_idx:] = np.eye(n_features - message_idx)

                X_test = (Y_pred_round @ P_c + A @ Y_pred_round @ P_m).astype(int)
                # --- End of computational logic ---

                # Log the updated state for the current iteration
                for e, row_vector in enumerate(X_test):
                    # Unflatten the data back into a dictionary
                    x_test_unflattened = utils.unflatten_sample(row_vector, template)
                    
                    # Build the string representation for the node's state
                    node_state_parts = [f"'{k}': {repr(x_test_unflattened[k])}" for k in ordered_keys]
                    node_state_str = ' '.join(node_state_parts)
                    
                    # Write the formatted line to the file
                    f.write(f"node{e}=>      {node_state_str}\n")
                    
                    # Append the unflattened state for the next iteration
                    X_test_list.append(x_test_unflattened)
            
            progress_bar.close()

        print(f"✅ Simulation log successfully saved to '{simulation_log_path}'")

        # Verify the result
        verified = verify_dfs_equivalence(A, X_test_list)
        verification_result = "PASS" if verified else "FAIL"
        
        print(f"Verification result for trial {trial}: {verification_result}")

        # Store result
        trial_result = {
            'trial': trial,
            'A': A.tolist(),  # Convert numpy array to list for JSON serialization
            'verified': verified,
            'verification_result': verification_result,
            'n': n,
            'D': D,
            'd_G': d_G,
            'l': l,
            'iterations': iterations,
            'colors': colors.tolist() if hasattr(colors, 'tolist') else colors
        }
        results.append(trial_result)

        # Save individual trial result
        result_filename = os.path.join(trial_directory, f"verification_result_{trial}.json")
        with open(result_filename, 'w') as f:
            json.dump(trial_result, f, indent=2)
        print(f"Trial result saved to {result_filename}")

    # Save summary of all results
    summary_filename = os.path.join(main_directory, "summary_results.json")
    with open(summary_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also create a human-readable summary
    readable_summary = {
        'total_trials': len(results),
        'passed_trials': sum(1 for r in results if r['verified']),
        'failed_trials': sum(1 for r in results if not r['verified']),
        'success_rate': sum(1 for r in results if r['verified']) / len(results) * 100 if results else 0,
        'detailed_results': results
    }
    
    readable_summary_filename = os.path.join(main_directory, "readable_summary.json")
    with open(readable_summary_filename, 'w') as f:
        json.dump(readable_summary, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"Total trials: {len(results)}")
    print(f"Passed: {readable_summary['passed_trials']}")
    print(f"Failed: {readable_summary['failed_trials']}")
    print(f"Success rate: {readable_summary['success_rate']:.2f}%")
    print(f"Detailed results saved to: {summary_filename}")
    print(f"Human-readable summary saved to: {readable_summary_filename}")

    # Print failed trials for quick review
    failed_trials = [r for r in results if not r['verified']]
    if failed_trials:
        print(f"\n⚠️  FAILED TRIALS: {[r['trial'] for r in failed_trials]}")
    else:
        print(f"\n✅ ALL TRIALS PASSED!")


