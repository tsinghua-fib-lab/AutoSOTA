import argparse
import time
import os
import random
import pickle
from typing import NamedTuple

import jax
import numpy as np
import jax.numpy as jnp
import pandas as pd
import wandb

# Helper imports (Ensure these modules are in your python path)
from jaxcld.models.asr_model import ASRModel
from jaxcld.models.cvx_relu_mlp import CVX_ReLU_MLP
from jaxcld.optimizers.admm import admm

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

class RunResults(NamedTuple):
    """Immutable structure to hold run results"""
    global_max_test_peak: float
    global_best_params: dict
    global_delta_test_peak: float
    global_best_delta_params: dict
    model_path: str

# -----------------------------------------------------------------------------
# Core Logic (Training Functions)
# -----------------------------------------------------------------------------

def estimate_tflops(duration_seconds):
    """
    Estimate TFLOPs for an NVIDIA RTX 4090 @ 70% bf16 Tensor Core efficiency
    """
    gflops_per_sec = 231000  # 231 TFLOPs = 70% of 330 peak bf16 performance
    tflops_used = (gflops_per_sec * duration_seconds) / 1000
    return tflops_used

def run(model_name, dataset_path, cronos_params, adamW_params, opt_seed, data_seed, output_dir, languages):
    """
    Run the CRONOS training pipeline for CVX-DPO.
    
    Args:
        model_name: Name of the model/dataset
        dataset_path: Directory containing input data
        cronos_params: Parameters for CRONOS optimizer
        adamW_params: Parameters for AdamW optimizer
        opt_seed: Random seed for optimization
        data_seed: Random seed for data loading
        output_dir: Directory to save outputs
        languages: Comma-separated list of languages to train on
    
    Returns:
        RunResults: NamedTuple with results and paths
    """
    global_max_test_peak = 0
    global_best_params = {}  # params that lead to highest CRONOS test peak
    global_delta_test_peak = 0
    global_best_delta_params = {}

    # Load the training and test data
    print(f"Loading data from {dataset_path} for languages {languages}...")
    asr_model = ASRModel.from_pretrained(model_name, config={"languages": languages})

    def _load_split(split):
        # Some backends (e.g. MMS) return (A, y, n_classes); Whisper returns (A, y).
        out = asr_model.load_data(dataset_path, data_seed=data_seed, caller_script="defrun", dataset_split=split)
        A, y = out[0], out[1]
        return A, y

    Atr, ytr = _load_split("train")
    Atst, ytst = _load_split("valid")
    
    ##### CRONOS #####
    # Number of neurons in the convex network (mapped from 'rank' parameter if P_S not set)
    num_neurons = cronos_params.get('neuron')
    
    # Create the convex neural network model
    model = CVX_ReLU_MLP(
        Atr, ytr, len(languages), num_neurons, 
        cronos_params['beta'], cronos_params['rho'], 
        jax.random.PRNGKey(0)
    )
    model.init_model()
    model.Xtst = Atst
    model.ytst = ytst

    print('Training model with CRONOS...')
    
    # Start timing CRONOS training
    cronos_start_time = time.time()
    
    # Run twice to get compiled version and accurate timing
    metrics = {}
    for i in range(2):
        _, metrics = admm(model, cronos_params)
        if i == 1:
            cronos_end_time = time.time()
            cronos_training_time = cronos_end_time - cronos_start_time
            print('Finished training with CRONOS')
            print(f"CRONOS training time: {cronos_training_time:.2f} seconds")

    # Get peak accuracies
    train_peak = np.max(metrics['train_acc'])
    test_peak = np.max(metrics['val_acc'])
    print(f"Peak train accuracy: {train_peak}")
    print(f"Peak test accuracy: {test_peak}")

    # Update global best if this run is better
    if test_peak > global_max_test_peak:
        global_max_test_peak = test_peak
        print(f"New global max test peak for CXV: {global_max_test_peak}")
        global_best_params = {
            "model_name": model_name,
            "cronos_params": cronos_params,
            "adamW_params": adamW_params,
            "opt_seed": opt_seed,
            "data_seed": data_seed,
            "test_peak": test_peak,
            "train_peak": train_peak
        }

    print("\n" + "="*50)
    print("TRAINING SUMMARY:")
    print(f"CRONOS training time: {cronos_training_time:.2f} seconds")
    print("="*50 + "\n")

    # Prepare output directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save global metrics CSV
    metrics_df = pd.DataFrame({
        "global_max_test_peak": [global_max_test_peak],
        "global_best_params": [global_best_params],
        "global_delta_test_peak": [global_delta_test_peak],
        "global_best_delta_params": [global_best_delta_params]
    })

    csv_path = os.path.join(model_dir, "global_metrics.csv")
    print(f"Saving CSV to: {csv_path}")
    metrics_df.to_csv(csv_path, sep='\t', encoding='utf-8', index=False, header=True)

    # Save the trained convex model
    model_name = model_name.replace("/", "_")
    trained_model_path = os.path.join(model_dir, f"{model_name}_trained_cvx_mlp.pkl")
    with open(trained_model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Trained convex MLP model saved at: {trained_model_path}")

    return RunResults(
        global_max_test_peak=global_max_test_peak,
        global_best_params=global_best_params,
        global_delta_test_peak=global_delta_test_peak,
        global_best_delta_params=global_best_delta_params,
        model_path=trained_model_path
    )

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRONOS Training Pipeline")
    
    # Required Arguments
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model/experiment")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to data directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to output directory")
    
    # Optional Arguments (General)
    parser.add_argument('--languages', type=str, required=True, help="Comma-separated list of languages to train on")
    parser.add_argument('--opt_seed', type=int, default=1024, help="Optimization seed")
    parser.add_argument('--data_seed', type=int, default=None, help="Data seed (default: random 1-10)")

    # CRONOS Hyperparameters
    parser.add_argument('--rank', type=int, default=20, help="Rank for CRONOS (default: 20)")
    parser.add_argument('--neuron', type=int, default=64, help="Number of neurons for CRONOS (default: 64)")
    parser.add_argument('--beta', type=float, default=0.001, help="Beta parameter (default: 0.001)")
    parser.add_argument('--rho', type=float, default=0.1, help="Rho parameter (default: 0.1)")
    parser.add_argument('--gamma_ratio', type=float, default=1, help="Gamma ratio (default: 1)")
    parser.add_argument('--admm_iters', type=int, default=6, help="ADMM iterations (default: 6)")
    parser.add_argument('--pcg_iters', type=int, default=32, help="PCG iterations (default: 32)")
    
    # AdamW Hyperparameters
    parser.add_argument('--adam_lr', type=float, default=1e-4, help="AdamW learning rate/gamma (default: 10^-4)")
    parser.add_argument('--adam_epochs', type=int, default=30, help="AdamW epochs (default: 30)")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size (default: 1024)")

    args = parser.parse_args()

    # Handle random data seed if not provided
    if args.data_seed is None:
        args.data_seed = random.randint(1, 10)

    # Construct parameter dictionaries from arguments
    cronos_params = dict(
        rank=args.rank,
        neuron=args.neuron,
        beta=args.beta,
        rho=args.rho,
        gamma_ratio=args.gamma_ratio,
        admm_iters=args.admm_iters,
        pcg_iters=args.pcg_iters,
        check_opt=False
    )

    adamW_params = dict(
        optimizer='AdamW', 
        gamma=args.adam_lr, 
        n_epoch=args.adam_epochs, 
        batch_size=args.batch_size
    )

    # Initialize wandb
    wandb.init(
        project="CLD",
        name=f"cronos_{args.model_name}",
        config={
            "model_name": args.model_name,
            "cronos_params": cronos_params,
            "adamW_params": adamW_params,
            "opt_seed": args.opt_seed,
            "data_seed": args.data_seed,
            "output_dir": args.output_dir,
            # Flattened config for easier filtering
            "rank": args.rank,
            "neuron": args.neuron,
            "beta": args.beta,
            "rho": args.rho,
            "admm_iters": args.admm_iters,
            "pcg_iters": args.pcg_iters,
            "learning_rate": args.adam_lr
        }
    )

    print(f"Starting run for model: {args.model_name}")
    start_time = time.time()

    # Run model training and evaluation
    results: RunResults = run(
        model_name=args.model_name, 
        dataset_path=args.dataset_path, 
        cronos_params=cronos_params, 
        adamW_params=adamW_params, 
        opt_seed=args.opt_seed, 
        data_seed=args.data_seed, 
        output_dir=args.output_dir, 
        languages=args.languages.split(",")
    )

    elapsed_time = time.time() - start_time
    estimated_tflops = estimate_tflops(elapsed_time)

    # Logging results to WandB
    wandb.log({
        "global_max_test_peak": results.global_max_test_peak,
        "global_delta_test_peak": results.global_delta_test_peak,
        "training_time": elapsed_time,
        "estimated_tflops": estimated_tflops,
        "model_path": results.model_path
    })

    # Log global best parameters
    for key, value in results.global_best_params.items():
        wandb.log({f"global_best_params_{key}": value})

    # Log global best delta parameters
    for key, value in results.global_best_delta_params.items():
        wandb.log({f"global_best_delta_params_{key}": value})

    # Summary data for local logging
    data = {
        "global_max_test_peak": [results.global_max_test_peak],
        "global_best_params": [results.global_best_params],
        "global_delta_test_peak": [results.global_delta_test_peak],
        "global_best_delta_params": [results.global_best_delta_params],
        "model_path": [results.model_path],
        "training_time": [elapsed_time],
        "estimated_tflops": [estimated_tflops]
    }

    df = pd.DataFrame(data)
    print("\nFinal Results:")
    print(df)
    print(f"Stage 1 Training completed in {elapsed_time:.2f} seconds")
    print(f"Estimated TFLOPS: {estimated_tflops:.2f}")

    # Save summary text file
    log_out_path = os.path.join(args.output_dir, args.model_name, "cronos_results.txt")
    # Ensure directory exists (it should be created inside run(), but just in case)
    os.makedirs(os.path.dirname(log_out_path), exist_ok=True)
    
    with open(log_out_path, "w") as f:
        for k, v in data.items():
            f.write(f"{k}: {v[0]}\n")

    wandb.finish()