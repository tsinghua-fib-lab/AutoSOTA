import math
import argparse
import os
import torch
import dataset
from model import NTKMLP
from torch.utils.data import DataLoader
from dataset import FloodingDataset, load_test_cases, load_test_samples
import numpy as np
import sys
import numpy as np

def to_padded_numpy(A, fill_value=0, dtype=np.int64):
    """
    Converts a ragged 3D list into a uniform 3D NumPy array by padding.
    """
    max_len = max(len(sublist) for sublist in A) if A else 0
    try:
        inner_len = next(len(sublist[0]) for sublist in A if sublist and sublist[0])
    except StopIteration:
        inner_len = 0
    padded_array = np.full((len(A), max_len, inner_len), fill_value, dtype=dtype)
    for i, sublist in enumerate(A):
        if sublist:
            padded_array[i, :len(sublist), :] = sublist
    return padded_array


def evaluate_cases(cases, wrong_pred_inp, dtype=np.int64):
    """
    Safe bit packing with overflow checking
    """
    A = cases
    B = wrong_pred_inp.astype(dtype)
    n, m, k = A.shape
    powers = 1 << np.arange(k - 1, -1, -1)
    A_int = np.dot(A.reshape(-1, k), powers).reshape(n, m).astype(dtype)
    B_int = np.dot(B, powers).astype(dtype)
    b_set = set(B_int)
    result = np.zeros(n, dtype=bool)
    for i in range(n):
        for val in A_int[i]:
            if val in b_set:
                result[i] = True
                break
    return np.mean(~result)


def train_model(train_loader: DataLoader, epochs: int, k_p: int, hidden_dim: int, k: int, sigma_w: float = None, sigma_b: float = None, device: str = 'cpu', lr: float = 1e-3):
    """
    Train a model on the given dataset.
    """
    sigma_w = sigma_w if sigma_w is not None else 1
    sigma_b = sigma_b if sigma_b is not None else 0.0
    model = NTKMLP(k_p, hidden_dim, k, sigma_w, sigma_b).to(device).double()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.functional.mse_loss
    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.to(device).double()
            y = y.to(device).double()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
    return model

def evaluate_samples(model, inputs, batch_size : int =256 , device:str='cuda'):
    """
    Evaluate the model on a given set of samples.
    """
    pred = []
    num_samples = len(inputs)
    num_batches = (num_samples + batch_size - 1) // batch_size
    row_sums = inputs.sum(dim=1, keepdim=True)
    inputs = inputs / torch.sqrt(row_sums + (row_sums == 0).float())
    model.eval()
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_inputs = inputs[start_idx:end_idx].to(device).double()
            batch_pred = model(batch_inputs)
            pred.append(batch_pred)
        pred = torch.concatenate(pred, axis=0)
    return pred


def train_and_evaluate(n: int, D: int, l: int, delta: float, outdir: str, trials: int = 1, epochs: int = 2000, batch_size: int = 256, hidden_dim: int = 2048, sigma_w: float = None, sigma_b: float = None, max_models: int = 5_000, device: str = 'cpu', seed: int = 0, test_sample_path: str=None, test_case_path: str=None):
    """
    Train an ensemble of models and evaluate, with resumability.
    """
    d = D + 1
    k_p = l + 4 * d * l
    k = l + 4 * d * l

    # --- Load data once for all trials ---
    train_data = FloodingDataset(l, D, mode='train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_inputs, test_targets = load_test_samples(json_path=test_sample_path)
    test_targets = test_targets.to(device)
    test_cases = load_test_cases(json_path=test_case_path)
    test_cases = to_padded_numpy(test_cases)
    
    # --- Main Trials Loop ---
    for t in range(trials):
        # --- Setup deterministic paths for the current trial for resumability ---
        trial_dir = os.path.join(outdir, f"n{n}_D{D}_l{l}", f"trial_{t}")
        os.makedirs(trial_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(trial_dir, "checkpoint.pt")
        results_path = os.path.join(trial_dir, "results.csv")
        predictions_path = os.path.join(trial_dir, "predictions.pt")

        # --- Initialize variables ---
        all_preds = None
        num_models = 1
        case_accuracy = 0.0
        error = 1e6
        eps = 1e-4

        # --- Resume logic: Check for an existing checkpoint ---
        if os.path.exists(checkpoint_path):
            print(f"--- Loading checkpoint for Trial {t+1} ---")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if checkpoint.get('completed', False):
                print(f"✅ Trial {t+1} is already completed. Skipping.")
                continue

            print(f"Resuming Trial {t+1}...")
            all_preds = checkpoint['all_preds'].to(device)
            # Load the number of the *next* model to be trained
            num_models = checkpoint['num_models']

            # Recalculate metrics from loaded predictions to correctly set the while-loop condition
            mean = torch.mean(all_preds, dim=0)
            error = torch.nn.functional.mse_loss(mean, test_targets)
            error = error if not torch.isnan(error) else 1e6
            
            mean[mean > 0 + eps] = 1.
            mean[mean <= 0 + eps] = 0.

            correct_pred_idx = (mean.int() == test_targets.int()).all(dim=1).cpu()
            wrong_pred_idx = ~correct_pred_idx
            sample_accuracy = torch.mean(correct_pred_idx.float())
            wong_pred_inp = test_inputs[wrong_pred_idx].cpu().numpy()
            case_accuracy = evaluate_cases(test_cases, wong_pred_inp)
            
            print(f"Resumed state: {num_models-1} models trained. Case Acc: {case_accuracy:.3f}, Error: {error:.3f}")
        else:
            print(f"--- Starting Trial {t+1} from scratch ---")
            # Create a fresh results file with a header
            with open(results_path, "w") as f:
                f.write("num_models,error,sample_accuracy,case_accuracy\n")

        # --- Training Loop for the current trial ---
        while case_accuracy < 1 - delta and num_models <= max_models:
            print(f"\nTrial {t+1}, Training model {num_models}...")
            model = train_model(train_loader, epochs, k_p, hidden_dim, k, sigma_w, sigma_b, device)
            
            # Evaluate the newly trained model
            pred = evaluate_samples(model, test_inputs, batch_size=batch_size, device=device).unsqueeze(0)

            # Aggregate predictions with previous ones
            all_preds = pred if all_preds is None else torch.cat((all_preds, pred), dim=0)

            # --- Evaluate the updated ensemble ---
            mean = torch.mean(all_preds, dim=0)
            error = torch.nn.functional.mse_loss(mean, test_targets)
            error = error if not torch.isnan(error) else 1e6

            mean_thresholded = mean.clone()
            mean_thresholded[mean_thresholded > 0 + eps] = 1.
            mean_thresholded[mean_thresholded <= 0 + eps] = 0.

            correct_pred_idx = (mean_thresholded.int() == test_targets.int()).all(dim=1).cpu()
            wrong_pred_idx = ~correct_pred_idx
            sample_accuracy = torch.mean(correct_pred_idx.float())
            wong_pred_inp = test_inputs[wrong_pred_idx].cpu().numpy()
            case_accuracy = evaluate_cases(test_cases, wong_pred_inp)

            print(f"-> Trial {t+1}, Ensemble size: {num_models} | Error: {error:.3f} | Sample Acc: {sample_accuracy:.3f} | Case Acc: {case_accuracy:.3f}")
            
            # --- Save results and checkpoint for this step ---
            torch.save(all_preds.to("cpu"), predictions_path)
            
            with open(results_path, "a") as f:
                f.write(f"{num_models},{error.item()},{sample_accuracy.item()},{case_accuracy}\n")

            # Save checkpoint for resumability. Store the *next* model's number.
            checkpoint_data = {
                'num_models': num_models + 1,
                'all_preds': all_preds.to("cpu"),
                'completed': False
            }
            torch.save(checkpoint_data, checkpoint_path)
            
            num_models += 1

        # --- Finalize Trial ---
        print(f"--- ✅ Trial {t+1} finished with {num_models-1} models. Final Case Acc: {case_accuracy:.3f} ---")
        
        # Update checkpoint to mark as completed
        final_checkpoint_data = {
            'num_models': num_models - 1, # Total models trained in this trial
            'all_preds': all_preds.to("cpu") if all_preds is not None else torch.Tensor([]),
            'completed': True
        }
        torch.save(final_checkpoint_data, checkpoint_path)

# --- Main execution block remains the same ---
argparse = argparse.ArgumentParser()
argparse.add_argument("--delta", type=float, default=0.001)
argparse.add_argument("--outdir", type=str, default="results")
argparse.add_argument("--trials", type=int, default=10)
argparse.add_argument("--epochs", type=int, default=7000)
argparse.add_argument("--batch_size", type=int, default=256)
argparse.add_argument("--hidden_dim", type=int, default=-1)
argparse.add_argument("--sigma_w", type=float, default=1)
argparse.add_argument("--sigma_b", type=float, default=None)
argparse.add_argument("--max_models", type=int, default=300)
argparse.add_argument("--device", type=str, default="cuda")
argparse.add_argument("--seed", type=int, default=2)
argparse.add_argument("--n", type=int, default=7)
argparse.add_argument("--l", type=int, default=1)
argparse.add_argument("--D", type=int, default=2)

if __name__ == "__main__":
    args = argparse.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Using CPU instead.")
        args.device = "cpu"

    k = args.l + 4 * (args.D + 1) * args.l
    args.hidden_dim = k * 2000 if args.hidden_dim == -1 else args.hidden_dim

    # --- Build test paths from n, D, l ---
    test_sample_path = f"test_samples/test_samples_l{args.l}_D{args.D}_n{args.n}.json"
    test_case_path   = f"test_samples/test_cases_l{args.l}_D{args.D}_n{args.n}.json"

    train_and_evaluate(
        args.n,
        args.D,
        args.l,
        args.delta,
        args.outdir,
        args.trials,
        args.epochs,
        args.batch_size,
        args.hidden_dim,
        args.sigma_w,
        args.sigma_b,
        args.max_models,
        args.device,
        args.seed,
        test_sample_path,
        test_case_path
    )
