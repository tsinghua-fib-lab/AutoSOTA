import argparse
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import time
import random
import os
import csv
from numerical_data_generator import generate_data
from evaluation import compute_mcc_g as compute_mcc

# -----------------------------
# set random seed
# -----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)






# -----------------------------
# Loss Function
# -----------------------------
def loss_function_torch(A_hat, D_flat, sigma_vec, B, Cov_list, T, n):
    """
    Compute loss with diagonal sigma (different values per variable).
    """
    device = A_hat.device
    I = torch.eye(n, device=device)
    I_B_inv = torch.inverse(I - B)

    # build Sigma from learnable diagonal
    Sigma = torch.diag(sigma_vec)
    M = I_B_inv @ (Sigma @ Sigma) @ I_B_inv.T

    Cov_tensor = torch.stack([torch.tensor(c, dtype=torch.float32, device=device) for c in Cov_list])
    D_tensor = D_flat.view(T, n)

    dM_batch = D_tensor[:, :, None] * M[None, :, :] * D_tensor[:, None, :]
    ADAT_batch = A_hat @ dM_batch @ A_hat.T

    diff = Cov_tensor - ADAT_batch
    loss = torch.sum(diff ** 2)
    return loss


# -----------------------------
# PyTorch Optimization
# -----------------------------
def run_torch_optimization(Cov_list, X_list, Z_list, T, n, xn, num_steps=2000,
                           lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"


    A_hat = nn.Parameter(torch.rand(xn, n, dtype=torch.float32, device=device))
    D_flat = nn.Parameter(torch.rand(n * T, dtype=torch.float32, device=device) * 0.8 + 0.2)
    sigma_hat_vec = nn.Parameter(torch.rand(n, dtype=torch.float32, device=device) * 0.8 + 0.2)  # vector sigma
    B_free = nn.Parameter(torch.zeros((n, n), dtype=torch.float32, device=device))
    optimizer = Adam([A_hat, D_flat, sigma_hat_vec, B_free], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        B = B_free.clone()
        B.fill_diagonal_(0.0)
        loss = loss_function_torch(A_hat, D_flat, sigma_hat_vec, B, Cov_list, T, n)
        loss.backward()
        optimizer.step()

        if step % 1000 == 0 or step == num_steps - 1:
            maxcor, _, _ = compute_mcc(A_hat, X_list, Z_list, n)
            print(f"Step {step} — Loss: {loss.item():.6f}, MCC: {maxcor:.4f}")

    A_hat_opt = A_hat.detach().cpu().numpy()
    D_hats = [D_flat.detach().cpu().numpy()[i * n:(i + 1) * n] for i in range(T)]
    sigma_hat_vec = sigma_hat_vec.detach().cpu().numpy()
    sigma_hat = np.diag(sigma_hat_vec)
    B_hat = B_free.detach().cpu().numpy()

    return A_hat_opt, D_hats, sigma_hat, B_hat, loss.item()


# -----------------------------
# Multi-Initialization Wrapper
# -----------------------------
def estimate_parameters_multiple_initializations_torch(X_list, Z_list, T, n, xn, num_steps=2000, lr=1e-3, num_initializations=5):
    Cov_list = [np.cov(X_list[t], rowvar=False) for t in range(T)]
    best_loss = np.inf
    best_result = None

    for i in range(num_initializations):
        print(f"Initialization {i + 1}/{num_initializations}")
        A_hat, D_hats, sigma_hat, B_hat, final_loss = run_torch_optimization(
            Cov_list, X_list, Z_list, T, n, xn, num_steps=num_steps, lr=lr
        )
        if final_loss < best_loss:
            best_loss = final_loss
            best_result = (A_hat, D_hats, sigma_hat, B_hat, final_loss)

    return best_result


# -----------------------------
# Main Script
# -----------------------------
def main(args):
    mcc_scores = []
    args.model_dir = os.path.join(args.model_dir, f'lr{args.lr}_initial{args.num_initializations}')
    args.x_n = int(args.x_n * args.n)
    args.T = int(args.T * args.n)
    if args.nonlinear:
        nonlinear = 'nonlinear'
    else:
        nonlinear = 'linear'
    for seed in args.seeds:
        set_seed(seed)
        start_time = time.perf_counter()
        args.save_dir = os.path.join(
            args.model_dir,
            f'n{args.n}_xn{args.x_n}_T{args.T}_N{args.N}_k{args.graph_dense}_{args.noise_type}_noise_{nonlinear}_rs{seed}'
        )
        os.makedirs(args.save_dir, exist_ok=True)
        results_path = os.path.join(args.save_dir, "results.npz")

        # Generate data
        X_list, A_true, B_true, Ds, sigma_vec, Z_list, skeleton = generate_data(
            T=args.T, n=args.n, xn=args.x_n, N=args.N,
            graph_dense=args.graph_dense, graph_type=args.graph_type, noise_type=args.noise_type, nn_nonlinear=args.nonlinear
        )

        if args.evaluate:
            if os.path.exists(results_path):
                data = np.load(results_path, allow_pickle=True)
                A_hat = data['A_hat']
                cor_abs = data['cor_abs']
                maxcor = data['MCC'].item()
                print(f"Loaded MCC: {maxcor:.4f}")
            else:
                raise FileNotFoundError(f"No saved results found in {results_path}")
        else:
            A_hat, D_hats, sigma_hat, B_hat, final_loss = estimate_parameters_multiple_initializations_torch(
                X_list, Z_list, args.T, args.n, args.x_n, args.num_steps, args.lr, args.num_initializations)

            # Evaluation
            maxcor, mcc_list, cov_list = compute_mcc(A_hat, X_list, Z_list, args.n)

            np.savez(results_path, A_hat=A_hat, D_hats=D_hats, sigma_hat=sigma_hat,
                     B_hat=B_hat, MCC=maxcor)
            print(f"Results saved to {results_path}")

        fileobj = open(args.model_dir + '.csv', 'a+')
        writer = csv.writer(fileobj)
        writer.writerow([args.n, args.x_n, args.T, args.N, args.graph_dense, args.noise_type, nonlinear, seed, maxcor])
        fileobj.close()
        mcc_scores.append(maxcor)

        end_time = time.perf_counter()

        print(f"Seed {seed} — MCC: {maxcor:.4f}, runtime: {end_time - start_time:.2f}s")

    print(np.mean(mcc_scores), np.std(mcc_scores))

    fileobj = open(args.model_dir + '_SUM_MCC' + '.csv', 'a+')
    writer = csv.writer(fileobj)
    writer.writerow([args.n, args.x_n, args.T, args.N, args.graph_dense,
                     args.noise_type, nonlinear, np.mean(mcc_scores), np.std(mcc_scores)])
    fileobj.close()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    parser = argparse.ArgumentParser(description="multi-domain latent variable estimation.")
    parser.add_argument("--T", type=int, default=2)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--x-n", type=int, default=1)
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--seeds", nargs='+', type=int, default=[2, 22])
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num-steps", type=int, default=50000)
    parser.add_argument("--num-initializations", type=int, default=3)
    parser.add_argument("--graph-dense", type=float, default=2.0)
    parser.add_argument("--graph-type", type=str, default="ER")
    parser.add_argument("--noise-type", type=str, default="gauss")
    parser.add_argument("--nonlinear", action='store_true')
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--model-dir", type=str, default="MuDo")
    parser.add_argument("--save-dir", type=str, default="")
    args = parser.parse_args()
    main(args)
