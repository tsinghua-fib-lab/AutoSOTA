from torch import nn
from torch.optim import Adam
import numpy as np
from sklearn.decomposition import FastICA
import random
import argparse
import os
from sklearn.preprocessing import scale
import pandas as pd
import matplotlib.pyplot as plt
import torch
from nilearn.plotting import plot_markers
from qndiag import qndiag
import tensorly as tl
from tensorly.decomposition import parafac


# -----------------------------
# set random seed
# -----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Batched Loss Function
# -----------------------------
def loss_function_batched(
    A_hat,          # (K, xn, n)
    D_flat,         # (K, T*n)
    sigma_vec,      # (K, n)
    B_free,         # (K, n, n)
    Cov_tensor,     # (T, n, n)
    T, n
):
    device = A_hat.device
    K = A_hat.shape[0]

    # Zero diagonal of B
    B = B_free.clone()
    idx = torch.arange(n, device=device)
    B[:, idx, idx] = 0.0

    I = torch.eye(n, device=device).expand(K, n, n)

    # Sigma^2
    Sigma2 = torch.diag_embed(sigma_vec ** 2)

    # (I-B)^{-1} Sigma^2 (I-B)^{-T} using solves
    X = torch.linalg.solve(I - B, Sigma2)
    M = torch.linalg.solve(
        (I - B).transpose(1, 2),
        X.transpose(1, 2)
    ).transpose(1, 2)

    # D tensor
    D = D_flat.view(K, T, n)

    # ADAT computation
    ADAT = torch.einsum(
        "kin,ktn,knm,ktm,kjm->ktij",
        A_hat,
        D,
        M,
        D,
        A_hat
    )

    diff = Cov_tensor.unsqueeze(0) - ADAT
    loss = (diff ** 2).sum(dim=(1, 2, 3))  # (K,)

    return loss


# -----------------------------
# Batched Optimization
# -----------------------------
def run_batched_optimization(
    Cov_list,
    T, n, xn,
    num_initializations=50,
    num_steps=10000,
    lr=5e-3
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = num_initializations

    # Move covariances to GPU once
    Cov_tensor = torch.tensor(
        np.stack(Cov_list),
        dtype=torch.float32,
        device=device
    )

    # Parameters (batched)
    A_hat = nn.Parameter(torch.rand(K, xn, n, device=device))
    D_flat = nn.Parameter(torch.rand(K, T * n, device=device) * 0.8 + 0.2)
    sigma_vec = nn.Parameter(torch.rand(K, n, device=device) * 0.8 + 0.2)
    B_free = nn.Parameter(torch.zeros(K, n, n, device=device))

    optimizer = Adam([A_hat, D_flat, sigma_vec, B_free], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        loss_vec = loss_function_batched(
            A_hat, D_flat, sigma_vec, B_free,
            Cov_tensor, T, n
        )

        loss = loss_vec.mean()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0 or step == num_steps - 1:
            print(
                f"Step {step:5d} | "
                f"Mean loss: {loss.item():.4e} | "
                f"Best loss: {loss_vec.min().item():.4e}"
            )

    # Select best initialization
    best_idx = loss_vec.argmin().item()

    A_best = A_hat[best_idx].detach().cpu().numpy()
    D_best = D_flat[best_idx].detach().cpu().numpy()
    sigma_best = np.diag(sigma_vec[best_idx].detach().cpu().numpy())
    B_best = B_free[best_idx].detach().cpu().numpy()

    D_hats = [D_best[i*n:(i+1)*n] for i in range(T)]

    return A_best, D_hats, sigma_best, B_best, loss_vec[best_idx].item()


# -----------------------------
# Plot brain
# -----------------------------
def plot_brain(W, ROIcoord, clusterID, threshold, title, save_path):
    """
    INPUT:
        - ROIcoord: MNNI coordinates
        - clusterID: which cluster should we plot

    """
    absmax = np.max(np.abs(W[:, clusterID]))
    if absmax == 0:
        print("Nothing to plot")
        return



    ii = np.where(np.abs(W[:, clusterID]) > threshold)[0]
    weights = W[:, clusterID]
    if len(ii) == 0:
        print("No nodes passed the threshold — nothing to plot.")
        return
    node_values = weights[ii]
    plot_markers(
        node_values,
        ROIcoord[ii],
        node_cmap='coolwarm',
        node_vmin=-absmax,
        node_vmax=absmax,
        alpha=0.8,
        colorbar=False,
        # title=False
    )
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # PDF ok here
    plt.show()
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main(args):
    set_seed(args.seed)
    args.model_dir = os.path.join(args.model_dir, f'lr{args.lr}_initial{args.num_initializations}')

    data_dir = os.path.join(os.getcwd(), "data", "camCAN")
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    X_list = [pd.read_csv(os.path.join(data_dir, f)).to_numpy() for f in files]
    Cov_list = [np.cov(scale(X).T) for X in X_list if not np.isnan(X).any()]

    T = len(Cov_list)
    print(f"Collected {T} valid subjects")

    if args.method == 'MuDo':

        A_hat, D_hats, sigma_hat, B_hat, final_loss = run_batched_optimization(
            Cov_list,
            T,
            args.n,
            args.x_n,
            num_initializations=args.num_initializations,
            num_steps=args.num_steps,
            lr=args.lr
        )

        args.save_dir = os.path.join(
                args.model_dir,
                f'n{args.n}_noise_rs{args.seed}'
            )
        os.makedirs(args.save_dir, exist_ok=True)
        results_path = os.path.join(args.save_dir, "results.npz")

        np.savez(
            results_path,
            A_hat=A_hat,
            D_hats=np.array(D_hats, dtype=object),
            sigma_hat=sigma_hat,
            B_hat=B_hat,
            final_loss=final_loss,
            lr=args.lr,
            num_steps=args.num_steps,
            num_initializations=args.num_initializations,
            seed=args.seed
        )

        print(f"Saved results to {results_path}")
    elif args.method == 'ICA':
        X_all = np.vstack(X_list)
        ica = FastICA(
            n_components=args.n,
            whiten='arbitrary-variance',  # already whitened by PCA, 'unit-variance', 'arbitrary-variance'
            max_iter=10000,
            tol=1e-6
        )
        S_est = ica.fit_transform(X_all)
        A_hat = ica.mixing_

    elif args.method == 'qndiag':
        Cov_full = np.mean(Cov_list, axis=0)
        evals, evecs = np.linalg.eigh(Cov_full)
        idx = np.argsort(evals)[::-1]
        evals_n = evals[idx[:args.n]]
        evecs_n = evecs[:, idx[:args.n]]  # shape: (xn × n)
        W = (evecs_n / np.sqrt(evals_n)).T  # shape: (n × xn)
        Cov_list_reduced = np.array([
            W @ Cov_list[t] @ W.T for t in range(args.T)
        ])
        B_reduced, _ = qndiag(Cov_list_reduced)
        B_full = B_reduced @ W  # shape: (n × xn)

        A_hat = np.linalg.pinv(B_full)
        X_all = np.vstack(X_list)
        S_est = B_full @ (X_all.T)
        print(np.cov(scale(S_est.T).T))
    elif args.method == 'candecomp':
        X_tensor = tl.tensor(Cov_list, dtype=float)
        CP_model = parafac(
            X_tensor,
            rank=args.n,
            init='svd',
            n_iter_max=10000,
            tol=1e-10
        )

        _, _, A_hat = CP_model.factors
        B_full = np.linalg.pinv(A_hat)
        X_all = np.vstack(X_list)
        S_est = B_full @ (X_all.T)
        print(np.cov(scale(S_est.T).T))




    ROIcoord = np.loadtxt(os.path.join("data", "MNI.txt"))
    for i in range(args.n):
        plot_brain(A_hat, ROIcoord, i, 0.0, f'latent{i+1}', f"{args.method}_no_legend_latent_{i+1}.pdf")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    parser = argparse.ArgumentParser(description="camCAN.")
    parser.add_argument("--method", type=str, default="MuDo")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--x-n", dest="x_n", type=int, default=264)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--num-initializations", type=int, default=20)
    parser.add_argument("--model-dir", type=str, default="camCAN")
    parser.add_argument("--save-dir", type=str, default="")


    args = parser.parse_args()
    main(args)
