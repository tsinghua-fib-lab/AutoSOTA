import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def compute_mcc_g(A_hat, X_list, Z_list, n, inv=False):
    if isinstance(A_hat, np.ndarray):
        # Convert numpy → torch; use CPU by default
        A_hat = torch.from_numpy(A_hat).float()
        device = torch.device("cpu")
    else:
        # It's already a torch tensor
        device = A_hat.device

    def corrcoef_torch(X, Y):
        Xm = X - X.mean(dim=0, keepdim=True)
        Ym = Y - Y.mean(dim=0, keepdim=True)
        cov = (Xm.T @ Ym) / (X.shape[0] - 1)
        std_X = Xm.std(dim=0, unbiased=True).unsqueeze(1)
        std_Y = Ym.std(dim=0, unbiased=True).unsqueeze(0)
        corr = cov / (std_X * std_Y + 1e-8)
        return corr

    T = len(X_list)
    mcc_list = []
    cor_list = []

    for t in range(T):
        # X_t : (N, xn), Z_t : (N, n)
        X_t = torch.tensor(X_list[t], dtype=torch.float32, device=device)
        Z_t = torch.tensor(Z_list[t], dtype=torch.float32, device=device)

        # Project X_t to latent space

        if inv:
            hz_t = X_t @ A_hat.T
        else:

            AtA = A_hat.T @ A_hat
            AtX = A_hat.T @ X_t.T

            hz_t = torch.linalg.solve(AtA, AtX).T

        # Compute correlation between true Z_t and estimated hz_t
        cor_abs_t = torch.abs(corrcoef_torch(Z_t, hz_t))
        cor_abs_np = cor_abs_t.detach().cpu().numpy()

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(-cor_abs_np)
        mcc_t = cor_abs_np[row_ind, col_ind].sum() / n

        mcc_list.append(mcc_t)
        cor_list.append(cor_abs_np)

    # Average MCC over domains
    avg_mcc = np.mean(mcc_list)

    return avg_mcc, mcc_list, cor_list

def compute_mcc(A_hat, X_list, Z_list, n, inv=False):
    if isinstance(A_hat, np.ndarray):
        # Convert numpy → torch; use CPU by default
        A_hat = torch.from_numpy(A_hat).float()
        device = torch.device("cpu")
    else:
        # It's already a torch tensor
        device = A_hat.device

    def corrcoef_torch(X, Y):
        Xm = X - X.mean(dim=0, keepdim=True)
        Ym = Y - Y.mean(dim=0, keepdim=True)
        cov = (Xm.T @ Ym) / (X.shape[0] - 1)
        std_X = Xm.std(dim=0, unbiased=True).unsqueeze(1)
        std_Y = Ym.std(dim=0, unbiased=True).unsqueeze(0)
        corr = cov / (std_X * std_Y + 1e-8)
        return corr

    T = len(X_list)
    Z_all = []
    hz_all = []

    for t in range(T):
        # X_t : (N, xn), Z_t : (N, n)
        X_t = torch.tensor(X_list[t], dtype=torch.float32, device=device)
        Z_t = torch.tensor(Z_list[t], dtype=torch.float32, device=device)

        # Project X_t to latent space

        if inv:
            hz_t = X_t @ A_hat.T
        else:

            AtA = A_hat.T @ A_hat
            AtX = A_hat.T @ X_t.T

            hz_t = torch.linalg.solve(AtA, AtX).T



        # Accumulate true Z
        Z_all.append(Z_t)
        # Accumulate hZ
        hz_all.append(hz_t)
    # 2. Concatenate across time
    Z_cat = torch.cat(Z_all, dim=0)
    hz_cat = torch.cat(hz_all, dim=0)

    # 3. Correlation
    cor_abs = torch.abs(corrcoef_torch(Z_cat, hz_cat))
    cor_abs_np = cor_abs.detach().cpu().numpy()

    # 4. Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-cor_abs_np)
    mcc = cor_abs_np[row_ind, col_ind].sum() / n
    return mcc, cor_abs_np, col_ind

def amari_distance_rect(A, A_hat, eps=1e-12, Inv=False):
    """
    Amari distance for possibly rectangular mixing matrices.

    A     : (m, n) true mixing matrix
    A_hat : (m, n) estimated mixing matrix
    """
    n = A.shape[1]
    if Inv:
        P = A_hat @ A

    else:
        P = np.linalg.pinv(A) @ A_hat
    P = np.abs(P)

    row_term = np.sum(
        (np.sum(P, axis=1) / (np.max(P, axis=1) + eps)) - 1
    )
    col_term = np.sum(
        (np.sum(P, axis=0) / (np.max(P, axis=0) + eps)) - 1
    )

    return (row_term + col_term) / (2 * n)

