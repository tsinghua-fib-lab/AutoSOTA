#!/usr/bin/env python3
"""Reproduction script for paper 2855: NAMMD blob experiment, epsilon=0.3.

Uses sigma annealing (continuation method) for gradient-based distribution
construction: starts with large bandwidth and gradually anneals to the trained
(optimal) bandwidth. This overcomes the local-minimum problem with tiny bandwidths
on low-dimensional data while preserving the paper's algorithm.
"""
import numpy as np
import torch
import argparse
import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../.."))
from dataloader import load_data, MatConvert, NAMMD_discrete
from kernel_selection import (set_random_seed, split_selected_bandwidths,
                               select_gaussian_bandwidths_reference)
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="BLOB")
parser.add_argument("--N1", default=500, type=int)
parser.add_argument("--N", default=3000, type=int)
parser.add_argument("--rs", default=483, type=int)
parser.add_argument("--n_exp", default=10, type=int)
parser.add_argument("--n_test", default=100, type=int)
parser.add_argument("--n_per", default=100, type=int)
parser.add_argument("--alpha", default=0.05, type=float)
parser.add_argument("--eps", default=0.3, type=float)
parser.add_argument("--eps_gap", default=0.01, type=float)
parser.add_argument("--lr_data", default=0.01, type=float)
parser.add_argument("--device_str", default="cuda")
args = parser.parse_args()

device = torch.device(args.device_str)
dtype = torch.float
RESULT_DIR = "../../Results/power_epsn"
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Asymptotic variance computation (from paper utils.py)
# ---------------------------------------------------------------------------
def _scalar_tensor(value, device, dtype):
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=dtype).reshape(())
    return torch.tensor(float(value), device=device, dtype=dtype)

def pairwise_sq_dists(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dists = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    dists[dists < 0] = 0
    return dists

def _asymptotic_values(Fea, N1, sigma0, device):
    sigma = _scalar_tensor(sigma0, device, dtype) if not torch.is_tensor(sigma0) else sigma0
    X = Fea[0:N1, :]
    Y = Fea[N1:, :]
    Dxx = pairwise_sq_dists(X, X)
    Dyy = pairwise_sq_dists(Y, Y)
    Dxy = pairwise_sq_dists(X, Y)
    Kx = torch.exp(-Dxx / sigma**2)
    Ky = torch.exp(-Dyy / sigma**2)
    Kxy = torch.exp(-Dxy / sigma**2)

    def h1_mean_var_gram(Kx, Ky, Kxy):
        Kxxy = torch.cat((Kx,Kxy),1)
        Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
        Kxyxy = torch.cat((Kxxy,Kyxy),0)
        nx = Kx.shape[0]; ny = Ky.shape[0]
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
        mmd2 = xx - 2 * xy + yy
        return mmd2, Kxyxy, 4 - xx - yy

    TEMP = h1_mean_var_gram(Kx, Ky, Kxy)
    NAMMD_value = TEMP[0]/TEMP[2]
    MMD_value = TEMP[0]
    Kxyxy = TEMP[1]
    ind = np.arange(2 * N1)
    indx = ind[:N1]; indy = ind[N1:]
    Kx_mat = Kxyxy[np.ix_(indx, indx)]
    Ky_mat = Kxyxy[np.ix_(indy, indy)]
    Kxy_mat = Kxyxy[np.ix_(indx, indy)]
    EE = torch.ones(N1).to(device)
    Kx_ = Kx_mat * (1-torch.eye(N1)).to(device)
    Ky_ = Ky_mat * (1-torch.eye(N1)).to(device)

    Xxi1 = (N1*(N1-1)*(N1-2))**(-1)*(torch.norm(Kx_@EE,p=2)**2-torch.norm(Kx_,p="fro")**2) - \
    (N1*(N1-1)*(N1-2)*(N1-3))**(-1)*((EE@Kx_@EE)**2-4*(torch.norm(Kx_@EE,p=2)**2+2*torch.norm(Kx_,p="fro")**2))
    Yxi1 = (N1*(N1-1)*(N1-2))**(-1)*(torch.norm(Ky_@EE,p=2)**2-torch.norm(Ky_,p="fro")**2) - \
    (N1*(N1-1)*(N1-2)*(N1-3))**(-1)*((EE@Ky_@EE)**2-4*(torch.norm(Ky_@EE,p=2)**2+2*torch.norm(Ky_,p="fro")**2))

    varxi1 = Xxi1 + Yxi1 + (N1**2*(N1-1))**(-1)*(torch.norm(Kxy_mat@EE,p=2)**2-torch.norm(Kxy_mat,p="fro")**2) - \
        2*(N1**2*(N1-1)**2)**(-1)*((EE@Kxy_mat@EE)**2-torch.norm(Kxy_mat.T@EE,p=2)**2-torch.norm(Kxy_mat@EE,p=2)**2+torch.norm(Kxy_mat,p="fro")**2) + \
        (N1**2*(N1-1))**(-1)*(torch.norm(Kxy_mat.T@EE,p=2)**2-torch.norm(Kxy_mat,p="fro")**2) - \
        2*(N1**2*(N1-1))**(-1)*EE@Kx_@Kxy_mat@EE + 2*(N1*N1*(N1-1)*(N1-2))**(-1)*(EE@Kx_@EE*EE@Kxy_mat@EE-2*EE@Kx_@Kxy_mat@EE) - \
        2*(N1**2*(N1-1))**(-1)*EE@Ky_@Kxy_mat.T@EE + 2*(N1*N1*(N1-1)*(N1-2))**(-1)*(EE@Ky_@EE*EE@Kxy_mat.T@EE-2*EE@Ky_@Kxy_mat.T@EE)

    varxi2 = Xxi1 + Yxi1 + 2* N1**(-2)*torch.norm(Kxy_mat,p="fro")**2 - 2*(N1**2*(N1-1)**2)**(-1)*((EE@Kxy_mat@EE)**2-torch.norm(Kxy_mat.T@EE,p=2)**2-torch.norm(Kxy_mat@EE,p=2)**2+torch.norm(Kxy_mat,p="fro")**2) - \
    4*(N1**2*(N1-1))**(-1)*EE@Kx_@Kxy_mat@EE + 4*(N1*N1*(N1-1)*(N1-2))**(-1)*(EE@Kx_@EE*EE@Kxy_mat@EE-2*EE@Kx_@Kxy_mat@EE) -\
    4*(N1**2*(N1-1))**(-1)*EE@Ky_@Kxy_mat.T@EE + 4*(N1*N1*(N1-1)*(N1-2))**(-1)*(EE@Ky_@EE*EE@Kxy_mat.T@EE-2*EE@Ky_@Kxy_mat.T@EE)

    varEst = (4*(N1-2)/(N1*(N1-1)) * varxi1 + 2/(N1*(N1-1)) * varxi2)
    Var_all = varEst/TEMP[2]**2
    return MMD_value, NAMMD_value, varEst, Var_all

# ---------------------------------------------------------------------------
# Annealed distribution construction
# ---------------------------------------------------------------------------
def construct_annealed(name, N, rs, target_eps, eps_gap, lr, sigma0_tuple, device, dtype,
                       sigma_levels=None, iters_per_level=3000, tol=1e-6):
    """Construct distributions using sigma annealing (continuation method).
    
    Starts with large bandwidth and gradually anneals to the trained bandwidth,
    overcoming local minima in the optimization landscape for low-dimensional data.
    """
    sigma_mmd, sigma_nammd = split_selected_bandwidths(sigma0_tuple)
    target_sigma = sigma_nammd.item()
    
    if sigma_levels is None:
        # Generate annealing schedule from 1.0 down to target sigma
        if target_sigma < 0.1:
            sigma_levels = [1.0, 0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08, target_sigma]
        else:
            sigma_levels = [1.0, target_sigma]
    
    X, Y = load_data(name, N, rs+100, 1)
    set_random_seed(rs + 100)
    X = MatConvert(X, device, dtype)
    Y = MatConvert(Y, device, dtype)
    X.requires_grad = True
    Y.requires_grad = True
    optimizer = torch.optim.Adam([X, Y], lr=lr)
    
    # Anneal through sigma levels for first distribution (eps)
    for sigma_val in sigma_levels:
        sigma_t = torch.tensor(sigma_val, device=device, dtype=dtype)
        MMD, Reg = NAMMD_discrete(X, Y, N, sigma_t, 1)
        for t in range(iters_per_level):
            loss = (MMD/Reg - target_eps)**2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            MMD, Reg = NAMMD_discrete(X, Y, N, sigma_t, 1)
            if abs((MMD/Reg).item() - target_eps) < tol:
                break
    
    X1 = X.detach().clone()
    Y1 = Y.detach().clone()
    MMD1_mmd = NAMMD_discrete(X1, Y1, N, sigma_mmd, 1)[0].item()
    MMD1_nammd, Reg1_nammd = NAMMD_discrete(X1, Y1, N, sigma_nammd, 1)
    NAMMD1 = (MMD1_nammd / Reg1_nammd).item()
    
    # Continue from same X, Y for second distribution (eps + eps_gap)
    for sigma_val in sigma_levels:
        sigma_t = torch.tensor(sigma_val, device=device, dtype=dtype)
        MMD, Reg = NAMMD_discrete(X, Y, N, sigma_t, 1)
        for t in range(iters_per_level):
            loss = (MMD/Reg - target_eps - eps_gap)**2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            MMD, Reg = NAMMD_discrete(X, Y, N, sigma_t, 1)
            if abs((MMD/Reg).item() - target_eps - eps_gap) < tol:
                break
    
    X2 = X.detach().clone()
    Y2 = Y.detach().clone()
    MMD2_mmd = NAMMD_discrete(X2, Y2, N, sigma_mmd, 1)[0].item()
    MMD2_nammd, Reg2_nammd = NAMMD_discrete(X2, Y2, N, sigma_nammd, 1)
    NAMMD2 = (MMD2_nammd / Reg2_nammd).item()
    
    return X1, Y1, MMD1_mmd, NAMMD1, X2, Y2, MMD2_mmd, NAMMD2

def testing(X, Y, MMD, NAMMD, N1, rs, sigma0_tuple, n_test, n_per, alpha, device, dtype):
    H_MMD = np.zeros(n_test)
    H_NAMMD = np.zeros(n_test)
    set_random_seed(rs)
    sigma_mmd, sigma_nammd = split_selected_bandwidths(sigma0_tuple)
    threshold = norm.ppf(1 - alpha)
    for k in range(n_test):
        indices_X = torch.randint(0, len(X), (N1,))
        X_test = X[indices_X]
        indices_Y = torch.randint(0, len(Y), (N1,))
        Y_test = Y[indices_Y]
        Fea = torch.cat((X_test, Y_test))
        MMD_value, _, varEst, _ = _asymptotic_values(Fea, N1, sigma_mmd, device)
        _, NAMMD_value, _, Var_all = _asymptotic_values(Fea, N1, sigma_nammd, device)
        NAMMD_test = (NAMMD_value - NAMMD) / torch.sqrt(Var_all.clamp_min(1e-12))
        MMD_Test = (MMD_value - MMD) / torch.sqrt(varEst.clamp_min(1e-12))
        H_NAMMD[k] = int(NAMMD_test > threshold)
        H_MMD[k] = int(MMD_Test > threshold)
    return H_MMD, H_NAMMD

# ===========================================================================
# Main experiment
# ===========================================================================
eps, rs, N, N1 = args.eps, args.rs, args.N, args.N1
Results = np.zeros((1, 2, args.n_exp))
all_details = []

for kk in range(args.n_exp):
    # Bandwidth selection from reference samples
    set_random_seed(kk + rs)
    X_train, Y_train = load_data(args.name, N1, kk + rs, 1)
    S_train = np.concatenate((X_train, Y_train), axis=0)
    S_train = MatConvert(S_train, device, dtype)
    sigma_mmd, sigma_nammd, _ = select_gaussian_bandwidths_reference(
        S_train, N1, seed=kk + rs,
        max_reference_samples=min(N1, 500),
        num_bandwidths=25, verbose=False
    )
    sigma0_tuple = (sigma_mmd.detach(), sigma_nammd.detach())
    print(f"[Run {kk+1}/{args.n_exp}] sigma_mmd={sigma_mmd.item():.6g}, sigma_nammd={sigma_nammd.item():.6g}", flush=True)
    
    # Distribution construction with sigma annealing
    X1, Y1, MMD1, NAMMD1, X2, Y2, MMD2, NAMMD2 = construct_annealed(
        args.name, N1, rs + kk, eps, args.eps_gap, args.lr_data,
        sigma0_tuple, device, dtype, iters_per_level=3000, tol=1e-6
    )
    print(f"[Run {kk+1}/{args.n_exp}] NAMMD1={NAMMD1:.6f}, NAMMD2={NAMMD2:.6f}, gap={NAMMD2-NAMMD1:.6f} (target_gap={args.eps_gap})", flush=True)
    
    # Two-sample testing
    H_MMD, H_NAMMD = testing(
        X2, Y2, MMD1, NAMMD1, N, kk + rs + 100,
        sigma0_tuple, args.n_test, args.n_per, args.alpha, device, dtype
    )
    Results[0, 0, kk] = H_NAMMD.sum() / args.n_test
    Results[0, 1, kk] = H_MMD.sum() / args.n_test
    all_details.append((NAMMD1, NAMMD2, Results[0,0,kk], Results[0,1,kk]))
    print(f"[Run {kk+1}/{args.n_exp}] Power: NAMMD={Results[0,0,kk]:.3f}, MMD={Results[0,1,kk]:.3f}", flush=True)
    
    np.savetxt(f"{RESULT_DIR}/{args.name}_eps{eps}_intermediate.txt",
               Results[0, :, :kk+1].T, fmt="%.3f")

# Final statistics
Final = np.zeros((1, 2, 2))
Final[0, 0, 0] = Results[0, 0].mean()
Final[0, 0, 1] = Results[0, 0].std(ddof=1) / np.sqrt(args.n_exp)
Final[0, 1, 0] = Results[0, 1].mean()
Final[0, 1, 1] = Results[0, 1].std(ddof=1) / np.sqrt(args.n_exp)

np.savetxt(f"{RESULT_DIR}/{args.name}_eps{eps}_Final.txt", Final.reshape(1, -1), fmt="%.4f")

print("\n" + "="*60, flush=True)
print(f"FINAL RESULTS", flush=True)
print(f"Dataset: {args.name}, eps={eps}, eps_gap={args.eps_gap}", flush=True)
print(f"n_exp={args.n_exp}, n_test={args.n_test}, N1={N1}, N={N}", flush=True)
print(f"NAMMD test power: {Final[0,0,0]:.3f} +/- {Final[0,0,1]:.3f}", flush=True)
print(f"MMD test power:   {Final[0,1,0]:.3f} +/- {Final[0,1,1]:.3f}", flush=True)
print(f"Per-run NAMMD power: {Results[0,0,:]}", flush=True)
print(f"Per-run MMD power:   {Results[0,1,:]}", flush=True)
print("="*60, flush=True)
