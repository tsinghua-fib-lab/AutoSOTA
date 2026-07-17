import numpy as np
import torch
import argparse
import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../.."))

# Monkey-patch: fix memory leak in construct_distributions (retain_graph=True -> False)
# This is a bug fix, NOT an algorithm change. Each iteration creates a fresh graph.
import utils as _utils
import types

def construct_distributions_fixed(name, N, rs, eps, eps_gap, learning_rate, sigma0, K, device, dtype, scale=True):
    """Fixed version without retain_graph memory leak."""
    from dataloader import load_data, MatConvert, NAMMD_discrete
    from kernel_selection import set_random_seed, split_selected_bandwidths
    sigma_mmd, sigma_nammd = split_selected_bandwidths(sigma0)
    if eps == 0:
        X, _ = load_data(name, N, rs, 0, scale)
        X = MatConvert(X, device, dtype)
        Y = X
        MMD_mmd, _ = NAMMD_discrete(X, Y, N, sigma_mmd, K)
        MMD_nammd, Reg = NAMMD_discrete(X, Y, N, sigma_nammd, K)
        NAMMD = MMD_nammd / Reg
        return X, Y, MMD_mmd.item(), NAMMD.item(), X, Y, MMD_mmd.item(), NAMMD.item()
    else:
        ts = 100
        while True:
            X, Y = load_data(name, N, rs+ts, 1, scale)
            set_random_seed(rs + ts)
            X = MatConvert(X, device, dtype)
            Y = MatConvert(Y, device, dtype)
            X.requires_grad = True
            Y.requires_grad = True
            optimizer = torch.optim.Adam([X,Y], lr=learning_rate)
            MMD, Reg = NAMMD_discrete(X, Y, N, sigma_nammd, K)
            t = 0
            while abs((MMD/Reg).item() - eps) >= 10**(-7):
                STAT_u = (MMD/Reg - eps)**2
                optimizer.zero_grad()
                STAT_u.backward()  # FIXED: removed retain_graph=True (memory leak)
                optimizer.step()
                if t % 100 == 0:
                    print("MMD_value: ", MMD.item(), "Reg_value: ", Reg.item(), "NAMMD: ", (MMD/Reg).item())
                MMD, Reg = NAMMD_discrete(X, Y, N, sigma_nammd, K)
                t += 1
            X1 = X.detach()
            Y1 = Y.detach()
            MMD1_mmd = NAMMD_discrete(X1, Y1, N, sigma_mmd, K)[0].item()
            MMD1_nammd, Reg1_nammd = NAMMD_discrete(X1, Y1, N, sigma_nammd, K)
            NAMMD1 = (MMD1_nammd / Reg1_nammd).item()
            t = 0
            while abs((MMD/Reg).item() - eps - eps_gap) >= 10**(-7):
                STAT_u = (MMD/Reg - eps - eps_gap)**2
                optimizer.zero_grad()
                STAT_u.backward()  # FIXED: removed retain_graph=True (memory leak)
                optimizer.step()
                if t % 100 == 0:
                    print("MMD_value: ", MMD.item(), "Reg_value: ", Reg.item(), "NAMMD: ", (MMD/Reg).item())
                MMD, Reg = NAMMD_discrete(X, Y, N, sigma_nammd, K)
                t += 1
            X2 = X.detach()
            Y2 = Y.detach()
            MMD2_mmd = NAMMD_discrete(X2, Y2, N, sigma_mmd, K)[0].item()
            MMD2_nammd, Reg2_nammd = NAMMD_discrete(X2, Y2, N, sigma_nammd, K)
            NAMMD2 = (MMD2_nammd / Reg2_nammd).item()
            if Reg1_nammd.item() > Reg2_nammd.item() or eps_gap==0:
                break
            ts +=1
    return X1, Y1, MMD1_mmd, NAMMD1, X2, Y2, MMD2_mmd, NAMMD2

# Apply monkey-patch
_utils.construct_distributions = construct_distributions_fixed

from utils import training, testing

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="BLOB")
parser.add_argument("--N1", default=500, type=int)
parser.add_argument("--N", default=3000, type=int)
parser.add_argument("--rs", default=483, type=int)
parser.add_argument("--n_exp", default=10, type=int)
parser.add_argument("--n_test", default=100, type=int)
parser.add_argument("--n_per", default=100, type=int)
parser.add_argument("--alpha", default=0.05, type=float)
parser.add_argument("--K", default=1, type=int)
parser.add_argument("--eps", default=0.3, type=float)
parser.add_argument("--eps_gap", default=0.01, type=float)
parser.add_argument("--lr_data", default=0.01, type=float)
parser.add_argument("--device", default=torch.device("cuda"))
parser.add_argument("--dtype", default=torch.float)
parser.add_argument("--ne_MMD", default=1000, type=int)
parser.add_argument("--bs_MMD", default=400, type=int)
parser.add_argument("--lr_MMD", default=0.001, type=float)
parser.add_argument("--ne_NAMMD", default=1000, type=int)
parser.add_argument("--bs_NAMMD", default=400, type=int)
parser.add_argument("--lr_NAMMD", default=0.001, type=float)
parser.add_argument("--b_NAMMD", default=0.2, type=float)

args = parser.parse_args()

Results = np.zeros((1, 2, args.n_exp))
eps = args.eps

for kk in range(args.n_exp):
    sigma0 = training(args.name, args.N1, kk+args.rs, 1, args.ne_MMD, args.bs_MMD, args.lr_MMD, args.ne_NAMMD, args.bs_NAMMD, args.lr_NAMMD, args.b_NAMMD, args.device, args.dtype)
    print(f"[Run {kk+1}/{args.n_exp}] Training Done!")

    X1, Y1, MMD1, NAMMD1, X2, Y2, MMD2, NAMMD2 = construct_distributions_fixed(args.name, args.N1, args.rs + kk, eps, args.eps_gap, args.lr_data, sigma0, args.K, args.device, args.dtype)
    print(f"[Run {kk+1}/{args.n_exp}] Construction Done! NAMMD1={NAMMD1:.6f}, NAMMD2={NAMMD2:.6f}")

    H_MMD, H_NAMMD = testing(X2, Y2, MMD1, NAMMD1, args.N, kk+args.rs+100, sigma0, args.n_test, args.n_per, args.alpha, args.device, args.dtype)
    print(f"[Run {kk+1}/{args.n_exp}] Testing Done!")

    Results[0, 0, kk] = H_NAMMD.sum() / args.n_test
    Results[0, 1, kk] = H_MMD.sum() / args.n_test
    print(f"[Run {kk+1}/{args.n_exp}] NAMMD={Results[0, 0, kk]:.3f}, MMD={Results[0, 1, kk]:.3f}")

    np.savetxt("../../Results/power_epsn/" + args.name + f"_eps{eps}_Results", Results[0, :, :kk+1].T, fmt="%.3f")

Final_results = np.zeros((1, 2, 2))
Final_results[0][0][0] = Results[0][0].sum() / args.n_exp
Final_results[0][0][1] = Results[0][0].std() / np.sqrt(args.n_exp)
Final_results[0][1][0] = Results[0][1].sum() / args.n_exp
Final_results[0][1][1] = Results[0][1].std() / np.sqrt(args.n_exp)

np.savetxt("../../Results/power_epsn/" + args.name + f"_eps{eps}_Final", Final_results.reshape(1, -1), fmt="%.3f")

print(f"\n=== FINAL RESULTS ===")
print(f"Dataset: {args.name}, eps={eps}, eps_gap={args.eps_gap}")
print(f"NAMMD: {Final_results[0][0][0]:.3f} +/- {Final_results[0][0][1]:.3f}")
print(f"MMD:   {Final_results[0][1][0]:.3f} +/- {Final_results[0][1][1]:.3f}")
