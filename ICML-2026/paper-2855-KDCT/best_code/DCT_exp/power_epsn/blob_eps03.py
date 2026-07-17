import numpy as np
import torch
import argparse
import sys
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../.."))
from utils import training, testing, construct_distributions

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="BLOB")
parser.add_argument("--N1", default=500, type=int, help="Size of sample in construction")
parser.add_argument("--N", default=3000, type=int, help="Size of sample in testing")
parser.add_argument("--rs", default=483, type=int, help="Random seed")
parser.add_argument("--n_exp", default=10, type=int, help="Number of experiment runs")
parser.add_argument("--n_test", default=100, type=int, help="Number of two-sample test runs")
parser.add_argument("--n_per", default=100, type=int, help="Number of permutation test runs")
parser.add_argument("--alpha", default=0.05, type=float, help="Confidence level")
parser.add_argument("--K", default=1, type=int, help="Upper bound of kernel")
parser.add_argument("--eps", default=0.3, type=float, help="Epsilon value")
parser.add_argument("--eps_gap", default=0.01, type=float, help="Epsilon gap")
parser.add_argument("--sigma0", default=1, type=float, help="Parameter for Gaussian kernel")
parser.add_argument("--lr_data", default=0.01, type=float, help="Learning rate for data construction")
parser.add_argument("--device", default=torch.device("cuda"), help="Device")
parser.add_argument("--dtype", default=torch.float, help="Dtype")
parser.add_argument("--ne_MMD", default=1000, type=int, help="MMD optimization epochs")
parser.add_argument("--bs_MMD", default=400, type=int, help="MMD batch size")
parser.add_argument("--lr_MMD", default=0.001, type=float, help="MMD learning rate")
parser.add_argument("--ne_NAMMD", default=1000, type=int, help="NAMMD optimization epochs")
parser.add_argument("--bs_NAMMD", default=400, type=int, help="NAMMD batch size")
parser.add_argument("--lr_NAMMD", default=0.001, type=float, help="NAMMD learning rate")
parser.add_argument("--b_NAMMD", default=0.2, type=float, help="Balance parameter")

args = parser.parse_args()

Results = np.zeros((1, 2, args.n_exp))

eps = args.eps
N = args.N
rs = args.rs
N1 = args.N1

for kk in range(args.n_exp):
    sigma0 = training(args.name, N1, kk+rs, 1, args.ne_MMD, args.bs_MMD, args.lr_MMD, args.ne_NAMMD, args.bs_NAMMD, args.lr_NAMMD, args.b_NAMMD, args.device, args.dtype)
    args.sigma0 = sigma0
    print(f"Training Done for run {kk+1}/{args.n_exp}!")

    H_NAMMD = np.zeros(args.n_test)
    H_MMD = np.zeros(args.n_test)

    X1, Y1, MMD1, NAMMD1, X2, Y2, MMD2, NAMMD2 = construct_distributions(args.name, N1, rs + kk, eps, args.eps_gap, args.lr_data, args.sigma0, args.K, args.device, args.dtype)
    print(f"Construction Done for run {kk+1}/{args.n_exp}!")

    H_MMD, H_NAMMD = testing(X2, Y2, MMD1, NAMMD1, N, kk+rs+100, args.sigma0, args.n_test, args.n_per, args.alpha, args.device, args.dtype)
    print(f"Testing Done for run {kk+1}/{args.n_exp}!")

    Results[0, 0, kk] = H_NAMMD.sum() / args.n_test
    Results[0, 1, kk] = H_MMD.sum() / args.n_test

    np.savetxt("../../Results/power_epsn/" + args.name + f"_eps{eps}_Results", Results.reshape(1, -1), fmt="%.3f")

    print(f"Run {kk+1}: NAMMD={Results[0, 0, kk]:.3f}, MMD={Results[0, 1, kk]:.3f}")

Final_results = np.zeros((1, 2, 2))
Final_results[0][0][0] = Results[0][0].sum() / args.n_exp
Final_results[0][0][1] = Results[0][0].std() / np.sqrt(args.n_exp)
Final_results[0][1][0] = Results[0][1].sum() / args.n_exp
Final_results[0][1][1] = Results[0][1].std() / np.sqrt(args.n_exp)

np.savetxt("../../Results/power_epsn/" + args.name + f"_eps{eps}_Final", Final_results.reshape(1, -1), fmt="%.3f")

print(f"\ntest power of {args.name}, N1={N1}, N={N}, eps={eps}, eps_gap={args.eps_gap}")
print(f"NAMMD: {Final_results[0][0][0]:.3f} +/- {Final_results[0][0][1]:.3f}")
print(f"MMD: {Final_results[0][1][0]:.3f} +/- {Final_results[0][1][1]:.3f}")
