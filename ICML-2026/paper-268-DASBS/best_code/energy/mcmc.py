import os
import argparse
import numpy as np
from ising import ising2d_mh, ising2d_wolff, ising2d_swendsen_wang
from potts import potts2d_mh, potts2d_glauber, potts2d_swendsen_wang

parser = argparse.ArgumentParser()

parser.add_argument('--dist', type=str, default='ising', choices=['ising', 'potts'], help='Target distribution')
parser.add_argument('--L', type=int, default=16, help='Lattice size (L x L)')
parser.add_argument('--beta', type=float, default=0.28, help='Inverse temperature')
parser.add_argument('--J', type=float, default=1.0, help='Interaction strength')
parser.add_argument('--h', type=float, default=0.0, help='External magnetic field')
parser.add_argument('--q', type=int, default=3, help='Number of states for Potts model')

parser.add_argument('--method', type=str, default='wolff', choices=['mh', 'wolff', 'sw'], help='MCMC method')

parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_collect', type=int, default=32, help='Rounds of collecting samples')
parser.add_argument('--burn_in', type=int, default=2**16, help='Number of burn-in steps')
parser.add_argument('--collect_every', type=int, default=128, help='Collect a sample every x iterations')
args = parser.parse_args()

os.makedirs('exp_local', exist_ok=True)

if args.dist == 'potts':
    sampler = {'mh': potts2d_mh, 'sw': potts2d_swendsen_wang, 'glauber': potts2d_glauber}.get(args.method)
    if sampler is None:
        raise ValueError(f"Unknown method {args.method} for Potts model.")
    samples = sampler(L=args.L, beta=args.beta, J=args.J, h=args.h, q=args.q,
                      batch_size=args.batch_size, num_collect=args.num_collect,
                      burn_in=args.burn_in, collect_every=args.collect_every).astype(np.int8)
    np.save(f'exp_local/{args.dist}_{args.method}_L{args.L}_q{args.q}_beta{args.beta}_J{args.J}_h{args.h}.npy', samples)


elif args.dist == 'ising':
    sampler = {'mh': ising2d_mh, 'wolff': ising2d_wolff, 'sw': ising2d_swendsen_wang}.get(args.method)
    if sampler is None:
        raise ValueError(f"Unknown method {args.method} for Ising model.")
    samples = sampler(L=args.L, beta=args.beta, J=args.J, h=args.h,
                      batch_size=args.batch_size, num_collect=args.num_collect,
                      burn_in=args.burn_in, collect_every=args.collect_every).astype(np.int8)
    np.save(f'exp_local/{args.dist}_{args.method}_L{args.L}_beta{args.beta}_J{args.J}_h{args.h}.npy', samples)

else:
    raise ValueError(f"Unknown distribution {args.dist}.")
