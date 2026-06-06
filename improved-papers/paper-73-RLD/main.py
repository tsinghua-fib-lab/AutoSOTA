import torch
import numpy as np
import argparse
import os
import random
from solvers import *
from pathlib import Path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description="RLD")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--train_samples", type=int, default=-1)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--valid_samples", type=int, default=-1)   
    parser.add_argument("--test_path", type=str, default='./')
    parser.add_argument("--test_samples", type=int, default=-1)  
    parser.add_argument("--problem", type=str, default='mis', choices=['mis', 'mcl', 'mcut'])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--method", type=str, default='RLSA')
    parser.add_argument("--loss", type=str, default='erdoes', choices=['erdoes', 'reinforce'])
    parser.add_argument("--save_dir", type=str, default='./')
    parser.add_argument("--log_file", type=str, default='log.txt')
    parser.add_argument("--num_t", type=int, default=1000, help='steps for sampling')
    parser.add_argument("--num_k", type=int, default=100, help='number of particles')
    parser.add_argument("--num_tp", type=int, default=100, help='number of particles')
    parser.add_argument("--num_kp", type=int, default=10, help='number of particles')
    parser.add_argument("--num_l", type=int, default=1000, help='number of layers')
    parser.add_argument("--num_h", type=int, default=100, help='hidden dimension')
    parser.add_argument("--num_d", type=int, default=30, help='step size')
    parser.add_argument("--beta", type=float, default=1.02, help='penalty coefficient')
    parser.add_argument("--tau0", type=float, default=0.05, help='initial temperature')
    parser.add_argument("--lambd", type=float, default=0.5, help='regularization coefficient')
    parser.add_argument("--mixed_precision", type=str, default='no', choices=['no', 'bf16', 'fp16'])
    parser.add_argument('--do_train', action='store_true', help='whether do training')
    parser.add_argument('--skip_decode', action='store_true', help='whether skip decoding')
    
    args = parser.parse_args()
    # fix seeds
    set_seed(args.seed)
    if args.method == 'RLSA':
        solver = RLSA(args)
    elif args.method == 'RLPGSA':
        solver = RLPGSA(args)
    elif args.method == 'RLNN':
        solver = RLNN(args)
    else:
        raise NotImplementedError

    solver.prepare()

    if args.do_train:
        os.makedirs(args.save_dir, exist_ok = True)
        solver.train(args.save_dir)
    else:
        os.makedirs(Path(args.log_file).parent, exist_ok = True)
        solver.evaluate('test', log_path=args.log_file)
    
if __name__=='__main__':
    main()