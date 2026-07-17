# check = 1 for test power check = 0 for Type-I error
import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
import sys
import os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
from utils import training, testing, construct_distributions

parser.add_argument('--name',  default='MNIST', help = 'Dataset')
parser.add_argument('--N1s',    default=[500,500,500,500],    help = 'Size of sample in construction')
parser.add_argument('--N',     default=[100,300,400,400],    help = 'Size of sample in testing')
parser.add_argument('--rs',    default=[183,483,283,283],    help = 'Random seed')

parser.add_argument('--n_exp',  default=10,              help='Number of experiment runs')
parser.add_argument('--n_test', default=100,             help='Number of two-sample test runs')
parser.add_argument('--n_per',  default=100,             help='Number of permutation test runs')
parser.add_argument('--alpha',  default=0.05,            help='Confidence level of two-sample test')
parser.add_argument('--K',       default=1,            help='the upperbound of kernel')

parser.add_argument('--epss',  default=[0.1,0.3,0.5,0.7],           help='epsas')
parser.add_argument('--eps_gap',  default=0.01,           help='epsa gap')
parser.add_argument('--sigma0',     default=1,           help='parameter for Gaussian kernel')
parser.add_argument('--lr_data',     default=0.01,           help='learning rate for data cosntruction')

parser.add_argument('--device', default=torch.device("cuda"),  help='Device of data')
parser.add_argument('--dtype',  default=torch.float,          help='Dtype of data')


# parameters of MMD
parser.add_argument('--ne_MMD', default=1000,   help='Number of MMD optimization epochs')
parser.add_argument('--bs_MMD', default=400,    help='Batch size of MMD in optimization')
parser.add_argument('--lr_MMD', default=0.001, help='Learning rate of MMD in optimization')

# parameters of NAMMD
parser.add_argument('--ne_NAMMD', default=1000,   help='Number of NAMMD optimization epochs')
parser.add_argument('--bs_NAMMD', default=400,    help='Batch size of NAMMD in optimization')
parser.add_argument('--lr_NAMMD', default=0.001, help='Learning rate of NAMMD in optimization')
parser.add_argument('--b_NAMMD', default=0.1,     help='Balance parameter of MMD and Reg terms in ptimization')

args = parser.parse_args()

Results = np.zeros((len(args.epss), 2, args.n_exp))
Final_results = np.zeros((Results.shape[0],2, 2))

for dd in range(len(args.epss)):

    eps = args.epss[dd]
    N = args.N[dd]
    rs = args.rs[dd]
    N1 = args.N1s[dd]

    Z = None
    P1 = None
    P2 = None
    MMD1 = None
    NAMMD1 = None
    MMD2 = None
    NAMMD2 = None

    for kk in range(args.n_exp):
        sigma0 = training(args.name, N1, kk+rs, 1, args.ne_MMD, args.bs_MMD, args.lr_MMD, args.ne_NAMMD, args.bs_NAMMD, args.lr_NAMMD, args.b_NAMMD, args.device, args.dtype)
        args.sigma0 = sigma0
        print('Training Done!')

        H_NAMMD = np.zeros(args.n_test)
        H_MMD = np.zeros(args.n_test)

        X1, Y1, MMD1, NAMMD1, X2, Y2, MMD2, NAMMD2 = construct_distributions(args.name, N1, rs + kk, eps, args.eps_gap, args.lr_data, args.sigma0, args.K, args.device, args.dtype)
        print('Construction Done!')

        H_MMD, H_NAMMD,  = testing(X2, Y2, MMD1, NAMMD1, N, kk+rs+100, args.sigma0, args.n_test, args.n_per, args.alpha, args.device, args.dtype)
        print('Testing Done!')

        Results[dd, 0, kk] = H_NAMMD.sum() / args.n_test
        Results[dd, 1, kk] = H_MMD.sum() / args.n_test

        np.savetxt('../../Results/power_epsn/'+args.name+'_Results_'+str(args.N1s)+'_'+str(args.N)+'_'+str(args.eps_gap), Results.reshape(len(args.epss),-1), fmt='%.3f')

    Final_results[dd][0][0] = Results[dd][0].sum()/args.n_exp
    Final_results[dd][0][1] = Results[dd][0].std()/np.sqrt(args.n_exp)
    Final_results[dd][1][0] = Results[dd][1].sum()/args.n_exp
    Final_results[dd][1][1] = Results[dd][1].std()/np.sqrt(args.n_exp)

    np.savetxt('../../Results/power_epsn/'+args.name+'_'+str(args.N1s)+'_'+str(args.N)+'_'+str(args.eps_gap), Final_results.reshape(len(args.epss),-1), fmt='%.3f')

    print("test power of ", args.name, ", N1 = ", str(args.N1s), ", N = ", str(args.N), ", eps = ", str(eps), ", eps_gap = ", str(args.eps_gap))

    print("NAMMD: {:.3f}±{:.3f}".format(Final_results[dd][0][0], Final_results[dd][0][1]))
    print("MMD: {:.3f}±{:.3f}".format(Final_results[dd][1][0], Final_results[dd][1][1]))
