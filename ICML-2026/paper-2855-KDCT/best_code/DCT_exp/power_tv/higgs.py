# check = 1 for test power check = 0 for Type-I error
import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
import sys
import os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
from dataloader import construct_distributions_tv
from TST_NAMMD import TV_testing
from util_Mahalanobis import training, testing

parser.add_argument('--name',  default='HIGGS', help = 'Dataset')
parser.add_argument('--N1',    default=50,    help = 'Size of sample in construction')
parser.add_argument('--N',     default=[1800,1000,1100,1100],    help = 'Size of sample in testing')
parser.add_argument('--check', default=1,      help = '1 for test power; 0 for type-I error')
parser.add_argument('--rs',    default=[283,283,15,15],    help = 'Random seed')
parser.add_argument('--way',   default=['uni', 'uni', 'uni', 'uni'],    help = 'way to construct distribution with total variation')
parser.add_argument('--num',   default=2,    help = 'the number of elements mainly considered in distribution construction with total variation')

parser.add_argument('--n_exp',  default=10,              help='Number of experiment runs')
parser.add_argument('--n_test', default=100,             help='Number of two-sample test runs')
parser.add_argument('--n_per',  default=100,             help='Number of permutation test runs')
parser.add_argument('--alpha',  default=0.05,            help='Confidence level of two-sample test')
parser.add_argument('--K',       default=1,            help='the upperbound of kernel')

parser.add_argument('--delts',  default=[0.1,0.3,0.5,0.7],           help='deltas')
parser.add_argument('--delt_gap',  default=0.2,           help='delta gap')
parser.add_argument('--sigma0',     default=1,           help='parameter for Gaussian kernel')

parser.add_argument('--device', default=torch.device("cuda"),  help='Device of data')
parser.add_argument('--dtype',  default=torch.float,          help='Dtype of data')



# parameters of MMD
parser.add_argument('--ne_MMD', default=[2000,2000,2000,2000],   help='Number of MMD optimization epochs')
parser.add_argument('--bs_MMD', default=[128,128,128,128],    help='Batch size of MMD in optimization')
parser.add_argument('--lr_MMD', default=[0.001,0.001,0.003,0.003], help='Learning rate of MMD in optimization')

# parameters of NAMMD
parser.add_argument('--ne_NAMMD', default=[0,0,0,0],   help='Number of NAMMD optimization epochs')
parser.add_argument('--bs_NAMMD', default=[128,128,128,128],    help='Batch size of NAMMD in optimization')
parser.add_argument('--lr_NAMMD', default=[0.001,0.001,0.001,0.001], help='Learning rate of NAMMD in optimization')
parser.add_argument('--b_NAMMD', default=[0.002,0.002,0.002,0.002],     help='Balance parameter of MMD and Reg terms in ptimization')

args = parser.parse_args()

Results = np.zeros((len(args.delts), 3, args.n_exp))
Final_results = np.zeros((Results.shape[0],3, 2))

for dd in range(len(args.delts)):

    if dd<=1:
        continue

    way = args.way[dd]
    delt = args.delts[dd]
    N = args.N[dd]
    rs = args.rs[dd]

    ne_MMD = args.ne_MMD[dd]
    bs_MMD = args.bs_MMD[dd]
    lr_MMD = args.lr_MMD[dd]

    ne_NAMMD = args.ne_NAMMD[dd]
    bs_NAMMD = args.bs_NAMMD[dd]
    lr_NAMMD = args.lr_NAMMD[dd]
    b_NAMMD = args.b_NAMMD[dd]

    Z = None
    P1 = None
    P2 = None
    MMD1 = None
    Reg1 = None
    MMD2 = None
    Reg2 = None

    for kk in range(args.n_exp):
        H_NAMMD = np.zeros(args.n_test)
        H_MMD = np.zeros(args.n_test)
        H_TV = np.zeros(args.n_test)

        if args.check == 1:    
            Z, P1, P2, _, _, _, _ = construct_distributions_tv(args.name, args.N1, rs + kk, delt, delt+ args.delt_gap, args.sigma0, args.K, args.device, args.dtype, way, args.num)
        else:
            Z, P1, P2, _, _, _, _ = construct_distributions_tv(args.name, args.N1, rs + kk, delt, delt, args.sigma0, args.K, args.device, args.dtype, way, args.num)

        sigma0, M_matrix = training(Z, P1, P2, N, kk+rs, ne_MMD, bs_MMD, lr_MMD, ne_NAMMD, bs_NAMMD, lr_NAMMD, b_NAMMD, args.device, args.dtype)
        print('Training Done!')

        H_MMD, H_NAMMD,  = testing(Z, P1, P2, N, kk+rs+100, sigma0, M_matrix, args.n_test, args.n_per, args.alpha, args.device, args.dtype)

        H_TV = TV_testing(Z, P1, P2, N, kk+rs+100, args.sigma0, args.n_test, args.n_per, args.alpha, args.device, args.dtype)
        print('Testing Done!')

        Results[dd, 0, kk] = H_NAMMD.sum() / args.n_test
        Results[dd, 1, kk] = H_MMD.sum() / args.n_test
        Results[dd, 2, kk] = H_TV.sum() / args.n_test

        if args.check == 1:
            np.savetxt('../../Results/power_tv/test_power/'+args.name+'_Results_'+str(args.N1)+'_'+str(args.N)+'_'+str(args.delt_gap), Results.reshape(len(args.delts),-1), fmt='%.3f')
        else:
            np.savetxt('../../Results/power_tv/typeI_error/'+args.name+'_Results_'+str(args.N1)+'_'+str(args.N)+'_'+str(args.delt_gap), Results.reshape(len(args.delts),-1), fmt='%.3f')

        Final_results[dd][0][0] = Results[dd][0].sum()/args.n_exp
        Final_results[dd][0][1] = Results[dd][0].std()/np.sqrt(args.n_exp)
        Final_results[dd][1][0] = Results[dd][1].sum()/args.n_exp
        Final_results[dd][1][1] = Results[dd][1].std()/np.sqrt(args.n_exp)
        Final_results[dd][2][0] = Results[dd][2].sum()/args.n_exp
        Final_results[dd][2][1] = Results[dd][2].std()/np.sqrt(args.n_exp)

    if args.check == 1:
        np.savetxt('../../Results/power_tv/test_power/'+args.name+'_'+str(args.N1)+'_'+str(args.N)+'_'+str(args.delt_gap), Final_results.reshape(len(args.delts),-1), fmt='%.3f')
    else:
        np.savetxt('../../Results/power_tv/typeI_error/'+args.name+'_'+str(args.N1)+'_'+str(args.N)+'_'+str(args.delt_gap), Final_results.reshape(len(args.delts),-1), fmt='%.3f')

    if args.check == 1:
        print("test power of ", args.name, ", N1 = ", str(args.N1), ", N = ", str(args.N), ", delt = ", str(delt), ", delt_gap = ", str(args.delt_gap))
    else:
        print("type-I error of ", args.name, ", N1 = ", str(args.N1), ", N = ", str(args.N), ", delt = ", str(delt), ", delt_gap = ", str(args.delt_gap))

    print("NAMMD: {:.3f}±{:.3f}".format(Final_results[dd][0][0], Final_results[dd][0][1]))
    print("MMD: {:.3f}±{:.3f}".format(Final_results[dd][1][0], Final_results[dd][1][1]))
    print("TV: {:.3f}±{:.3f}".format(Final_results[dd][2][0], Final_results[dd][2][1]))