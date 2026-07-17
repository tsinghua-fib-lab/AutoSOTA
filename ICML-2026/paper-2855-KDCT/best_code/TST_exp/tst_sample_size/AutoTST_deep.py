# check = 1 for test power check = 0 for Type-I error
import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
import sys
import os
sys.path.append(os.path.abspath('../..'))
from baseline_tests.C2ST import TST_C2ST_D

# parameters of experimental setting
parser.add_argument('--check', default=1,      help = '1 for test power; 0 for type-I error')
parser.add_argument('--rs',    default=283,      help = 'Random seed')
parser.add_argument('--n_exp',  default=10,                   help='Number of experiment runs')
parser.add_argument('--n_test', default=100,                   help='Number of two-sample test runs')
parser.add_argument('--n_per',  default=100,                   help='Number of permutation test runs')
parser.add_argument('--alpha',  default=0.05,                  help='Confidence level of two-sample test')
parser.add_argument('--device', default=torch.device("cuda"),  help='Device of data')
parser.add_argument('--dtype',  default=torch.float,           help='Dtype of data')
parser.add_argument('--channels',  default=1,    help='channels of data')
parser.add_argument('--img_size',  default=32,   help='img_size of data')
args = parser.parse_args()

NAME = 'MNIST'
N1_LIST = [80,20,30,40,50,60,70,80]
xout_C2ST=[100,100,100,100,100,100,100,100] #
ne_C2ST=[2000,2000,2000,2000,2000,2000,2000,2000]
bs_C2ST=[500,500,500,500,500,500,500,500]
lr_C2ST=[0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002]

Results = np.zeros((len(N1_LIST), args.n_exp))
Final_results = np.zeros((len(N1_LIST),2))
H_AutoTST = np.zeros(args.n_test)
for N1 in range(len(N1_LIST)):
    for kk in range(args.n_exp):                             
        H_AutoTST, _, _ = TST_C2ST_D(NAME, N1_LIST[N1], kk+args.rs, args.check, args.n_test, args.n_per, args.alpha, args.device, args.dtype, args.channels, args.img_size, xout_C2ST[N1], ne_C2ST[N1], bs_C2ST[N1], lr_C2ST[N1])
        print('AutoTST Done!')

        Results[N1, kk] = H_AutoTST.sum() / args.n_test
        if args.check == 1:
            np.savetxt('../../Results/tst_sample_size/test_power/'+NAME+'_Results_AutoTST', Results, fmt='%.3f')
        else:
            np.savetxt('../../Results/tst_sample_size/typeI_error/'+NAME+'_Results_AutoTST', Results, fmt='%.3f')


    Final_results[N1][0] = Results[N1].sum()/args.n_exp
    Final_results[N1][1] = Results[N1].std()/np.sqrt(args.n_exp)

    if args.check == 1:
        np.savetxt('../../Results/tst_sample_size/test_power/'+NAME+'_AutoTST', Final_results, fmt='%.3f')
    else:
        np.savetxt('../../Results/tst_sample_size/typeI_error/'+NAME+'_AutoTST', Final_results, fmt='%.3f')

    if args.check == 1:
        print(NAME, ", N1 = ", str(N1_LIST[N1]), ", test power")
    else:
        print(NAME, ", N1 = ", str(N1_LIST[N1]), ", type-I error")

    print("AutoTST: {:.3f}±{:.3f}".format(Final_results[N1][0], Final_results[N1][1]))