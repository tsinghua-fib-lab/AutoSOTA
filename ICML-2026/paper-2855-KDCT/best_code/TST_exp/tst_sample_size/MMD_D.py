# check = 1 for test power check = 0 for Type-I error
import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
import sys
import os
sys.path.append(os.path.abspath('../..'))
from baseline_tests.MMD_D import TST_MMD_D

# parameters of experimental setting
parser.add_argument('--check', default=1,      help = '1 for test power; 0 for type-I error')
parser.add_argument('--rs',    default=283,      help = 'Random seed')
parser.add_argument('--n_exp',  default=10,                   help='Number of experiment runs')
parser.add_argument('--n_test', default=100,                   help='Number of two-sample test runs')
parser.add_argument('--n_per',  default=100,                   help='Number of permutation test runs')
parser.add_argument('--alpha',  default=0.05,                  help='Confidence level of two-sample test')
parser.add_argument('--device', default=torch.device("cuda"),  help='Device of data')
parser.add_argument('--dtype',  default=torch.float,           help='Dtype of data')
args = parser.parse_args()


# NAME = 'BLOB'
# N1_LIST = [50,100,150,200,250,300,350,400]
# xin_MMD_D=[2,2,2,2,2,2,2,2]
# H_MMD =[50,50,50,50,50,50,50,50]
# xout_MMD_D=[50,50,50,50,50,50,50,50]
# ne_MMD_D=[1000,1000,1000,1000,1000,1000,1000,1000]
# bs_MMD_D=[128,128,128,128,128,128,128,128]
# lr_MMD_D=[0.0005,0.0005,0.0005,0.0005,0.0005,0.0005,0.0005,0.0005]

NAME = 'HIGGS'
N1_LIST = [500,1000,1500,2000,2500,3000,3500,4000]
xin_MMD_D=[4,4,4,4,4,4,4,4]
H_MMD =[30,30,30,30,30,30,30,30]
xout_MMD_D=[30,30,30,30,30,30,30,30]
ne_MMD_D=[2000,2000,2000,2000,2000,2000,2000,2000]
bs_MMD_D=[500,500,500,500,500,500,500,500]
lr_MMD_D=[0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]

# NAME = 'HDGM'
# N1_LIST = [500,1000,1500,2000,2500,3000,3500,4000]
# xin_MMD_D=[10,10,10,10,10,10,10,10]
# H_MMD =[20,20,20,20,20,20,20,20]
# xout_MMD_D=[20,20,20,20,20,20,20,20]
# ne_MMD_D=[2000,2000,2000,2000,2000,2000,2000,2000]
# bs_MMD_D=[500,500,500,500,500,500,500,500]
# lr_MMD_D=[0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]

# NAME = 'MNIST'
# N1_LIST = [10,20,30,40,50,60,70,80]
# xin_MMD_D=[4,4,4,4,4,4,4,4] #
# H_MMD =[20,20,20,20,20,20,20,20] #
# xout_MMD_D=[20,20,20,20,20,20,20,20] #
# ne_MMD_D=[1000,1000,1000,1000,1000,1000,1000,1000]
# bs_MMD_D=[500,500,500,500,500,500,500,500]
# lr_MMD_D=[0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001]

Results = np.zeros((len(N1_LIST), args.n_exp))
Final_results = np.zeros((len(N1_LIST),2))
H_MMD_D = np.zeros(args.n_test)
for N1 in range(len(N1_LIST)):
    if N1<=5:
        continue
    for kk in range(args.n_exp):                             
        H_MMD_D, _, _ = TST_MMD_D(NAME, N1_LIST[N1], kk+args.rs, args.check, args.n_test, args.n_per, args.alpha, args.device, args.dtype, xin_MMD_D[N1], H_MMD[N1], xout_MMD_D[N1], ne_MMD_D[N1], bs_MMD_D[N1], lr_MMD_D[N1])
        print('MMD-D Done!')

        Results[N1, kk] = H_MMD_D.sum() / args.n_test
        if args.check == 1:
            np.savetxt('../../Results/tst_sample_size/test_power/'+NAME+'_Results_MMD_D', Results, fmt='%.3f')
        else:
            np.savetxt('../../Results/tst_sample_size/typeI_error/'+NAME+'_Results_MMD_D', Results, fmt='%.3f')


    Final_results[N1][0] = Results[N1].sum()/args.n_exp
    Final_results[N1][1] = Results[N1].std()/np.sqrt(args.n_exp)

    if args.check == 1:
        np.savetxt('../../Results/tst_sample_size/test_power/'+NAME+'_MMD_D', Final_results, fmt='%.3f')
    else:
        np.savetxt('../../Results/tst_sample_size/typeI_error/'+NAME+'_MMD_D', Final_results, fmt='%.3f')

    if args.check == 1:
        print(NAME, ", N1 = ", str(N1_LIST[N1]), ", test power")
    else:
        print(NAME, ", N1 = ", str(N1_LIST[N1]), ", type-I error")

    print("MMD_D: {:.3f}±{:.3f}".format(Final_results[N1][0], Final_results[N1][1]))