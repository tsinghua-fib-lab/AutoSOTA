# check = 1 for test power check = 0 for Type-I error
import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
import sys
import os
sys.path.append(os.path.abspath('../..'))
from baseline_tests.ACTT import TST_ACTT

# parameters of experimental setting
parser.add_argument('--check', default=1,      help = '1 for test power; 0 for type-I error')
parser.add_argument('--rs',    default=283,      help = 'Random seed')
parser.add_argument('--n_exp',  default=10,                   help='Number of experiment runs')
parser.add_argument('--n_test', default=100,                   help='Number of two-sample test runs')
parser.add_argument('--n_per',  default=100,                   help='Number of permutation test runs')
parser.add_argument('--alpha',  default=0.05,                  help='Confidence level of two-sample test')
parser.add_argument('--device', default=torch.device("cpu"),  help='Device of data')
parser.add_argument('--dtype',  default=torch.float,           help='Dtype of data')
args = parser.parse_args()


# NAME = 'BLOB'
# N1_LIST = [50,100,150,200,250,300,350,400]

# NAME = 'HIGGS'
# # # N1_LIST = [500,1000,1500,2000,2500,3000,3500,4000]
# N1_LIST = [40]

# NAME = 'HDGM'
# N1_LIST = [500,1000,1500,2000,2500,3000,3500,4000]

NAME = 'MNIST'
N1_LIST = [40,50,60,70,80]

Results = np.zeros((len(N1_LIST), args.n_exp))
Final_results = np.zeros((len(N1_LIST),2))
H_ACTT = np.zeros(args.n_test)
for N1 in range(len(N1_LIST)):
    for kk in range(args.n_exp):                             
        H_ACTT, _, _ = TST_ACTT(NAME, N1_LIST[N1], kk+args.rs, args.check, args.n_test, args.alpha)
        print('ACTT Done!')

        Results[N1, kk] = H_ACTT.sum() / args.n_test
        if args.check == 1:
            np.savetxt('../../Results/tst_sample_size/test_power/'+NAME+'_Results_ACTT', Results, fmt='%.3f')
        else:
            np.savetxt('../../Results/tst_sample_size/typeI_error/'+NAME+'_Results_ACTT', Results, fmt='%.3f')


    Final_results[N1][0] = Results[N1].sum()/args.n_exp
    Final_results[N1][1] = Results[N1].std()/np.sqrt(args.n_exp)

    if args.check == 1:
        np.savetxt('../../Results/tst_sample_size/test_power/'+NAME+'_ACTT', Final_results, fmt='%.3f')
    else:
        np.savetxt('../../Results/tst_sample_size/typeI_error/'+NAME+'_ACTT', Final_results, fmt='%.3f')

    if args.check == 1:
        print(NAME, ", N1 = ", str(N1_LIST[N1]), ", test power")
    else:
        print(NAME, ", N1 = ", str(N1_LIST[N1]), ", type-I error")

    print("ACTT: {:.3f}±{:.3f}".format(Final_results[N1][0], Final_results[N1][1]))