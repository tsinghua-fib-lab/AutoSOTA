# check = 1 for test power check = 0 for Type-I error
import numpy as np
import torch
from sklearn.utils import check_random_state
import argparse
parser = argparse.ArgumentParser()
import sys
import os
sys.path.append(os.path.abspath('..'))

from util_Deep import training, testing

# parameters to generate data
parser.add_argument('--name',  default='HDGM', help = 'Dataset')
parser.add_argument('--N1',    default=4000,    help = 'Size of each sample')
parser.add_argument('--check', default=1,      help = '1 for test power; 0 for type-I error')
parser.add_argument('--rs',    default=483,    help = 'Random seed')

# parameters of experimental setting
parser.add_argument('--n_exp',  default=10,              help='Number of experiment runs')
parser.add_argument('--n_test', default=100,             help='Number of two-sample test runs')
parser.add_argument('--n_per',  default=100,             help='Number of permutation test runs')
parser.add_argument('--alpha',  default=0.05,            help='Confidence level of two-sample test')
parser.add_argument('--device', default=torch.device("cpu"),  help='Device of data')
parser.add_argument('--dtype',  default=torch.float,          help='Dtype of data')

# parameters of kernel
parser.add_argument('--sigma',   default=np.array(2),    help='Parameter of Gaussian kernel')
parser.add_argument('--sigma0',  default=np.array(0.5),  help='Parameter of deep kernel')
parser.add_argument('--epsilon', default=np.array(0.5),   help='Parameter of smooth')
parser.add_argument('--H',       default=50,          help='Dimension of hidden layer')
parser.add_argument('--x_out',   default=50,      help='Dimension of output layer')

# parameters of MMD
parser.add_argument('--ne_MMD', default=1000,   help='Number of MMD optimization epochs')
parser.add_argument('--bs_MMD', default=400,    help='Batch size of MMD in optimization')
parser.add_argument('--lr_MMD', default=0.05, help='Learning rate of MMD in optimization')

# parameters of NAMMD
parser.add_argument('--ne_NAMMD', default=1000,   help='Number of NAMMD optimization epochs')
parser.add_argument('--bs_NAMMD', default=400,    help='Batch size of NAMMD in optimization')
parser.add_argument('--lr_NAMMD', default=0.05,   help='Learning rate of NAMMD in optimization')
parser.add_argument('--b_NAMMD', default=0.0001,     help='Balance parameter of MMD and Reg terms in ptimization')

args = parser.parse_args()

Results = np.zeros((2, args.n_exp))

H_MMD = np.zeros(args.n_test)
H_NAMMD = np.zeros(args.n_test)

for kk in range(args.n_exp):     
                                                      
    sigma, sigma0, epsilon, model = training(args.name, args.N1, kk+args.rs, args.check, args.sigma, args.sigma0, args.epsilon, args.H, args.x_out, args.ne_MMD, args.bs_MMD, args.lr_MMD, args.ne_NAMMD, args.bs_NAMMD, args.lr_NAMMD, args.b_NAMMD, args.device, args.dtype)
    print('Training Done!')

    H_MMD, H_NAMMD,  = testing(args.name, args.N1, kk+args.rs, args.check, sigma, sigma0, epsilon, model, args.n_test, args.n_per, args.alpha, args.device, args.dtype)
    print('Testing Done!')
    
    Results[0, kk] = H_MMD.sum() / args.n_test
    Results[1, kk] = H_NAMMD.sum() / args.n_test

    if args.check == 1:
        np.savetxt('../../Results/same_kernel/test_power/Deep/'+args.name+'_Results_'+str(args.N1)+'_'+str(args.n_exp), Results, fmt='%.3f')
    else:
        np.savetxt('../../Results/same_kernel/typeI_error/Deep/'+args.name+'_Results_'+str(args.N1)+'_'+str(args.n_exp), Results, fmt='%.3f')
    
Final_results = np.zeros((Results.shape[0],2))

for i in range(Results.shape[0]):
    Final_results[i][0] = Results[i].sum()/args.n_exp
    Final_results[i][1] = Results[i].std()/np.sqrt(args.n_exp)

if args.check == 1:
    np.savetxt('../../Results/same_kernel/test_power/Deep/'+args.name+'_'+str(args.N1)+'_'+str(args.n_exp), Final_results, fmt='%.3f')
else:
    np.savetxt('../../Results/same_kernel/typeI_error/Deep/'+args.name+'_'+str(args.N1)+'_'+str(args.n_exp), Final_results, fmt='%.3f')

if args.check == 1:
    print(args.name, ", N1 = ", str(args.N1), ", test power of ",  str(args.n_exp), " experiment runs")
else:
    print(args.name, ", N1 = ", str(args.N1), ", type-I error of ",  str(args.n_exp), " experiment runs")
    
print("MMD: {:.3f}±{:.3f}".format(Results[0].sum()/args.n_exp, Results[0].std()/np.sqrt(args.n_exp)))
print("NAMMD: {:.3f}±{:.3f}".format(Results[1].sum()/args.n_exp, Results[1].std()/np.sqrt(args.n_exp)))