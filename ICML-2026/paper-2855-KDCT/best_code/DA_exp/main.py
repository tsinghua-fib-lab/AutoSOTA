import torch
import argparse
parser = argparse.ArgumentParser()
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models
from torchvision import transforms
import torch.nn as nn
import os
import torch.nn.functional as F
import json
import numpy as np
from utils import *
import time
start_time = time.time()
# parameters of experimental setting
parser.add_argument('--N1',    default=150,    help = 'Size of each sample')
parser.add_argument('--TN1',    default=2000,    help = 'Size of training sample')
parser.add_argument('--rs',    default=383,    help = 'Random seed')

parser.add_argument('--N1_test',    default=150,    help = 'Size of each sample in testing')
parser.add_argument('--rs_test',    default=183,    help = 'Random seed in testing')

# parameters of experimental setting
parser.add_argument('--n_exp',  default=10,              help='Number of experiment runs')
parser.add_argument('--n_test', default=100,             help='Number of two-sample test runs')
parser.add_argument('--alpha',  default=0.05,            help='Confidence level of two-sample test')
parser.add_argument('--K',       default=1,            help='the upperbound of kernel')
parser.add_argument('--device', default=torch.device("cuda"),  help='Device of data')
parser.add_argument('--dtype',  default=torch.float,          help='Dtype of data')

parser.add_argument('--indices',  default=[0,1,2,3,4,5,6,7,8,9],           help='indices')
parser.add_argument('--num',  default=25,           help='num of classes')

# parameters of MMD
parser.add_argument('--ne_MMD', default=100,   help='Number of MMD optimization epochs')
parser.add_argument('--bs_MMD', default=2000,    help='Batch size of MMD in optimization')
parser.add_argument('--lr_MMD', default=0.1, help='Learning rate of MMD in optimization')

# parameters of NAMMD
parser.add_argument('--ne_NAMMD', default=2000,   help='Number of NAMMD optimization epochs')
parser.add_argument('--bs_NAMMD', default=200,    help='Batch size of NAMMD in optimization')
parser.add_argument('--lr_NAMMD', default=0.1, help='Learning rate of NAMMD in optimization')
parser.add_argument('--b_NAMMD', default=0.1,     help='Balance parameter of MMD and Reg terms in ptimization')

args = parser.parse_args()

set_random_seed(args.rs)

resnet50 = models.resnet50(pretrained=True).cuda()

with open('margin_imagenet.json', 'r') as f:
    margin_imagenet = json.load(f)
margin_imagenet = torch.tensor(margin_imagenet)/5
with open('margin_imagenetv2.json', 'r') as f:
    margin_imagenetv2 = json.load(f)
margin_imagenetv2 = torch.tensor(margin_imagenetv2)
margin = torch.abs(margin_imagenet-margin_imagenetv2)
sorted_margin, sorted_indices = torch.sort(margin)

for ee in range(len(args.indices)):
    index = args.indices[ee]
    print(np.mean(np.array(sorted_margin[index:index+args.num])))
    
imagenet_Fea = torch.load('imagenet_Fea.pt')
imagenetv2_Fea = torch.load('imagenetv2_Fea.pt')

indX = np.random.choice(len(imagenet_Fea), args.TN1, replace=False)
indY = np.random.choice(len(imagenetv2_Fea), args.TN1)
S = torch.cat((imagenet_Fea[indX],imagenetv2_Fea[indY]))
sigma_mmd = MMD_fit(S, args.TN1, args.lr_MMD, args.ne_MMD, args.bs_MMD, args.device, seed=args.rs)
sigma_nammd = NAMMD_fit(S, args.TN1, sigma_mmd, args.lr_NAMMD, args.ne_NAMMD, args.bs_NAMMD, args.b_NAMMD, args.device, seed=args.rs)
sigma0 = (sigma_mmd, sigma_nammd)
print("Selected bandwidths: MMD = {:.6g}, NAMMD = {:.6g}".format(sigma_mmd.item(), sigma_nammd.item()))
print('Training Done!')

Results = np.zeros((2, len(args.indices), args.n_exp))
Final_results = np.zeros((len(args.indices),2))
for ee in range(len(args.indices)):
    index = args.indices[ee]
    imagenet_Fea_part=torch.tensor([]).to(args.device)
    imagenetv2_Fea_part=torch.tensor([]).to(args.device)
    indices = sorted_indices[index:index+args.num]
    for idx in indices:
        imagenet_Fea_part=torch.cat([imagenet_Fea_part, imagenet_Fea[idx*50:(idx+1)*50]], dim=0)
        imagenetv2_Fea_part=torch.cat([imagenetv2_Fea_part, imagenetv2_Fea[idx*10:(idx+1)*10]], dim=0)
    for kk in range(args.n_exp):
        indX = np.random.choice(len(imagenet_Fea_part), args.N1, replace=False)
        indY = np.random.choice(len(imagenetv2_Fea_part), args.N1, replace=False)
        S = torch.cat((imagenet_Fea_part[indX],imagenetv2_Fea_part[indY]))
        TEMP = MMDu(S, args.N1, sigma_mmd)
        MMD = get_item(TEMP[0], args.device)
        TEMP = MMDu(S, args.N1, sigma_nammd)
        NAMMD = get_item(TEMP[0]/TEMP[2], args.device)
        Results[0, ee, kk] = MMD
        Results[1, ee, kk] = NAMMD
    Final_results[ee][0] = Results[0][ee].sum()/args.n_exp
    Final_results[ee][1] = Results[1][ee].sum()/args.n_exp
    print("N1 = ", str(args.N1), ", values of ",  str(ee))
    print("MMD: {:.10f}".format(Final_results[ee][0]))
    print("NAMMD: {:.10f}".format(Final_results[ee][1]))
    np.savetxt('./'+str(args.N1)+'_'+str(args.indices), Final_results, fmt='%.10f')

Results_ = np.zeros((len(args.indices), 2, args.n_exp))
Final_results_ = np.zeros((Results_.shape[0],2, 2))
Results_P = np.zeros((len(args.indices), 2, args.n_exp))
Final_results_P = np.zeros((Results_P.shape[0],2, 2))
for ee in range(len(args.indices)):
    index = args.indices[ee]
    imagenet_Fea_part=torch.tensor([]).to(args.device)
    imagenetv2_Fea_part=torch.tensor([]).to(args.device)
    indices = sorted_indices[index:index+args.num]
    for idx in indices:
        imagenet_Fea_part=torch.cat([imagenet_Fea_part, imagenet_Fea[idx*50:(idx+1)*50]], dim=0)
        imagenetv2_Fea_part=torch.cat([imagenetv2_Fea_part, imagenetv2_Fea[idx*10:(idx+1)*10]], dim=0)
    for kk in range(args.n_exp):
        H_MMD, H_NAMMD, P_MMD, P_NAMMD  = testing(imagenet_Fea_part, imagenetv2_Fea_part, Final_results[3][0], Final_results[3][1], args.N1_test, kk+args.rs_test, sigma0, args.n_test, args.alpha, args.device)
        Results_[ee, 0, kk] = H_MMD.sum() / args.n_test
        Results_[ee, 1, kk] = H_NAMMD.sum() / args.n_test
        Results_P[ee, 0, kk] = P_MMD.sum() / args.n_test
        Results_P[ee, 1, kk] = P_NAMMD.sum() / args.n_test

    Final_results_[ee][0][0] = Results_[ee][0].sum()/args.n_exp
    Final_results_[ee][0][1] = Results_[ee][0].std()/np.sqrt(args.n_exp)
    Final_results_[ee][1][0] = Results_[ee][1].sum()/args.n_exp
    Final_results_[ee][1][1] = Results_[ee][1].std()/np.sqrt(args.n_exp)
    np.savetxt(str(args.N1_test)+'_'+str(args.rs_test), Final_results_.reshape(len(args.indices),-1), fmt='%.3f')
    print("test power:", " index = ", str(index), ", N1_test = ", str(args.N1_test), ", rs_test = ", str(args.rs_test))
    print("MMD: {:.3f}±{:.3f}".format(Final_results_[ee][0][0], Final_results_[ee][0][1]))
    print("NAMMD: {:.3f}±{:.3f}".format(Final_results_[ee][1][0], Final_results_[ee][1][1]))
    
    Final_results_P[ee][0][0] = Results_P[ee][0].sum()/args.n_exp
    Final_results_P[ee][0][1] = Results_P[ee][0].std()/np.sqrt(args.n_exp)
    Final_results_P[ee][1][0] = Results_P[ee][1].sum()/args.n_exp
    Final_results_P[ee][1][1] = Results_P[ee][1].std()/np.sqrt(args.n_exp)
    np.savetxt(str(args.N1_test)+'_'+str(args.rs_test)+'_P', Final_results_P.reshape(len(args.indices),-1), fmt='%.10f')
    print("p-value:", " index = ", str(index), ", N1_test = ", str(args.N1_test), ", rs_test = ", str(args.rs_test))
    print("MMD: {:.10f}±{:.10f}".format(Final_results_P[ee][0][0], Final_results_P[ee][0][1]))
    print("NAMMD: {:.10f}±{:.10f}".format(Final_results_P[ee][1][0], Final_results_P[ee][1][1]))

# End timing
end_time = time.time()
# Calculate execution time
execution_time = end_time - start_time
print(f"Program execution time: {execution_time} seconds", flush=True)
