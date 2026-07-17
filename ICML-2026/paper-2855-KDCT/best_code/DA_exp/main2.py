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
import pickle
import time
start_time = time.time()
# parameters of experimental setting
parser.add_argument('--TN1',    default=20000,    help = 'Size of training sample')
parser.add_argument('--rs',    default=683,    help = 'Random seed')
parser.add_argument('--rs_test',    default=783,    help = 'Random seed in testing')

# parameters of experimental setting
parser.add_argument('--n_exp',  default=10,              help='Number of experiment runs')
parser.add_argument('--n_test', default=100,             help='Number of two-sample test runs')
parser.add_argument('--N_per',  default=100,              help='Number of permutation test runs')
parser.add_argument('--alpha',  default=0.05,            help='Confidence level of two-sample test')
parser.add_argument('--K',       default=1,            help='the upperbound of kernel')
parser.add_argument('--device', default=torch.device("cuda"),  help='Device of data')
parser.add_argument('--dtype',  default=torch.float,          help='Dtype of data')

parser.add_argument('--sample_sizes',  default=[10,20,30,40,50,60,70,80,90,100],           help='sample_sizes')

# parameters of MMD
parser.add_argument('--ne_MMD', default=100,   help='Number of MMD optimization epochs')
parser.add_argument('--bs_MMD', default=2000,    help='Batch size of MMD in optimization')
parser.add_argument('--lr_MMD', default=0.1, help='Learning rate of MMD in optimization')

# parameters of NAMMD
parser.add_argument('--ne_NAMMD', default=2000,   help='Number of NAMMD optimization epochs')
parser.add_argument('--bs_NAMMD', default=200,    help='Batch size of NAMMD in optimization')
parser.add_argument('--lr_NAMMD', default=0.1, help='Learning rate of NAMMD in optimization')
parser.add_argument('--b_NAMMD', default=0.1,     help='Balance parameter of MMD and Reg terms in ptimization')

parser.add_argument('--adv_levels',  default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],           help='perturb_steps')

args = parser.parse_args()

set_random_seed(args.rs)

resnet50 = models.resnet50(pretrained=True).cuda()
resnet50.eval()

# with open('margin_imagenet.json', 'r') as f:
#     margin_imagenet = json.load(f)
# margin_imagenet = torch.tensor(margin_imagenet)/50
with open('acc_imagenet.json', 'r') as f:
    margin_imagenet = json.load(f)
margin_imagenet = torch.tensor(margin_imagenet)/50
imagenet_Fea = torch.load('imagenet_Fea.pt')
imagenetv2_Fea = torch.load('imagenetv2_Fea.pt')
indX = np.random.choice(len(imagenet_Fea), args.TN1//10, replace=False)
indY = np.random.choice(len(imagenetv2_Fea), args.TN1//10,)
S = torch.cat((imagenet_Fea[indX],imagenetv2_Fea[indY]))
sigma_mmd = MMD_fit(S, args.TN1//10, args.lr_MMD, args.ne_MMD, args.bs_MMD, args.device, seed=args.rs)
sigma_nammd = NAMMD_fit(S, args.TN1//10, sigma_mmd, args.lr_NAMMD, args.ne_NAMMD, args.bs_NAMMD, args.b_NAMMD, args.device, seed=args.rs)
sigma0 = (sigma_mmd, sigma_nammd)
print("Selected bandwidths: MMD = {:.6g}, NAMMD = {:.6g}".format(sigma_mmd.item(), sigma_nammd.item()))
print('Training Done!')
del imagenetv2_Fea

def reference_values(S, sample_size):
    temp_mmd = MMDu(S, sample_size, sigma_mmd)
    temp_nammd = MMDu(S, sample_size, sigma_nammd)
    mmd = get_item(temp_mmd[0], args.device)
    nammd = get_item(temp_nammd[0] / temp_nammd[2], args.device)
    return mmd, nammd

with torch.no_grad():
    # with open('margin_imagenetv2.json', 'r') as f:
    #     margin_imagenetv2 = json.load(f)
    # margin_imagenetv2 = torch.tensor(margin_imagenetv2)/10
    with open('acc_imagenetv2.json', 'r') as f:
        margin_imagenetv2 = json.load(f)
    margin_imagenetv2 = torch.tensor(margin_imagenetv2)/10
    # margin_imagenetv2 = torch.mean(margin_imagenetv2).item()
    margin_imagenetv2 = torch.mean(torch.abs(margin_imagenet-margin_imagenetv2)).item()
    # margin_imagenetv2 = torch.mean(margin_imagenetv2).item()
    imagenetv2_Fea = torch.load('imagenetv2_Fea.pt')
    indX = np.random.choice(len(imagenet_Fea), args.TN1, replace=False)
    indY = np.random.choice(len(imagenetv2_Fea), args.TN1)
    S = torch.cat((imagenet_Fea[indX],imagenetv2_Fea[indY]))
    imagenetv2_MMD, imagenetv2_NAMMD = reference_values(S, args.TN1)
    print("imagenetv2. MMD: ", str(imagenetv2_MMD), " NAMMD: ", str(imagenetv2_NAMMD), " Margin: ", str(margin_imagenetv2))
    del margin_imagenetv2, imagenetv2_Fea

    # with open('margin_imagenetr.json', 'r') as f:
    #     margin_imagenetr = json.load(f)
    with open('acc_imagenetr.json', 'r') as f:
        margin_imagenetr = json.load(f)
    with open('imagenetr_label_to_indices.pkl', 'rb') as f:
        imagenetr_label_to_indices = pickle.load(f)
    i = 0
    for key in list(imagenetr_label_to_indices.keys()):
        margin_imagenetr[i] = margin_imagenetr[i]/len(imagenetr_label_to_indices[key])
        i += 1
    margin_imagenetr = torch.tensor(margin_imagenetr)
    # margin_imagenetr = torch.mean(margin_imagenetr).item()
    margin_imagenetr = torch.mean(torch.abs(margin_imagenet[np.array(list(imagenetr_label_to_indices.keys()))]-margin_imagenetr)).item()
    # margin_imagenetr = torch.mean(margin_imagenetr).item()
    imagenetr_Fea = torch.load('imagenetr_Fea.pt')
    indX = np.random.choice(len(imagenet_Fea), args.TN1, replace=False)
    indY = np.random.choice(len(imagenetr_Fea), args.TN1, replace=False)
    S = torch.cat((imagenet_Fea[indX],imagenetr_Fea[indY]))
    imagenetr_MMD, imagenetr_NAMMD = reference_values(S, args.TN1)
    print("imagenetr. MMD: ", str(imagenetr_MMD), " NAMMD: ", str(imagenetr_NAMMD), " Margin: ", str(margin_imagenetr))
    del margin_imagenetr, imagenetr_Fea, imagenetr_label_to_indices

    # with open('margin_imagenetsk.json', 'r') as f:
    #     margin_imagenetsk = json.load(f)
    with open('acc_imagenetsk.json', 'r') as f:
        margin_imagenetsk = json.load(f)
    with open('imagenetsk_label_to_indices.pkl', 'rb') as f:
        imagenetsk_label_to_indices = pickle.load(f)
    i = 0
    for key in list(imagenetsk_label_to_indices.keys()):
        margin_imagenetsk[i] = margin_imagenetsk[i]/len(imagenetsk_label_to_indices[key])
        i += 1
    margin_imagenetsk = torch.tensor(margin_imagenetsk)
    # margin_imagenetsk = torch.mean(margin_imagenetsk).item()
    margin_imagenetsk = torch.mean(torch.abs(margin_imagenet[np.array(list(imagenetsk_label_to_indices.keys()))]-margin_imagenetsk)).item()
    # margin_imagenetsk = torch.mean(margin_imagenetsk).item()
    imagenetsk_Fea = torch.load('imagenetsk_Fea.pt')
    indX = np.random.choice(len(imagenet_Fea), args.TN1, replace=False)
    indY = np.random.choice(len(imagenetsk_Fea), args.TN1, replace=False)
    S = torch.cat((imagenet_Fea[indX],imagenetsk_Fea[indY]))
    imagenetsk_MMD, imagenetsk_NAMMD = reference_values(S, args.TN1)
    print("imagenetsk. MMD: ", str(imagenetsk_MMD), " NAMMD: ", str(imagenetsk_NAMMD), " Margin: ", str(margin_imagenetsk))
    del margin_imagenetsk, imagenetsk_Fea

    # with open('margin_imageneta.json', 'r') as f:
    #     margin_imageneta = json.load(f)
    with open('acc_imageneta.json', 'r') as f:
        margin_imageneta = json.load(f)
    with open('imageneta_label_to_indices.pkl', 'rb') as f:
        imageneta_label_to_indices = pickle.load(f)
    i = 0
    for key in list(imageneta_label_to_indices.keys()):
        margin_imageneta[i] = margin_imageneta[i]/len(imageneta_label_to_indices[key])
        i += 1
    margin_imageneta = torch.tensor(margin_imageneta)
    # margin_imageneta = torch.mean(margin_imageneta).item()
    margin_imageneta = torch.mean(torch.abs(margin_imagenet[np.array(list(imageneta_label_to_indices.keys()))]-margin_imageneta)).item()
    # margin_imageneta = torch.mean(margin_imageneta).item()
    imageneta_Fea = torch.load('imageneta_Fea.pt')
    indX = np.random.choice(len(imagenet_Fea), args.TN1, replace=False)
    indY = np.random.choice(len(imageneta_Fea), args.TN1)
    S = torch.cat((imagenet_Fea[indX],imageneta_Fea[indY]))
    imageneta_MMD, imageneta_NAMMD = reference_values(S, args.TN1)
    print("imageneta. MMD: ", str(imageneta_MMD), " NAMMD: ", str(imageneta_NAMMD), " Margin: ", str(margin_imageneta))
    del margin_imageneta, imageneta_Fea

    # noise_levels = np.arange(1, 21, step=1.0)
    # for noise in noise_levels:
    #     noise = round(noise, 1)
    #     with open('margin_imagenet_'+str(noise)+'.json', 'r') as f:
    #         margin_imagenetnoise = json.load(f)
    #     margin_imagenetnoise = torch.tensor(margin_imagenetnoise)/50
    #     margin_imagenetnoise = torch.mean(torch.abs(margin_imagenet-margin_imagenetnoise)).item()
    #     imagenetnoise_Fea = torch.load('imagenet_Fea_'+str(noise)+'.pt')
    #     indX = np.random.choice(len(imagenet_Fea), args.TN1, replace=False)
    #     indY = np.random.choice(len(imagenetnoise_Fea), args.TN1, replace=False)
    #     S = torch.cat((imagenet_Fea[indX],imagenetnoise_Fea[indY]))
    #     TEMP = MMDu(S, args.TN1, sigma0)
    #     MMD = get_item(TEMP[0], args.device)
    #     Reg = get_item(TEMP[2], args.device)
    #     NAMMD = MMD/Reg
    #     print("imagenetnoise. noise: ", str(noise), " MMD: ", str(MMD), " NAMMD: ", str(NAMMD), str(margin_imagenetnoise))
    #     del margin_imagenetnoise, imagenetnoise_Fea

    Results = np.zeros((len(args.sample_sizes), 2, args.n_exp))
    Final_Results = np.zeros((Results.shape[0], 2, 2))
    ResultsP = np.zeros(Results.shape)
    Final_ResultsP = np.zeros(Final_Results.shape)
    imagenet_Fea = torch.load('imagenet_Fea.pt')
    imagenetsk_Fea = torch.load('imagenetsk_Fea.pt')
    for ee in range(len(args.sample_sizes)):
        samplesize = args.sample_sizes[ee]
        for kk in range(args.n_exp):
            H_MMD, H_NAMMD, P_MMD, P_NAMMD  = testing(imagenet_Fea, imagenetsk_Fea, imagenetsk_MMD * 0.3, imagenetsk_NAMMD * 0.3 / 1.2, samplesize, kk+args.rs_test, sigma0, args.n_test, args.alpha, args.device)
            # H_MMD, H_NAMMD, P_MMD, P_NAMMD  = testing_per(imagenet_Fea, imagenetsk_Fea, args.N_per, samplesize, kk+args.rs_test, sigma0, args.n_test, args.alpha, args.device)
            Results[ee, 0, kk] = H_MMD.sum() / args.n_test
            Results[ee, 1, kk] = H_NAMMD.sum() / args.n_test
            ResultsP[ee, 0, kk] = P_MMD.sum() / args.n_test
            ResultsP[ee, 1, kk] = P_NAMMD.sum() / args.n_test

        Final_Results[ee][0][0] = Results[ee][0].sum()/args.n_exp
        Final_Results[ee][0][1] = Results[ee][0].std()/np.sqrt(args.n_exp)
        Final_Results[ee][1][0] = Results[ee][1].sum()/args.n_exp
        Final_Results[ee][1][1] = Results[ee][1].std()/np.sqrt(args.n_exp)
        np.savetxt('imagenetsk_'+str(args.sample_sizes)+str(args.rs_test), Final_Results.reshape(len(args.sample_sizes),-1), fmt='%.3f')
        print("imagenetsk test power:", " samplesize = ", str(samplesize))
        print("MMD: {:.3f}±{:.3f}".format(Final_Results[ee][0][0], Final_Results[ee][0][1]))
        print("NAMMD: {:.3f}±{:.3f}".format(Final_Results[ee][1][0], Final_Results[ee][1][1]))
        
        Final_ResultsP[ee][0][0] = ResultsP[ee][0].sum()/args.n_exp
        Final_ResultsP[ee][0][1] = ResultsP[ee][0].std()/np.sqrt(args.n_exp)
        Final_ResultsP[ee][1][0] = ResultsP[ee][1].sum()/args.n_exp
        Final_ResultsP[ee][1][1] = ResultsP[ee][1].std()/np.sqrt(args.n_exp)
        np.savetxt('imagenetsk_'+str(args.sample_sizes)+str(args.rs_test)+'_P', Final_ResultsP.reshape(len(args.sample_sizes),-1), fmt='%.10f')
        print("imagenetsk p-value:", " samplesize = ", str(samplesize))
        print("MMD: {:.10f}±{:.10f}".format(Final_ResultsP[ee][0][0], Final_ResultsP[ee][0][1]))
        print("NAMMD: {:.10f}±{:.10f}".format(Final_ResultsP[ee][1][0], Final_ResultsP[ee][1][1]))
    del imagenet_Fea, imagenetsk_Fea
    
    args.sample_sizes = [samplesize * 30 for samplesize in args.sample_sizes]
    Results = np.zeros((len(args.sample_sizes), 2, args.n_exp))
    Final_Results = np.zeros((Results.shape[0], 2, 2))
    ResultsP = np.zeros(Results.shape)
    Final_ResultsP = np.zeros(Final_Results.shape)
    imagenet_Fea = torch.load('imagenet_Fea.pt')
    imagenetr_Fea = torch.load('imagenetr_Fea.pt')
    for ee in range(len(args.sample_sizes)):
        samplesize = args.sample_sizes[ee]
        for kk in range(args.n_exp):
            H_MMD, H_NAMMD, P_MMD, P_NAMMD  = testing(imagenet_Fea, imagenetr_Fea, imagenetsk_MMD, imagenetsk_NAMMD, samplesize, kk+args.rs_test, sigma0, args.n_test, args.alpha, args.device)
            Results[ee, 0, kk] = H_MMD.sum() / args.n_test
            Results[ee, 1, kk] = H_NAMMD.sum() / args.n_test
            ResultsP[ee, 0, kk] = P_MMD.sum() / args.n_test
            ResultsP[ee, 1, kk] = P_NAMMD.sum() / args.n_test

        Final_Results[ee][0][0] = Results[ee][0].sum()/args.n_exp
        Final_Results[ee][0][1] = Results[ee][0].std()/np.sqrt(args.n_exp)
        Final_Results[ee][1][0] = Results[ee][1].sum()/args.n_exp
        Final_Results[ee][1][1] = Results[ee][1].std()/np.sqrt(args.n_exp)
        np.savetxt('imagenetr_'+str(args.sample_sizes)+str(args.rs_test), Final_Results.reshape(len(args.sample_sizes),-1), fmt='%.3f')
        print("imagenetr test power:", " samplesize = ", str(samplesize))
        print("MMD: {:.3f}±{:.3f}".format(Final_Results[ee][0][0], Final_Results[ee][0][1]))
        print("NAMMD: {:.3f}±{:.3f}".format(Final_Results[ee][1][0], Final_Results[ee][1][1]))
        
        Final_ResultsP[ee][0][0] = ResultsP[ee][0].sum()/args.n_exp
        Final_ResultsP[ee][0][1] = ResultsP[ee][0].std()/np.sqrt(args.n_exp)
        Final_ResultsP[ee][1][0] = ResultsP[ee][1].sum()/args.n_exp
        Final_ResultsP[ee][1][1] = ResultsP[ee][1].std()/np.sqrt(args.n_exp)
        np.savetxt('imagenetr_'+str(args.sample_sizes)+str(args.rs_test)+'_P', Final_ResultsP.reshape(len(args.sample_sizes),-1), fmt='%.10f')
        print("imagenetr p-value:", " samplesize = ", str(samplesize))
        print("MMD: {:.10f}±{:.10f}".format(Final_ResultsP[ee][0][0], Final_ResultsP[ee][0][1]))
        print("NAMMD: {:.10f}±{:.10f}".format(Final_ResultsP[ee][1][0], Final_ResultsP[ee][1][1]))
    del imagenet_Fea, imagenetr_Fea
    args.sample_sizes = [samplesize // 30 for samplesize in args.sample_sizes]
    
    Results = np.zeros((len(args.sample_sizes), 2, args.n_exp))
    Final_Results = np.zeros((Results.shape[0], 2, 2))
    ResultsP = np.zeros(Results.shape)
    Final_ResultsP = np.zeros(Final_Results.shape)
    imagenet_Fea = torch.load('imagenet_Fea.pt')
    imagenetv2_Fea = torch.load('imagenetv2_Fea.pt')
    for ee in range(len(args.sample_sizes)):
        samplesize = args.sample_sizes[ee]
        for kk in range(args.n_exp):
            H_MMD, H_NAMMD, P_MMD, P_NAMMD  = testing(imagenet_Fea, imagenetv2_Fea, imagenetr_MMD, imagenetr_NAMMD, samplesize, kk+args.rs_test, sigma0, args.n_test, args.alpha, args.device)
            Results[ee, 0, kk] = H_MMD.sum() / args.n_test
            Results[ee, 1, kk] = H_NAMMD.sum() / args.n_test
            ResultsP[ee, 0, kk] = P_MMD.sum() / args.n_test
            ResultsP[ee, 1, kk] = P_NAMMD.sum() / args.n_test

        Final_Results[ee][0][0] = Results[ee][0].sum()/args.n_exp
        Final_Results[ee][0][1] = Results[ee][0].std()/np.sqrt(args.n_exp)
        Final_Results[ee][1][0] = Results[ee][1].sum()/args.n_exp
        Final_Results[ee][1][1] = Results[ee][1].std()/np.sqrt(args.n_exp)
        np.savetxt('imagenetv2_'+str(args.sample_sizes)+str(args.rs_test), Final_Results.reshape(len(args.sample_sizes),-1), fmt='%.3f')
        print("imagenetv2 test power:", " samplesize = ", str(samplesize))
        print("MMD: {:.3f}±{:.3f}".format(Final_Results[ee][0][0], Final_Results[ee][0][1]))
        print("NAMMD: {:.3f}±{:.3f}".format(Final_Results[ee][1][0], Final_Results[ee][1][1]))
        
        Final_ResultsP[ee][0][0] = ResultsP[ee][0].sum()/args.n_exp
        Final_ResultsP[ee][0][1] = ResultsP[ee][0].std()/np.sqrt(args.n_exp)
        Final_ResultsP[ee][1][0] = ResultsP[ee][1].sum()/args.n_exp
        Final_ResultsP[ee][1][1] = ResultsP[ee][1].std()/np.sqrt(args.n_exp)
        np.savetxt('imagenetv2_'+str(args.sample_sizes)+str(args.rs_test)+'_P', Final_ResultsP.reshape(len(args.sample_sizes),-1), fmt='%.10f')
        print("imagenetv2 p-value:", " samplesize = ", str(samplesize))
        print("MMD: {:.10f}±{:.10f}".format(Final_ResultsP[ee][0][0], Final_ResultsP[ee][0][1]))
        print("NAMMD: {:.10f}±{:.10f}".format(Final_ResultsP[ee][1][0], Final_ResultsP[ee][1][1]))
    del imagenet_Fea, imagenetv2_Fea
    
    args.sample_sizes = [samplesize * 3 for samplesize in args.sample_sizes]
    Results = np.zeros((len(args.sample_sizes), 2, args.n_exp))
    Final_Results = np.zeros((Results.shape[0], 2, 2))
    ResultsP = np.zeros(Results.shape)
    Final_ResultsP = np.zeros(Final_Results.shape)
    imagenet_Fea = torch.load('imagenet_Fea.pt')
    imageneta_Fea = torch.load('imageneta_Fea.pt')
    for ee in range(len(args.sample_sizes)):
        samplesize = args.sample_sizes[ee]
        for kk in range(args.n_exp):
            H_MMD, H_NAMMD, P_MMD, P_NAMMD  = testing(imagenet_Fea, imageneta_Fea, imagenetv2_MMD, imagenetv2_NAMMD, samplesize, kk+args.rs_test, sigma0, args.n_test, args.alpha, args.device)
            Results[ee, 0, kk] = H_MMD.sum() / args.n_test
            Results[ee, 1, kk] = H_NAMMD.sum() / args.n_test
            ResultsP[ee, 0, kk] = P_MMD.sum() / args.n_test
            ResultsP[ee, 1, kk] = P_NAMMD.sum() / args.n_test

        Final_Results[ee][0][0] = Results[ee][0].sum()/args.n_exp
        Final_Results[ee][0][1] = Results[ee][0].std()/np.sqrt(args.n_exp)
        Final_Results[ee][1][0] = Results[ee][1].sum()/args.n_exp
        Final_Results[ee][1][1] = Results[ee][1].std()/np.sqrt(args.n_exp)
        np.savetxt('imageneta_'+str(args.sample_sizes)+str(args.rs_test), Final_Results.reshape(len(args.sample_sizes),-1), fmt='%.3f')
        print("imageneta test power:", " samplesize = ", str(samplesize))
        print("MMD: {:.3f}±{:.3f}".format(Final_Results[ee][0][0], Final_Results[ee][0][1]))
        print("NAMMD: {:.3f}±{:.3f}".format(Final_Results[ee][1][0], Final_Results[ee][1][1]))
        
        Final_ResultsP[ee][0][0] = ResultsP[ee][0].sum()/args.n_exp
        Final_ResultsP[ee][0][1] = ResultsP[ee][0].std()/np.sqrt(args.n_exp)
        Final_ResultsP[ee][1][0] = ResultsP[ee][1].sum()/args.n_exp
        Final_ResultsP[ee][1][1] = ResultsP[ee][1].std()/np.sqrt(args.n_exp)
        np.savetxt('imageneta_'+str(args.sample_sizes)+str(args.rs_test)+'_P', Final_ResultsP.reshape(len(args.sample_sizes),-1), fmt='%.10f')
        print("imageneta p-value:", " samplesize = ", str(samplesize))
        print("MMD: {:.10f}±{:.10f}".format(Final_ResultsP[ee][0][0], Final_ResultsP[ee][0][1]))
        print("NAMMD: {:.10f}±{:.10f}".format(Final_ResultsP[ee][1][0], Final_ResultsP[ee][1][1]))
    del imagenet_Fea, imageneta_Fea
    args.sample_sizes = [samplesize // 3 for samplesize in args.sample_sizes]
