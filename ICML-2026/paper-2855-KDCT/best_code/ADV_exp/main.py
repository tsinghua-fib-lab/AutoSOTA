import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
import sys
import os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
from utils import *
from train_model import *
from adv_generator import * 
import time
start_time = time.time()

# parameters of experimental setting
parser.add_argument('--N1',    default=10000,    help = 'Size of each sample in optimization')
parser.add_argument('--rs',    default=283,    help = 'Random seed')
parser.add_argument('--N1_',    default=10000,    help = 'Size of each sample in estimation')


parser.add_argument('--N1_test',    default=1500,    help = 'Size of each sample in testing')
parser.add_argument('--rs_test',    default=183,    help = 'Random seed in testing')

# parameters of experimental setting
parser.add_argument('--n_exp',  default=10,              help='Number of experiment runs')
parser.add_argument('--n_test', default=100,             help='Number of two-sample test runs')
parser.add_argument('--alpha',  default=0.05,            help='Confidence level of two-sample test')
parser.add_argument('--K',       default=1,            help='the upperbound of kernel')
parser.add_argument('--device', default=torch.device("cuda"),  help='Device of data')
parser.add_argument('--dtype',  default=torch.float,          help='Dtype of data')

parser.add_argument('--perturb_steps',  default=20,           help='perturb_steps')
parser.add_argument('--epss',  default=[1,2,3,4,5,6,7,8,9,10],           help='perturb_steps')
parser.add_argument('--loss_fn',  default="cent",           help='loss_fn')
parser.add_argument('--category',  default="Madry",           help='category')
parser.add_argument('--net', type=str, default="resnet18", help="choose from resnet18, resnet34")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10, svhn")
parser.add_argument('--model_path', default='./Res18_model/net_150.pth', help='model checkpoint')

# parameters of MMD
parser.add_argument('--ne_MMD', default=2000,   help='Number of MMD optimization epochs')
parser.add_argument('--bs_MMD', default=2000,    help='Batch size of MMD in optimization')
parser.add_argument('--lr_MMD', default=1.0, help='Learning rate of MMD in optimization')

# parameters of NAMMD
parser.add_argument('--ne_NAMMD', default=0,   help='Number of NAMMD optimization epochs')
parser.add_argument('--bs_NAMMD', default=2000,    help='Batch size of NAMMD in optimization')
parser.add_argument('--lr_NAMMD', default=1.0, help='Learning rate of NAMMD in optimization')
parser.add_argument('--b_NAMMD', default=0.1,     help='Balance parameter of MMD and Reg terms in ptimization')

args = parser.parse_args()

if args.net == "resnet18":
    model = ResNet18_Fea().to(args.device)
    net = "resnet18"
if args.net == "resnet34":
    model = ResNet34_Fea().cuda()
    net = "resnet34"

ckpt = torch.load(args.model_path)
model.load_state_dict(ckpt)

set_random_seed(args.rs)

transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)
for X, target in test_loader:
    X, target = X.detach().to(args.device), target.detach().to(args.device)
num_classes = 10
one_hot = torch.zeros(target.size(0), num_classes, device=target.device)
one_hot.scatter_(1, target.unsqueeze(1), 1)
ww = one_hot @ model.fc.weight.detach().to(args.device)
X_Fea = torch.zeros(ww.shape).to(args.device)
for i in range(20):
    X_Fea[i*500:(i+1)*500] = model(X[i*500:(i+1)*500]).detach().to(args.device).reshape(500,-1) * ww[i*500:(i+1)*500].detach()
del X
eps = args.epss[0]
Y=adv_generator(args.perturb_steps, epsilon=eps/255, step_size=eps/255/20,loss_fn=args.loss_fn, category=args.category, net=args.net, dataset=args.dataset, model_path=args.model_path).detach().to(args.device)
Y_Fea = torch.zeros(ww.shape).to(args.device)
for i in range(20):
    Y_Fea[i*500:(i+1)*500] = model(Y[i*500:(i+1)*500]).detach().to(args.device).reshape(500,-1) * ww[i*500:(i+1)*500].detach()
del Y
indX = np.random.choice(len(X_Fea), args.N1, replace=False)
indY = np.random.choice(len(Y_Fea), args.N1, replace=False)
S = torch.cat((X_Fea[indX],Y_Fea[indY]))
sigma_mmd = MMD_fit(S, args.N1, args.lr_MMD, args.ne_MMD, args.bs_MMD, args.device, seed=args.rs)
sigma_nammd = NAMMD_fit(S, args.N1, sigma_mmd, args.lr_NAMMD, args.ne_NAMMD, args.bs_NAMMD, args.b_NAMMD, args.device, seed=args.rs)
sigma0 = (sigma_mmd, sigma_nammd)
print("Selected bandwidths: MMD = {:.6g}, NAMMD = {:.6g}".format(sigma_mmd.item(), sigma_nammd.item()))
print('Training Done!')

Final_results = np.zeros((len(args.epss),2))
for ee in range(len(args.epss)):
    eps = args.epss[ee]
    Y = adv_generator(args.perturb_steps, epsilon=eps/255, step_size=eps/255/20, loss_fn=args.loss_fn, category=args.category, net=args.net, dataset=args.dataset, model_path=args.model_path).detach().to(args.device)
    Y_Fea = torch.zeros(ww.shape).to(args.device)
    for i in range(20):
        Y_Fea[i*500:(i+1)*500] = model(Y[i*500:(i+1)*500]).detach().to(args.device).reshape(500,-1) * ww[i*500:(i+1)*500].detach()
    del Y
    indX = np.random.choice(len(X_Fea), args.N1_, replace=False)
    indY = np.random.choice(len(Y_Fea), args.N1_, replace=False)
    S = torch.cat((X_Fea[indX],Y_Fea[indY]))
    TEMP = MMDu(S, args.N1_, sigma_mmd)
    MMD = get_item(TEMP[0], args.device)
    TEMP = MMDu(S, args.N1_, sigma_nammd)
    NAMMD = get_item(TEMP[0]/TEMP[2], args.device)
    Final_results[ee][0] = MMD
    Final_results[ee][1] = NAMMD
    print("N1_ = ", str(args.N1_), ", values of ",  str(eps))
    print("MMD: {:.10f}".format(Final_results[ee][0]))
    print("NAMMD: {:.10f}".format(Final_results[ee][1]))
    np.savetxt('./'+str(args.N1_)+'_'+str(args.epss), Final_results, fmt='%.10f')


Results_ = np.zeros((len(args.epss), 2, args.n_exp))
Final_results_ = np.zeros((Results_.shape[0],2, 2))
Results_P = np.zeros((len(args.epss), 2, args.n_exp))
Final_results_P = np.zeros((Results_P.shape[0],2, 2))
for ee in range(len(args.epss)):
    eps = args.epss[ee]
    Y = adv_generator(args.perturb_steps, epsilon=eps/255, step_size=eps/255/20, loss_fn=args.loss_fn, category=args.category, net=args.net, dataset=args.dataset, model_path=args.model_path).detach().to(args.device)
    Y_Fea = torch.zeros(ww.shape).to(args.device)
    for i in range(20):
        Y_Fea[i*500:(i+1)*500] = model(Y[i*500:(i+1)*500]).detach().to(args.device).reshape(500,-1) * ww[i*500:(i+1)*500].detach()
    del Y
    for kk in range(args.n_exp):
        H_MMD, H_NAMMD, P_MMD, P_NAMMD  = testing(X_Fea, Y_Fea, Final_results[3][0], Final_results[3][1], args.N1_test, kk+args.rs_test, sigma0, args.n_test, args.alpha, args.device)
        Results_[ee, 0, kk] = H_MMD.sum() / args.n_test
        Results_[ee, 1, kk] = H_NAMMD.sum() / args.n_test
        Results_P[ee, 0, kk] = P_MMD.sum() / args.n_test
        Results_P[ee, 1, kk] = P_NAMMD.sum() / args.n_test

    Final_results_[ee][0][0] = Results_[ee][0].sum()/args.n_exp
    Final_results_[ee][0][1] = Results_[ee][0].std()/np.sqrt(args.n_exp)
    Final_results_[ee][1][0] = Results_[ee][1].sum()/args.n_exp
    Final_results_[ee][1][1] = Results_[ee][1].std()/np.sqrt(args.n_exp)
    np.savetxt(str(args.N1_test)+'_'+str(args.rs_test), Final_results_.reshape(len(args.epss),-1), fmt='%.3f')
    print("test power:", " eps = ", str(eps), ", N1_test = ", str(args.N1_test), ", rs_test = ", str(args.rs_test))
    print("MMD: {:.3f}±{:.3f}".format(Final_results_[ee][0][0], Final_results_[ee][0][1]))
    print("NAMMD: {:.3f}±{:.3f}".format(Final_results_[ee][1][0], Final_results_[ee][1][1]))
    
    Final_results_P[ee][0][0] = Results_P[ee][0].sum()/args.n_exp
    Final_results_P[ee][0][1] = Results_P[ee][0].std()/np.sqrt(args.n_exp)
    Final_results_P[ee][1][0] = Results_P[ee][1].sum()/args.n_exp
    Final_results_P[ee][1][1] = Results_P[ee][1].std()/np.sqrt(args.n_exp)
    np.savetxt(str(args.N1_test)+'_'+str(args.rs_test)+'_P', Final_results_P.reshape(len(args.epss),-1), fmt='%.10f')
    print("p-value:", " eps = ", str(eps), ", N1_test = ", str(args.N1_test), ", rs_test = ", str(args.rs_test))
    print("MMD: {:.10f}±{:.10f}".format(Final_results_P[ee][0][0], Final_results_P[ee][0][1]))
    print("NAMMD: {:.10f}±{:.10f}".format(Final_results_P[ee][1][0], Final_results_P[ee][1][1]))


# End timing
end_time = time.time()
# Calculate execution time
execution_time = end_time - start_time
print(f"Program execution time: {execution_time} seconds", flush=True)
