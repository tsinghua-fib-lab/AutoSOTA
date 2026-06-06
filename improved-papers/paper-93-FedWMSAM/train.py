import torch
import argparse

from utils import *
from models import *
from server import *
from dataset import *
import numpy as np

#### ================= Open Float32 in A100 ================= ####
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#### ================= Open ignore warining ================= ####
import warnings
warnings.filterwarnings('ignore')
#### ======================================================== ####
print("##=============================================##")
print("##     Federated Learning Simulator Starts     ##")
print("##=============================================##")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100'], type=str, default='CIFAR10')             # select dataset
parser.add_argument('--model', choices=['LeNet', 'ResNet18'], type=str, default='ResNet18')                # select model
parser.add_argument('--non-iid', action='store_true', default=True)                                       # activate if use heterogeneous dataset 
parser.add_argument('--split-rule', choices=['Dirichlet', 'Pathological'], type=str, default='Dirichlet')  # select the dataset splitting rule
parser.add_argument('--split-coef', default=0.1, type=float)                                                  # --> if Dirichlet: select the Dirichlet coefficient (i.e. 0.1, 0.3, 0.6, 1)
                                                                                                              # --> if Pathological: select the Dirichlet coefficient (i.e. 3, 5)
parser.add_argument('--active-ratio', default=0.1, type=float)                                             # select the partial participating ratio (i.e. 0.1, 0.05)
parser.add_argument('--total-client', default=100, type=int)                                               # select the total number of clients (i.e. 100, 500)
parser.add_argument('--comm-rounds', default=700, type=int)                                               # select the global communication rounds T
parser.add_argument('--local-epochs', default=5, type=int)                                                 # select the local interval K
parser.add_argument('--batchsize', default=50, type=int)                                                   # select the batchsize
parser.add_argument('--weight-decay', default=0.001, type=float)                                           # select the weight-decay (i.e. 0.01, 0.001)
parser.add_argument('--local-learning-rate', default=0.1, type=float)                                      # select the local learning rate (generally 0.1 expect for local-adaptive-based)
parser.add_argument('--global-learning-rate', default=1.0, type=float)                                     # select the global learning rate (generally 1.0 expect for global-adaptive-based)
parser.add_argument('--lr-decay', default=0.998, type=float)                                               # select the learning rate decay (generally 0.998 expect for proxy-based)
parser.add_argument('--seed', default=20, type=int)                                                        # select the random seed
parser.add_argument('--cuda', default=0, type=int)                                                         # select the cuda ID
parser.add_argument('--data-file', default='./', type=str)                                                 # select the path of the root of Dataset
parser.add_argument('--out-file', default='out/', type=str)                                                # select the path of the log files
parser.add_argument('--save-model', action='store_true', default=False)                                    # activate if save the model
parser.add_argument('--use-RI', action='store_true', default=False)                                        # activate if use relaxed initialization (RI)

parser.add_argument('--alpha', default=0.1, type=float)                                                    # select the coefficient for client-momentum 
parser.add_argument('--beta', default=0.1, type=float)                                                     # select the coefficient for relaxed initialization
parser.add_argument('--beta1', default=0.9, type=float)                                                    # select the coefficient for the first-order momentum
parser.add_argument('--beta2', default=0.99, type=float)                                                   # select the coefficient for the second-order momentum
parser.add_argument('--lamb', default=0.1, type=float)                                                     # select the coefficient for the prox-term
parser.add_argument('--rho', default=0.05, type=float)                                                      # select the SAM perturbation rate
parser.add_argument('--gamma', default=1.0, type=float)                                                    # select the coefficient for the correction of SAM
parser.add_argument('--epsilon', default=0.01, type=float)  
parser.add_argument('--mode', choices=['normal'], type=str, default='normal')# select the minimal value for avoiding zero-division

parser.add_argument('--method', choices=['FedAvg', 'FedCM', 'FedDyn', 'SCAFFOLD', 'FedAdam', 'FedProx', 'FedSAM', 'MoFedSAM', \
                                         'FedGamma', 'FedSpeed', 'FedSMOO', 'FedLESAM', 'FedLESAM_D','FedLESAM_S','FedWMSAM'], type=str, default='FedWMSAM')
                                         
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    device = torch.device(args.cuda)
else:
    device = torch.device("cpu")

if __name__=='__main__':
    ### Generate IID or Heterogeneous Dataset
    if not args.non_iid:
        data_obj = DatasetObject(dataset=args.dataset, n_client=args.total_client, seed=args.seed, unbalanced_sgm=0, rule='iid',
                                     data_path=args.data_file)
        print("Initialize the Dataset     --->  {:s} {:s} {:d} clients".format(args.dataset, 'IID', args.total_client))
    else:
        data_obj = DatasetObject(dataset=args.dataset, n_client=args.total_client, seed=args.seed, unbalanced_sgm=0, rule=args.split_rule,
                                     rule_arg=args.split_coef, data_path=args.data_file)
        print("Initialize the Dataset     --->  {:s} {:s}-{:s} {:d} clients".format(args.dataset, args.split_rule, str(args.split_coef), args.total_client))
    

    ### Generate Model Function
    if args.dataset == 'CIFAR10':
        classes = 10
    elif args.dataset == 'CIFAR100':
        classes = 100
    else:
        raise NotImplementedError('not implemented dataset yet')

    ### Generate Model Function
    model_func = lambda: client_model(args.model, classes)
    print("Initialize the Model Func  --->  {:s} model".format(args.model))
    init_model = model_func()
    total_trainable_params = sum(p.numel() for p in init_model.parameters() if p.requires_grad)
    print("                           --->  {:d} parameters".format(total_trainable_params))
    init_par_list = get_mdl_params(init_model)
    
    ### Generate Server
    server_func = None
    if args.method == 'FedAvg':
        server_func = FedAvg
    elif args.method == 'FedCM':
        server_func = FedCM
    elif args.method == 'FedDyn':
        server_func = FedDyn
    elif args.method == 'SCAFFOLD':
        server_func = SCAFFOLD
    elif args.method == 'FedAdam':
        server_func = FedAdam
    elif args.method == 'FedProx':
        server_func = FedProx
    elif args.method == 'FedSAM':
        server_func = FedSAM
    elif args.method == 'MoFedSAM':
        server_func = MoFedSAM
    elif args.method == 'FedGamma':
        server_func = FedGamma
    elif args.method == 'FedSpeed':
        server_func = FedSpeed
    elif args.method == 'FedSMOO':
        server_func = FedSMOO
    elif args.method == 'FedLESAM':
        server_func = FedLESAM
    elif args.method == 'FedLESAM_D':
        server_func = FedLESAM_D
    elif args.method == 'FedLESAM_S':
        server_func = FedLESAM_S
    elif args.method == 'FedWMSAM':
        server_func = FedWMSAM
    else:
        raise NotImplementedError('not implemented method yet')
    
    _server = server_func(device=device, model_func=model_func, init_model=init_model, init_par_list=init_par_list,
                          datasets=data_obj, method=args.method, args=args)
    _server.train()
    