import copy
import torch
import argparse
import warnings
import numpy as np
import logging
import torchvision
import random
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverdyn import FedDyn
from flcore.servers.serversam import FedSAM
from flcore.servers.serverspeed import FedSpeed
from flcore.servers.serversmoo import FedSMOO
from flcore.servers.serverlesamd import FedLESAMD
from flcore.servers.servergmt import FedGMT
from flcore.trainmodel.resnet import resnet8,resnet18
from flcore.trainmodel.CNN import FedAvgNetCIFAR
from flcore.trainmodel.vit import ViT
from flcore.trainmodel.nlp import fastText
from utils.mem_utils import MemReporter


logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")


def run(args):
    reporter = MemReporter()
    print("Creating server and clients ...")
    if args.dataset == 'cinic10':
        args.model = ViT().to(args.device)
    elif args.dataset == 'cifar10':
        # args.model = resnet18(num_classes = 10).to(args.device)
        args.model = FedAvgNetCIFAR().to(args.device)
    elif args.dataset == 'cifar100':
        args.model = resnet8().to(args.device)
    elif args.dataset == 'agnews':
        args.model = fastText().to(args.device)
    else:
        print("check args.dataset !!!!!!")
        exit()

    if args.algorithm == "FedAvg":
        server = FedAvg(args)

    elif args.algorithm == "FedDyn":
        server = FedDyn(args)   
    elif args.algorithm == "FedSAM":
        server = FedSAM(args)  
    elif args.algorithm == "FedSpeed":
        server = FedSpeed(args)  

    elif args.algorithm == "FedSMOO":
        server = FedSMOO(args)  
    
    elif args.algorithm == "FedLESAMD":
        server = FedLESAMD(args)  


    elif args.algorithm == "FedGMT":
        server = FedGMT(args)

    else:
        raise NotImplementedError
    
    server.train()

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Universal setting
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-dev', "--device", type=str, default="cuda:0")
    parser.add_argument('-data', "--dataset", type=str, default="cifar10")
    parser.add_argument('-gr', "--global_rounds", type=int, default=500)
    parser.add_argument('-lbs', "--batch_size", type=int, default=50)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.998)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('-le', "--local_epochs", type=int, default=5)
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.1,help="Ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=100,help="Total number of clients")
    parser.add_argument("--save_gap", type=int, default=25,help="Rounds gap to save result")
    parser.add_argument("--seed", type=int, default=1)


    #FedGMT
    parser.add_argument('-ga', "--gama", type=float, default=1.0)
    parser.add_argument('-al', "--alpha", type=float, default=0.95)
    parser.add_argument('-tau', "--tau", type=float, default=3)
    #FedDyn
    parser.add_argument('-be', "--beta", type=float, default=10,help="penalty coefficient")
    #FedSAM
    parser.add_argument('-rh', "--rho", type=float, default=0.01)


    args = parser.parse_args()
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn 
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.dataset == 'cinic10':
        args.num_classes = 10

    elif args.dataset == 'cifar10':
        args.num_classes = 10

    elif args.dataset == 'cifar100':
        args.num_classes = 100

    elif args.dataset == 'agnews':
        args.num_classes = 4

    else:
        print("check your dataset!!!!!!!!!!!!!")
        exit()


    print("=" * 50)
    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Using device: {}".format(args.device))
    print("=" * 50)
    print("sam rho: {}".format(args.rho))
    print("=" * 50)

    run(args)



