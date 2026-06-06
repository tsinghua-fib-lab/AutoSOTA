import random
import os
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import yaml
    
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def make_optimizer(params_group, args):
    optimizer = getattr(torch.optim, args.OPTIMIZER['name'])(params_group, **args.OPTIMIZER['args'])
    return optimizer

def make_schedule(optimizer, args):
    scheduler = getattr(torch.optim.lr_scheduler, args.LR_SCHEDULER['name'])(optimizer=optimizer,  **args.LR_SCHEDULER['args'])
    return scheduler

def prepare_optimizer_scheduler(params_group, args):
    if args.OPTIMIZER is None:
        optimizer = torch.optim.AdamW(params_group, lr=args.lr)
    else:
        optimizer = make_optimizer(params_group, args)

    if args.LR_SCHEDULER is None:
        scheduler = None
    else:
        scheduler = make_schedule(optimizer, args)
    return optimizer, scheduler

def make_criterion(criterion_args: dict):
    args = criterion_args.get('args', None)
    criterion = getattr(torch.nn, criterion_args['name'])
    if args is None:
        criterion = criterion()
    else:
        criterion = criterion(**args)
    return criterion

def writing_log(log_path: str, info: str, encoding='utf-8', mode='a+'):
    with open(log_path, encoding=encoding, mode=mode) as f:
        f.write(info)
        
def make_dir(dir_path: str):
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

def record_hyper_parameter(path: str, name: str, **kwargs):
    with open(path + '{:}_config.yaml'.format(name), 'w') as f:
        yaml.dump(kwargs, f, default_flow_style=False)

class TempArgs:
    def __init__(self) -> None:
        pass

def read_yaml_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def set_config2args(args):
    cfg = read_yaml_config(args.config)
    for key, value in cfg.items():
        setattr(args, key, value)
    return args

def prepare_args(parser):
    args = parser.parse_args()
    args.save_path = getattr(args, 'dump_path', './checkpoint/') + args.dir + '/'
    if args.config is not None:
        args = set_config2args(args)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.world_size = torch.cuda.device_count()
    args.local_rank = 0
    if torch.cuda.is_available() and not args.cpu:
        args.use_cuda = True
    else:
        args.use_cuda = False
    return args

def deploy_config(is_parse: bool = True):
    parser = argparse.ArgumentParser(description='Pytorch Mnn Training Template')
    # my custom parameters
    parser.add_argument('-c', '--config', default=None, type=str, metavar='FILE', help='YAML config file')
    parser.add_argument('--bs', default=50, type=int, help='batch size')
    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--dir', default='results', type=str, help='dir path that used to save checkpoint')
    parser.add_argument('--cpu', action='store_true', default=False, help='Use CPU only or not')
    parser.add_argument('--dataset', default='mnist', type=str, help='type of dataset')
    parser.add_argument('--data_dir', default='./data/', type=str, help='type of dataset')
    parser.add_argument('--gpu', default=None, type=str, help='specify gpu idx if use gpu')
    parser.add_argument('--save_name', default='network', type=str, help='alias to save net')

    # Distributed Data Parallel setting
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--workers', default=1, type=int,
                        help='num workers for dataloader. ')
    parser.add_argument('--pin_mem', action='store_true', default=False, help='use pine memory')
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of nodes for distributed training')

    # Learning rate schedule parameters
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    if is_parse:
        args = prepare_args(parser)
        return args
    else:
        return parser
