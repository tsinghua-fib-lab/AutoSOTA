import copy
import logging
import os
import os.path
import sys
import time
import torch

from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from methods.hybrid_energy_distance_prompt_trainer import HybridEnergyDistancePromptTrainer
from methods.hybrid_energy_distance_prompt_eval import HybridEnergyDistancePromptEval
class saveModel():

    def __init__(self, args,network,all_keys):
        self.args=args
        self._network=network
        self.all_keys=all_keys
        
        
def run(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    for seed in seed_list:

        args['seed'] = seed
        args['device'] = device
        if(args["prefix"]=="train"):
            _train(args)
        elif(args["prefix"]=="eval"):
            _eval(args)
        else:
            raise ValueError('Unknown prefix: {}.'.format(args["prefix_type"]))


def _train(args):
    
    logfilename = './logs/{}_{}_'.format(args['model_name'],args['dataset'])+ time.strftime("%Y-%m-%d-%H %M %S", time.localtime())
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    os.makedirs(logfilename)
    _set_random(args['seed'])
    _set_device(args)
    print_args(args)
    
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
    args['class_order'] = data_manager._class_order
    model = HybridEnergyDistancePromptTrainer(args)
    cnn_curve = {'top1': []}
    

    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.begin_incremental(data_manager)
        model.incremental_train(data_manager)
        cnn_accy = model.eval_task()
        model.after_task()

        
        if cnn_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            cnn_curve['top1'].append(cnn_accy['grouped']['total'])# 记录历史CNN top1 curve
            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
    
        s_model=saveModel(args,model._network,model.all_keys)
        torch.save(s_model, os.path.join(logfilename, "task_{}.pth".format(int(task))))

def _eval(args):
    
    logfilename = './logs/{}_{}_'.format(args['model_name'],args['dataset'])+ time.strftime("%Y-%m-%d-%H %M %S", time.localtime())
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # os.makedirs(logfilename)
    _set_random(args['seed'])
    _set_device(args)
    print_args(args)
    
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
    args['class_order'] = data_manager._class_order
    model = HybridEnergyDistancePromptEval(args)
    model.eval_task_last(data_manager)
    

def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random(seed):
    torch.manual_seed(seed) #为CPU中设置种子，生成随机数
    torch.cuda.manual_seed(seed) #为特定GPU设置种子，生成随机数
    torch.cuda.manual_seed_all(seed) #为所有GPU设置种子，生成随机数
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
