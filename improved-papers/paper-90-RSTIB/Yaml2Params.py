import argparse
from utils.util import get_logger
import os
import yaml

def read_name(config_path):
    nameList = config_path.split('/')
    allName = nameList[-1]
    nameList = allName.split('.')
    return nameList[0]
def genPath(path_str, date):
    return path_str+date+'/'
def print_args(args):
    log_level = 'INFO'
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = get_logger(args.logdir, __name__, 'info_{}.log'.format(args.name), level=log_level)
    logger.info(args)

    return logger

def dict2args(config, args):
    keys = config.keys()
    for key in keys:
        for k, v in config[key].items(): 
            setattr(args, k, v)        
    return args

def parse_args():
    parser = argparse.ArgumentParser(description='RSTIB-MLP')
    parser.add_argument('--config', default='./config/student/pems04/student_rstib_mlp.yaml', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as y:
        config = yaml.safe_load(y)
    args = dict2args(config, args)
    args.mdir = genPath(args.mdir, args.date)
    args.logdir = genPath(args.logdir, args.date)
    args.name = read_name(args.config)
    return args
    
args = parse_args()
logger = print_args(args)