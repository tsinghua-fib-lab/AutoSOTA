from copy import deepcopy

import os

import torch

from utils.data import get_loaders
from utils.parser import parse_args
from utils.model import get_model
from utils.train import train_source, train_sfda2
from utils.utils import validate, log, ClassLevelMembershipInferenceAttack
from utils.forget import minimax, uc_minimax, unsir

import warnings




warnings.filterwarnings('ignore')

if not torch.cuda.is_available():
    print("Warning! cuda device not found")

args = parse_args()

config = {
    'seed': args.seed,
    'device': args.device,
    'split': 0.8,
    'batch': 32,
    'workers': 2,
    'bottleneck': 256,
    'epochs': args.epochs,
    'source_epochs': args.source_epochs,
    'smooth': 0.1,
    'iter_per_epoch': args.iter,
    'dataset': args.dataset,
    'source': args.source,
    'target': args.target,
    'data_path': './data',
    'dump_path': './dump',
    'vis_path': './vis',
    'save_path': './weights',
    'fast_train': args.fast_train,
    'method': args.method,
    'forget_classes': list(map(int, args.forget_classes.split(','))),
    'm_alpha': args.m_alpha,
    'm_samples': args.m_samples,
    'm_init': not args.m_dont_init,
    'm_update': args.m_update,
    'm_label': args.m_label
}

torch.manual_seed(config['seed'])
torch.cuda.set_device(config['device'])


# Data Loading

source_train_dl, source_test_dl, source_retain_train_dl, source_retain_test_dl, \
    source_forget_dl = get_loaders(config['source'], config)[:5]

target_train_dl, target_test_dl, target_retain_train_dl, target_retain_test_dl, \
    target_forget_dl, target_retain_subset_dl = get_loaders(config['target'], config)


# Source Model Training

classifier = get_model(config)

source_path = os.path.join(config['save_path'], config['dataset'], f"{config['source']}.pt")

source_classifier = deepcopy(classifier)

if not os.path.exists(source_path):
    source_classifier = train_source(source_classifier, source_train_dl, source_test_dl, config)

source_classifier.load_state_dict(torch.load(source_path, map_location=config['device']))


# DASEC Unlearning

if config['method'] == 'original':

    target_classifier = train_sfda2(source_classifier, target_retain_train_dl, target_retain_test_dl, source_test_dl, config, save='original')


if config['method'] == 'retrain':

    source_classifier = train_source(deepcopy(classifier), source_retain_train_dl, source_retain_test_dl, config, save=False)
    target_classifier = train_sfda2(source_classifier, target_retain_train_dl, target_retain_test_dl, source_test_dl, config, save='retrain')


if config['method'] == 'finetune':

    path = os.path.join(config['save_path'], config['dataset'], 'original', f"{config['source']}_{config['target']}.pt")
    if not os.path.exists(path):
        target_classifier = train_sfda2(source_classifier, target_retain_train_dl, target_retain_test_dl, source_test_dl, config, save='original')

    target_classifier = deepcopy(classifier)
    target_classifier.load_state_dict(torch.load(path, map_location=config['device']))
    
    target_classifier = train_sfda2(target_classifier, target_retain_train_dl, target_retain_test_dl, source_test_dl, config)


if config['method'] == 'minimax':
    target_classifier = minimax(source_classifier, target_retain_train_dl, target_retain_test_dl, target_forget_dl, source_test_dl, config, save='minimax')


if config['method'] == 'uniform_minimax':
    config['m_label'] = 'uniform'
    target_classifier = minimax(source_classifier, target_retain_train_dl, target_retain_test_dl, target_forget_dl, source_test_dl, config, save='uniform_minimax')


if config['method'] == 'random_minimax':
    config['m_label'] = 'random'
    target_classifier = minimax(source_classifier, target_retain_train_dl, target_retain_test_dl, target_forget_dl, source_test_dl, config, save='random_minimax')


if config['method'] == 'uc_minimax':
    target_classifier = uc_minimax(source_classifier, target_retain_train_dl, target_retain_test_dl, target_forget_dl, source_test_dl, config)


if config['method'] == 'unsir':

    source_classifier = unsir(source_classifier, target_retain_train_dl, target_retain_test_dl, target_forget_dl, source_test_dl, config)
    target_classifier = train_sfda2(source_classifier, target_retain_train_dl, target_retain_test_dl, source_test_dl, config)


target_retain_acc = validate(target_classifier, target_retain_test_dl, config)
target_forget_acc = validate(target_classifier, target_forget_dl, config)

# mia = ClassLevelMembershipInferenceAttack()
# acc = mia.get_mia(target_classifier, target_retain_subset_dl, target_forget_dl, config)


log(f"[{config['source'][0]} -> {config['target'][0]}] [{config['method']}] = ({target_retain_acc:.1f} | {target_forget_acc:.1f})")