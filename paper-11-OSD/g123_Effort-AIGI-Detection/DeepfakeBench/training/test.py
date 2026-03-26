"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='./training/config/detector/effort.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str,
                    default='./weights/effort_ckpt.pth')
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

on_2060 = "2060" in torch.cuda.get_device_name()
def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test',
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())

    return np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)

def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps,feat_nps = test_one_dataset(model, test_data_loaders[key])

        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset

        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions




def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if on_2060:
        config['lmdb_dir'] = r'I:\transform_2_lmdb'
        config['train_batchSize'] = 10
        config['workers'] = 0
    else:
        config['workers'] = 8
        config['lmdb_dir'] = r'/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/jikangcheng/data/LMDBs'
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters in the model: {total_trainable_params}")
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        # # 加载模型的状态字典
        # model_dict = model.state_dict()
        # new_ckpt={}
        # for key in ckpt.keys():
        #     # 替换键
        #     new_key = key.replace('common_encoder_f','student_encoder')
        #     # 将旧的值复制到新的键下
        #     new_ckpt[new_key] = ckpt[key]
        # # 获取ckpt和model的key集合
        # ckpt_keys = set(new_ckpt.keys())
        # model_keys = set(model_dict.keys())
        #
        # # 找出共同的key
        # common_keys = ckpt_keys & model_keys
        # print("Common keys:")
        # for key in common_keys:
        #     print(key)
        #
        # # 找出只在ckpt中的key
        # ckpt_unique_keys = ckpt_keys - model_keys
        # print("\nKeys only in ckpt:")
        # for key in ckpt_unique_keys:
        #     print(key)
        #
        # # 找出只在model中的key
        # model_unique_keys = model_keys - ckpt_keys
        # print("\nKeys only in model:")
        # for key in model_unique_keys:
        #     print(key)

        # 创建一个新的字典，删除module前缀
        new_weights = {}
        for key, value in ckpt.items():
            new_key = key.replace('module.', '')  # 删除module前缀
            #new_key = 'backbone.' + new_key  # 删除module前缀
          #  if 'base_model.' in new_key:
          #      new_key = new_key.replace('base_model.', 'backbone.')
          #  if 'classifier.' in new_key:
          #      new_key = new_key.replace('classifier.', 'head.')
            new_weights[new_key] = value


        model.load_state_dict(new_weights, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')

 #   clip_rank_results = analyze_clip_effective_rank(model.backbone)

 #   for layer_name, rank in clip_rank_results.items():
 #       print(f"Layer: {layer_name}, Effective Rank: {rank}")


    # start testing
    best_metric = test_epoch(model, test_data_loaders)
    print('===> Test Done!')

if __name__ == '__main__':
    main()
