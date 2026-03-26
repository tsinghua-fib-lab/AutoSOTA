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
import matplotlib.pyplot as plt
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument("--model_name", nargs="+")
parser.add_argument('--weights_path', type=str,
                    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
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

    assert len(config['test_dataset'])==1
    test_data_loader = get_test_data_loader(config, config['test_dataset'][0])
    return test_data_loader


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def plot_logit_distribution(real_logits, fake_logits, save_path):
    plt.figure(figsize=(8, 8))

    # 绘制散点图
    plt.scatter(real_logits[:, 0], real_logits[:, 1], color='blue', label='Real', alpha=0.2)
    plt.scatter(fake_logits[:, 0], fake_logits[:, 1], color='red', label='Fake', alpha=0.2)

    # 添加过原点的决策边界 x = y
    min_val = min(real_logits[:, 0].min(), real_logits[:, 1].min(), fake_logits[:, 0].min(), fake_logits[:, 1].min())
    max_val = max(real_logits[:, 0].max(), real_logits[:, 1].max(), fake_logits[:, 0].max(), fake_logits[:, 1].max())
    x_vals = np.linspace(min_val, max_val, 100)
    plt.plot(x_vals, x_vals, color='black', linestyle='--', label='Decision Boundary (x = y)')

    # 设置标题和标签
    plt.title('Logit Outputs for Real and Fake Classes')
    plt.xlabel('Logit for Real Class')
    plt.ylabel('Logit for Fake Class')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, args.test_dataset[0] + '.png'))
    plt.show()


def calculate_logits_and_confidence(model, data_loader):
    model.eval()

    real_logits = []
    fake_logits = []
    real_confidences = []
    fake_confidences = []

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
        output = predictions['cls']
        logits = output.detach().cpu()
        confidences = F.softmax(logits, dim=-1)

        for i in range(logits.size(0)):
            if label[i] == 0:
                real_logits.append(logits[i])
                real_confidences.append(confidences[i])
            else:
                fake_logits.append(logits[i])
                fake_confidences.append(confidences[i])


    real_logits = torch.stack(real_logits, dim=0)
    fake_logits = torch.stack(fake_logits, dim=0)
    real_confidences = torch.stack(real_confidences, dim=0)
    fake_confidences = torch.stack(fake_confidences, dim=0)

    return real_logits, fake_logits, real_confidences, fake_confidences



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
    test_data_loader = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device).eval()
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        new_weights = {}
        for key, value in ckpt.items():
            new_key = key.replace('module.', '')
            new_weights[new_key] = value

        model.load_state_dict(new_weights, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')

    # # start testing
    # best_metric = test_epoch(model, test_data_loaders)
    # print('===> Test Done!')

    # Box plot
    real_logits, fake_logits, real_confidences, fake_confidences = calculate_logits_and_confidence(model, test_data_loader)

    # Plot the logit distribution in 2D space
    save_path = f"scatter_plot/visualization/{args.model_name[0]}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_logit_distribution(real_logits, fake_logits, save_path)


    save_path = f"box_plot/visualization/{args.model_name[0]}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Convert tensors to numpy arrays
    # logits_real = real_logits.cpu().numpy().reshape(-1)
    # logits_fake = fake_logits.cpu().numpy().reshape(-1)

    # real_confidences = real_confidences.cpu().numpy().reshape(-1)
    # fake_confidences = fake_confidences.cpu().numpy().reshape(-1)

    # logits_real = torch.cat([real_logits[0], fake_logits[0]]).cpu().numpy().reshape(-1)
    # logits_fake = torch.cat([real_logits[1], fake_logits[1]]).cpu().numpy().reshape(-1)



    # Set Seaborn style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 14

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))



    if real_logits.shape[-1] == 2:
        logits_real = (real_logits[:, 0].cpu().numpy(), fake_logits[:, 0].cpu().numpy())
        logits_fake = (real_logits[:, 1].cpu().numpy(), fake_logits[:, 1].cpu().numpy())

        real_confidences_list = (real_confidences[:, 0].cpu().numpy(), fake_confidences[:, 0].cpu().numpy())
        fake_confidences_list = (real_confidences[:, 1].cpu().numpy(), fake_confidences[:, 1].cpu().numpy())

        # Plot histograms and KDE for logits
        sns.histplot(logits_real[0], bins=30, kde=True, stat='probability', label='real', color=sns.color_palette("Blues")[-1], ax=axs[0])
        sns.histplot(logits_real[1], bins=30, kde=True, stat='probability', label='fake', color=sns.color_palette("Reds")[-1], ax=axs[0])
        axs[0].set_ylim(0, 0.15)
        axs[0].legend()

        # sns.histplot(logits_fake[0], bins=30, kde=True, stat='probability', color=sns.color_palette("Blues")[-1], ax=axs[0, 1])
        # sns.histplot(logits_fake[1], bins=30, kde=True, stat='probability', color=sns.color_palette("Reds")[-1], ax=axs[0, 1])
        # axs[0, 1].set_ylim(0, 0.15)
        # axs[0, 1].legend()

        # Plot histograms and KDE for confidences
        sns.histplot(real_confidences_list[0], bins=30, kde=True, stat='probability', label='real', color=sns.color_palette("Blues")[-1], ax=axs[1])
        sns.histplot(real_confidences_list[1], bins=30, kde=True, stat='probability', label='fake', color=sns.color_palette("Reds")[-1], ax=axs[1])
        axs[1].legend()
        # sns.histplot(fake_confidences_list[0], bins=30, kde=True, stat='probability', color=sns.color_palette("Blues")[-1], ax=axs[1])
        # sns.histplot(fake_confidences_list[1], bins=30, kde=True, stat='probability', color=sns.color_palette("Reds")[-1], ax=axs[1])

        # Set titles
        axs[0].set_title('Logit Distribution of Real and Fake Classes')
        # axs[0, 1].set_title('Fake Logits')
        axs[1].set_title('Confidence Distribution of Real and Fake Classes')
        # axs[1, 1].set_title('Fake Confidences')

        # Show the plot
        plt.tight_layout()
        save_ = os.path.join(save_path, args.test_dataset[0] + '.png')
        plt.savefig(save_)

        # # Create a DataFrame with logits and confidences as columns
        # df_real = pd.DataFrame({'logits_real': logits_real, 'real_confidences': real_confidences})
        # df_fake = pd.DataFrame({'logits_fake': logits_fake, 'fake_confidences': fake_confidences})

        # # Write DataFrame to csv file
        # df_real.to_csv('output_logits_real.csv', index=False)
        # df_fake.to_csv('output_logits_fake.csv', index=False)




    elif real_logits.shape[-1] == 1:

        logits_real = real_logits.cpu().numpy().flatten()
        logits_fake = fake_logits.cpu().numpy().flatten()
        real_confidences_list = real_confidences.cpu().numpy().flatten()
        fake_confidences_list = fake_confidences.cpu().numpy().flatten()

        # Plot histograms and KDE for logits
        sns.histplot(logits_fake, bins=30, kde=True, stat='probability', label='real', color=sns.color_palette("Blues")[-1], ax=axs[0])
        sns.histplot(logits_real, bins=30, kde=True, stat='probability', label='fake', color=sns.color_palette("Reds")[-1], ax=axs[0])
        axs[0].set_ylim(0, 0.7)
        axs[0].legend()

        # Plot histograms and KDE for confidences
        sns.histplot(real_confidences_list, bins=30, kde=True, stat='probability', label='real', color=sns.color_palette("Blues")[-1], ax=axs[1])
        sns.histplot(fake_confidences_list, bins=30, kde=True, stat='probability', label='fake', color=sns.color_palette("Reds")[-1], ax=axs[1])
        axs[1].legend()

        # Set titles
        axs[0].set_title('Logit Distribution of Real and Fake Classes')
        axs[1].set_title('Confidence Distribution of Real and Fake Classes')

        # Show the plot
        plt.tight_layout()
        save_ = os.path.join(save_path, args.test_dataset[0] + '.png')
        plt.savefig(save_)




    else:
        raise ValueError('The number of classes is not 1 or 2')



if __name__ == '__main__':
    main()
