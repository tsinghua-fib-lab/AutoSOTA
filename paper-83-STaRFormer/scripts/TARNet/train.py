# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:09:45 2021

@author: Ranak Roy Chowdhury
"""
import os.path as osp
import tarnet.utils as utils
import argparse
import warnings
import sys
import shutil
import torch
import os
import numpy as np
import math
import random
import time 
import datetime

warnings.filterwarnings("ignore")

try:
    from src.datamodule import GeoLifeDatamodule
    from src.utils import DatasetOptions
except:
    import sys
    file_path = osp.abspath(__file__)
    dir_path = "/".join(file_path.split("/")[:-3])
    sys.path.append(dir_path)
    from src.datamodule import GeoLifeDatamodule
    from src.utils import DatasetOptions


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AF')
parser.add_argument('--batch', type=int, default=128) # dkt: 512
parser.add_argument('--lr', type=float, default=0.01) # dkt: 0.006104054297174028
parser.add_argument('--nlayers', type=int, default=2) # 4
parser.add_argument('--emb_size', type=int, default=64) # 32
parser.add_argument('--nhead', type=int, default=8) # 4
parser.add_argument('--task_rate', type=float, default=0.5)
parser.add_argument('--masking_ratio', type=float, default=0.15) # 0.3
parser.add_argument('--lamb', type=float, default=0.8)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--ratio_highest_attention', type=float, default=0.5) 
parser.add_argument('--avg', type=str, default='macro')
parser.add_argument('--dropout', type=float, default=0.01) # 0.2772483650222462
parser.add_argument('--nhid', type=int, default=128) # 8 
parser.add_argument('--nhid_task', type=int, default=128) # 8 
parser.add_argument('--nhid_tar', type=int, default=128) # 8
parser.add_argument('--task_type', type=str, default='classification', help='[classification, regression]')
# DKT specific args
parser.add_argument('--instance', type=str, default="2024-10-16_22:03:28", help="")
parser.add_argument('--aws_profile', type=str, default=None, help="")
parser.add_argument('--s3_bucket_path', type=str, default=None, help="")
parser.add_argument('--s3_bucket_prefix', type=str, default=None, help="")
parser.add_argument('--num_train', type=int, default=10240, help="")
parser.add_argument('--num_val', type=int, default=None, help="")
parser.add_argument('--num_test', type=int, default=None, help="")
parser.add_argument('--device', type=str, default='gpu', help="")
parser.add_argument('--seed', type=int, default=42, help="")
args = parser.parse_args()


def prepare_own_ds(prop, datamodule):
    train_data = []
    train_labels = []
    for sample in datamodule.train_dataset:
        # pad data to get same N
        pad_N = datamodule.dataset._max_seq_len - sample.data.shape[0]
        pad_D = 0
        sample_pad = np.pad(sample.data.numpy(), ((0, pad_N), (0, pad_D)), mode='constant', constant_values=0)
        train_data.append(sample_pad[None, ...]) # [1, N, D]
        if prop['dataset'].lower() == DatasetOptions.geolife:
            train_labels.append(sample.label.numpy().reshape(-1, 1))
        else:
            train_labels.append(sample.label.numpy())
    
    X_train_task = np.concatenate(train_data, axis=0, dtype=np.float32) # [num_train+num_val, N, D]
    y_train_task = np.concatenate(train_labels, axis=0, dtype=np.float32).reshape(-1, 1)

    n_samples_train = args.batch * (len(datamodule.train_dataset) // args.batch) # args.batch * len(datamodule.train_dataset) // args.batch # args.batch * (len(datamodule.train_dataset) // args.batch)
    
    print('Train :', n_samples_train)
    X_train_task = X_train_task[:n_samples_train, ...]
    y_train_task = y_train_task[:n_samples_train, ...]

    if np.where(np.isnan(X_train_task) == True)[0].size != 0:
        # found values where np.isnan is True
        print('WARNING -- Found NaN values in X_train_task, set them to 0.')
        X_train_task = np.nan_to_num(X_train_task)
    assert np.where(np.isnan(X_train_task) == True)[0].size == 0

    val_data = []
    val_labels = []
    for sample in datamodule.val_dataset:
        pad_N = datamodule.dataset._max_seq_len - sample.data.shape[0]
        pad_D = 0
        sample_pad = np.pad(sample.data.numpy(), ((0, pad_N), (0, pad_D)), mode='constant', constant_values=0)
        val_data.append(sample_pad[None, ...])
        if prop['dataset'].lower() == DatasetOptions.geolife:
            val_labels.append(sample.label.numpy().reshape(-1, 1))
        else:
            val_labels.append(sample.label.numpy())

    
    X_val = np.concatenate(val_data, axis=0, dtype=np.float32) # [num_train+num_val, N, D])
    y_val = np.concatenate(val_labels, axis=0).reshape(-1, 1)

    n_samples_val = args.batch * (len(datamodule.val_dataset) // args.batch) # args.batch * (len(datamodule.val_dataset) // args.batch)
    print('Val: ', n_samples_val)
    X_val = X_val[:n_samples_val, ...]
    y_val = y_val[:n_samples_val, ...]
    
    if np.where(np.isnan(X_val) == True)[0].size != 0:
        # found values where np.isnan is True
        print('WARNING -- Found NaN values in X_val, set them to 0.')
        X_val = np.nan_to_num(X_val)
    assert np.where(np.isnan(X_val) == True)[0].size == 0

    test_data = []
    test_labels = []
    for sample in datamodule.test_dataset:
        pad_N = datamodule.dataset._max_seq_len - sample.data.shape[0]
        pad_D = 0
        sample_pad = np.pad(sample.data.numpy(), ((0, pad_N), (0, pad_D)), mode='constant', constant_values=0)
        test_data.append(sample_pad[None, ...])
        if prop['dataset'].lower() == DatasetOptions.geolife:
            test_labels.append(sample.label.numpy().reshape(-1, 1))
        else:
            test_labels.append(sample.label.numpy())
    
    X_test = np.concatenate(test_data, axis=0, dtype=np.float32) # [num_train+num_val, N, D])
    y_test = np.concatenate(test_labels, axis=0).reshape(-1, 1)

    n_samples_test = args.batch * (len(datamodule.test_dataset) // args.batch) #args.batch * (len(datamodule.test_dataset) // args.batch)
    print('Test: ', n_samples_test)
    X_test = X_test[:n_samples_test, ...]
    y_test = y_test[:n_samples_test, ...]

    if np.where(np.isnan(X_test) == True)[0].size != 0:
        # found values where np.isnan is True
        print('WARNING -- Found NaN values in X_val, set them to 0.')
        X_test = np.nan_to_num(X_test)
    assert np.where(np.isnan(X_test) == True)[0].size == 0

    return X_train_task, y_train_task, X_val, y_val, X_test, y_test


def main():    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    prop = utils.get_prop(args)
    prop['seed'] = args.seed
    if prop['dataset'] in {'person', 'Cricket', 'AWR', 'AF', 'BM', 'DDG', 'EW', 'Epilepsy', 'ERing', 
                                     'EC', 'FD', 'FM', 'Opp_l', 'HMD', 'Handwriting', 'Heartbeat', 'IW', 'JV', 
                                     'Libras', 'LSST', 'NATOPS', 'Phoneme', 'RS', 'SCP1', 'SCP2', 'UWGL', 'SAD', 
                                     'SWJ', 'Opp_g', 'PD', 'Occupancy', 'PAMAP2', 'MI', 'CT', 'PEMS-SF', 
                                     # tsr
                                     'AE', 'BC', 'BPM10', 'BPM25', 'LFMC', 'IEEEPPG'}:
        # update with best Hyperparameters given by source code TARNet
        import pickle
        with open('./scripts/TARNet/tarnet/hyperparameters.pkl', 'rb') as f:
            hyparmas = pickle.load(f)
        specific_hyperparams = hyparmas[prop['dataset']]
        prop.update(specific_hyperparams)

        if prop['dataset'] == 'IEEEPPG':
            # Otherwise leads to VRAM Issues
            prop['batch'] = 32

    #path = './data/' + prop['dataset'] + '/'
    

    # centralized
    if prop['dataset'].lower() == DatasetOptions.geolife:
        dm_params = {
            'batch_size': args.batch,
            'num_workers': 0,
            'seed': args.seed,
            'training_method': 'centralized',
            'train_test_split': {
                'train': 0.7,
                'val': 0.1,
                'test': 0.2,
            },
            'identical_training_class_label_distribution': False,
            'aws_profile': args.aws_profile,
            's3_bucket_path': args.s3_bucket_path,
            'pad_sequence': True,
            'val_batch_size': args.batch,
            'test_batch_size': args.batch,
            'num_train': args.num_train,
            'num_val': args.num_val,
            'num_test': args.num_test,
            'max_trajectory_length': 512,
            'synthetic_minority_upsampling': False,
            'noise_level': 0.01,
            "text_labels": [0, 1, 2, 3],
            "display_labels": ['bike', 'bus', 'car', 'walk'],
        }

        datamodule = GeoLifeDatamodule(**dm_params)
        datamodule.setup()
        #instance = datamodule.dataset.dataset_dir_paths['centralized_instance'].split("/")[-1]
        X_train_task, y_train_task, X_val, y_val, X_test, y_test = prepare_own_ds(prop, datamodule)

    elif prop['dataset'] in {'person', 'Cricket', 'AWR', 'AF', 'BM', 'DDG', 'EW', 'Epilepsy', 'ERing', 
                                     'EC', 'FD', 'FM', 'Opp_l', 'HMD', 'Handwriting', 'Heartbeat', 'IW', 'JV', 
                                     'Libras', 'LSST', 'NATOPS', 'Phoneme', 'RS', 'SCP1', 'SCP2', 'UWGL', 'SAD', 
                                     'SWJ', 'Opp_g', 'PD', 'Occupancy', 'PAMAP2', 'MI', 'CT', 'PEMS-SF', 'AE', 
                                     'BC', 'BPM10', 'BPM25', 'LFMC', 'IEEEPPG',
                                     # new tsr
                                     'AR', 'BIDMCHR', 'BIDMCRR', 'BIDMCSpO2', 'C3M', 'FM1', 'FM2', 'FM3', 
                                     'HPC1', 'HPC2', 'NHS', 'NTS', 'PPG', }:
        import os
        _dir = os.path.join(*os.path.abspath(__file__).split('/')[5:-1])
        print(_dir)
        path = f'./{_dir}/tarnet/data/' + prop['dataset'] + '/'

        print('Data loading start...')
        X_train, y_train, X_test, y_test = utils.data_loader(args.dataset, path, prop['task_type'])
        print('Data loading complete...')    

        print('Data preprocessing start...')
        X_train_task, y_train_task, X_test, y_test = utils.preprocess(prop, X_train, y_train, X_test, y_test)
        print(X_train_task.shape, y_train_task.shape, X_test.shape, y_test.shape)
        if prop['dataset'] in ['BIDMCHR', 'BIDMCRR', 'BIDMCSpO2']:
            # shorten and select only every fourth datapoint, due to length of trajectory
            # otherwise gpu issues 
            X_train_task = X_train_task[:, ::4, :]
            X_test = X_test[:, ::4, :]
            print('Updated Dimension:', X_train_task.shape, y_train_task.shape, X_test.shape, y_test.shape)
        print('Data preprocessing complete...')
    else:
        raise NotImplementedError(f"{prop['dataset']}")

    
    if prop['dataset'].lower() == DatasetOptions.geolife:
        prop['nclasses'] = torch.max(torch.stack(datamodule.dataset.targets)).item() + 1 if prop['task_type'] == 'classification' else None
        prop['dataset'], prop['seq_len'], prop['input_size'] = prop['dataset'], datamodule.dataset._max_seq_len, datamodule.dataset.data[0].data.shape[-1]
    else:
        prop['nclasses'] = torch.max(y_train_task).item() + 1 if prop['task_type'] == 'classification' else None
        prop['dataset'], prop['seq_len'], prop['input_size'] = prop['dataset'], X_train_task.shape[1], X_train_task.shape[2]
        

    prop['device'] = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    print('Initializing model...')
    model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer = utils.initialize_training(prop)
    print('Model intialized...')

    #X_train_task = datamodule.train_dataset.data
    #y_train_task = torch.concatenate(datamodule.train_dataset.targets, axis=0).reshape(-1, 1)
    #X_test =  datamodule.test_dataset.data
    #y_test = torch.concatenate(datamodule.test_dataset.targets, axis=0).reshape(-1, 1)
    
    print('Training start...')
    t = time.time()
    utils.training(model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer, X_train_task, y_train_task, X_test, y_test, prop)
    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
    print('Training complete...')



if __name__ == "__main__":
    main()
