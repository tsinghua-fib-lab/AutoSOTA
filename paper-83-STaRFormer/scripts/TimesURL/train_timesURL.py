#
# This code is adjustd from https://github.com/Alrash/TimesURL/tree/main#
# Specifcally https://github.com/Alrash/TimesURL/blob/main/src/train.py
#
import argparse
import os.path as osp

import numpy as np
import argparse
import os
import sys
import time
import datetime
import torch
import random

import timesURL_src.tasks  as tasks
import timesURL_src.datautils  as datautils

from timesURL_src.timesurl import TimesURL
from timesURL_src.utils import (
    init_dl_program, name_with_datetime, 
    pkl_save, data_dropout, generate_mask
)

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


def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback


def prepare_own_ds(args, datamodule):
    train_data = []
    train_labels = []
    for sample in datamodule.train_dataset:
        # pad data to get same N
        pad_N = datamodule.dataset._max_seq_len - sample.data.shape[0]
        pad_D = 0
        sample_pad = np.pad(sample.data.numpy(), ((0, pad_N), (0, pad_D)), mode='constant', constant_values=0)
        train_data.append(sample_pad[None, ...]) # [1, N, D]
        if args.dataset.lower() == DatasetOptions.geolife:
            train_labels.append(sample.label.numpy().reshape(-1, 1))
        else:
            train_labels.append(sample.label.numpy())

    train_data_adjusted = []
    for sample in train_data: # add artifical time stamps
        # sampel shape [1, 265, 8]
        adjusted = np.zeros((sample.shape[0], sample.shape[1], sample.shape[2]+1))
        adjusted[..., :-1] = sample
        adjusted[..., -1] = np.arange(0, len(sample), 1)
        train_data_adjusted.append(adjusted)

    train_data = np.concatenate(train_data_adjusted, axis=0) # [num_train+num_val, N, D]
    train_labels = np.concatenate(train_labels, axis=0).reshape(-1, 1)
    train_data[train_data == 0] = np.nan # required by model

    test_data = []
    test_labels = []
    for sample in datamodule.test_dataset:
        pad_N = datamodule.dataset._max_seq_len - sample.data.shape[0]
        pad_D = 0
        sample_pad = np.pad(sample.data.numpy(), ((0, pad_N), (0, pad_D)), mode='constant', constant_values=0)
        test_data.append(sample_pad[None, ...])
        if args.dataset.lower() == DatasetOptions.geolife:
            test_labels.append(sample.label.numpy().reshape(-1, 1))
        else:
            test_labels.append(sample.label.numpy())

    test_data_adjusted = []
    for sample in test_data: # add artifical time stamps
        adjusted = np.zeros((sample.shape[0], sample.shape[1], sample.shape[2]+1))
        adjusted[..., :-1] = sample
        adjusted[..., -1] = np.arange(0, len(sample), 1)
        test_data_adjusted.append(adjusted)
    test_data = np.concatenate(test_data_adjusted, axis=0) # [num_train+num_val, N, D]
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1, 1)
    test_data[test_data == 0] = np.nan # required by model

    return train_data, train_labels, test_data, test_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch_size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr_dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max_train_length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save_every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--max_threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--sgd', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--load_tp', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--temp', type=float, default=1.0,)
    parser.add_argument('--lmd', type=float, default=0.01, )
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--segment_num', type=int, default=3,
                        help='number of time interval segment to mask, default: 3 time intervals')
    parser.add_argument('--mask_ratio_per_seg', type=float, default=0.05,
                        help='fraction of the sequence length to mask for each time interval, deafult: 0.05 * seq_len to be masked for each of the time interval')
    # DKT specific args
    parser.add_argument('--instance', type=str, default="2024-10-16_22:03:28", help="")
    parser.add_argument('--aws_profile', type=str, default=None, help="")
    parser.add_argument('--s3_bucket_path', type=str, default=None, help="")
    parser.add_argument('--s3_bucket_prefix', type=str, default=None, help="")
    parser.add_argument('--num_train', type=int, default=None, help="")
    parser.add_argument('--num_val', type=int, default=None, help="")
    parser.add_argument('--num_test', type=int, default=None, help="")
    parser.add_argument('--device', type=str, default='gpu', help="")
    
    args = parser.parse_args()
    
    # seed everything 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(args)
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads, deterministic=False)

    args.load_tp = True

    print('Loading data... ', end='')
    # centralized
    if args.loader.lower() == DatasetOptions.geolife:
        task_type = 'classification'
        dm_params = {
            'batch_size': args.batch_size,
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
            'val_batch_size': args.batch_size,
            'test_batch_size': args.batch_size,
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
    else:
        raise NotImplementedError

    if args.loader in [DatasetOptions.geolife]:
        train_data, train_labels, test_data, test_labels = prepare_own_ds(args, datamodule)
    else:
        raise NotImplementedError

    # generate mask 
    p = 1
    mask_tr, mask_te = generate_mask(train_data, p), generate_mask(test_data, p)

    n_samples_train = -1 if args.num_train is None else args.num_train # 1000
    n_samples_test = -1 if args.num_test is None else args.num_train # 1000
    if args.num_train is not None:
        train_data = {'x': train_data[:n_samples_train, ...], 'mask': mask_tr[:n_samples_train, ...]}
        train_labels = train_labels[:n_samples_train]
    else:
        train_data = {'x': train_data, 'mask': mask_tr}
        train_labels = train_labels

    if args.num_test is not None:
        test_data = {'x': test_data[:n_samples_test, ...], 'mask': mask_te[:n_samples_test, ...]}
        test_labels = test_labels[:n_samples_test]
    else:
        test_data = {'x': test_data, 'mask': mask_te}
        test_labels = test_labels
    
    print('Training shapes:', train_data['x'].shape, train_data['mask'].shape)
    print('Training shapes:',test_data['x'].shape, test_data['mask'].shape)

    args.task_type = task_type
    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        sgd=args.sgd,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
        args=args
    )

    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    t = time.time()

    model = TimesURL(
        input_dims=train_data['x'].shape[-1] - (1 if args.load_tp else 0),
        device=device,
        **config
    )
    print('\nStarting training...')
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True,
        is_scheduler=True if args.sgd else False,
        temp=args.temp
    )
    #model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
    te = time.time()
    if args.eval:
        if task_type == 'classification':
            train_data = {
                'x': train_data['x'][..., :-1],
                'mask': train_data['mask'][..., :-1]
            }
            test_data = {
                'x': test_data['x'][..., :-1],
                'mask': test_data['mask'][..., :-1]
            }
            for k, v in train_data.items():
                print(k, v.shape, test_data[k].shape)
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm', batch_size=16, seed=args.seed)
        # elif task_type == 'forecasting':
        #     out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
        # elif task_type == 'anomaly_detection':
        #     out, eval_res = tasks.eval_anomaly_detection(model, train_data_task, train_labels, train_timestamps, test_data, test_labels, test_timestamps, delay)
        # elif task_type == 'imputation':
        #     out, eval_res = tasks.eval_imputation(model, data, test_slice, args.missing_rate, n_covariate_cols, device)
        else:
            raise RuntimeError(f'{task_type=} has not been implemeted!')

        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)
    te = time.time() - te
    print(f"\Evaluation time: {datetime.timedelta(seconds=te)}\n")

    print("Finished.")
