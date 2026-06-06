import copy

import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from timesurl import TimesURL
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
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
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads, deterministic=False)

    args.load_tp = True
    
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset, load_tp = args.load_tp)

    elif args.loader == 'Others':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_others(args.dataset, load_tp = args.load_tp)

    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset, load_tp = args.load_tp)
        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting' if 'forecast' in args.run_name else 'imputation'
        offset = 0 if task_type == 'forecasting' else 96
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, offset=offset, load_tp=args.load_tp)
        train_data = {'x': data['x'][:, train_slice], 'mask': data['mask'][:, train_slice]}

    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting' if 'forecast' in args.run_name else 'imputation'
        offset = 0 if task_type == 'forecasting' else 96
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, offset=offset, univar=True, load_tp=args.load_tp)
        train_data = {'x': data['x'][:, train_slice], 'mask': data['mask'][:, train_slice]}
        
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        # all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data_task, train_labels, train_timestamps, test_data, test_labels, test_timestamps, delay = datautils.load_anomaly(args.dataset, load_tp=args.load_tp)
        train_data = datautils.gen_ano_train_data(train_data_task['x'])
        train_data = {
            'x': np.concatenate([train_data, datautils.gen_ano_train_data(train_timestamps, train_data.shape[1], True)], axis=-1),
            'mask': train_data_task['mask']}
        
    else:
        raise ValueError(f"Unknown loader {args.loader}.")

    args.task_type = task_type
    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')
    print(train_data['x'].shape)

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
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True,
        is_scheduler=True if args.sgd else False,
        temp=args.temp
    )
    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(model, train_data_task, train_labels, train_timestamps, test_data, test_labels, test_timestamps, delay)
        elif task_type == 'imputation':
            out, eval_res = tasks.eval_imputation(model, data, test_slice, args.missing_rate, n_covariate_cols, device)
        else:
            assert False

        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)

    print("Finished.")
