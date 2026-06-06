import argparse
import os
import torch
import torch.distributed as dist
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
try:
    from exp.exp_long_term_forecasting_chronos import Exp_Long_Term_Forecast_Chronos
except (ImportError, ModuleNotFoundError):
    Exp_Long_Term_Forecast_Chronos = None
import random
import numpy as np
from utils.tools import HiddenPrints

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='SEMPO')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name, options:[long_term_forecast, long_term_forecast_chronos]')
    parser.add_argument('--is_pretraining', type=int, default=1, help='status')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--is_zeroshot', type=int, default=1, help='status')
    parser.add_argument('--train_test', type=int, default=1, help='train_test')
    parser.add_argument('--model_id', type=str, default='ETTm1', help='model id')
    parser.add_argument('--model', type=str, default='SEMPO', help='model name, options: [SEMPO_CL, SEMPO, Chronos, Moirai, Timer, TimesFM]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--percent', type=int, default=10, help='few-shot or full-shot')
    parser.add_argument('--setting', type=str, default='experiment setting', help='randomly initialize an experimental setup')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--horizon_lengths', type=list, default=[1,96,192,336,720], help='prediction sequence length list')
    parser.add_argument('--patch_len', type=int, default=64, help='input sequence length')
    parser.add_argument('--stride', type=int, default=64, help='stride between patch')
   
    # model define
    parser.add_argument('--c_in', type=int, default=1, help='input size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=2, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')    
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--head_type', default='prediction', type=str, help='head type of different task')
    parser.add_argument('--domain_len', type=int, help='the number of domain', default=128)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=10, help='pretrain epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', choices=['constant_with_warmup'], help='adjust learning rate')
    # weight decay
    parser.add_argument('--warmup_steps', type=int, default=0, help='warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='adam beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.95, help='adam beta2')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    parser.add_argument('--visualize', action='store_true', help='visualize', default=False)
    parser.add_argument('--decay_fac', type=float, default=0.75)
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # Set up multi-GPU training
    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64210")
        hosts = int(os.environ.get("WORLD_SIZE", "1"))  # number of nodes
        rank = int(os.environ.get("RANK", "0"))  # node id
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()  # gpus per node
        args.local_rank = local_rank
        print('ip: {}, port: {}, hosts: {}, rank: {}, local_rank: {}, gpus: {}'.format(ip, port, hosts, rank, local_rank, gpus))
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts, rank=rank)
        print('init_process_group finished')
        torch.cuda.set_device(local_rank)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'long_term_forecast_chronos':
        Exp = Exp_Long_Term_Forecast_Chronos
    else:
        raise ValueError('task name not found')
   
    with HiddenPrints(int(os.environ.get("LOCAL_RANK", "0"))):
        print('Args in experiment:')
        # print_args(args)
           
        if args.is_pretraining == 1:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                    args.task_name,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.patch_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    ii)
                args.setting = setting
                exp = Exp(args)  # set experiments

                print('>>>>>>>start pretraining : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.pretrain(setting)    

        if args.is_training == 1:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                    args.task_name,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.patch_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    ii)
                args.setting = setting
                exp = Exp(args)  # set experiments

                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                if args.data == 'UTSD':
                    exp.train(setting)
                elif args.data == 'CI':
                    exp.train(setting, train=1)
                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.test(setting)
                torch.cuda.empty_cache()
        else:
            if args.data == 'CI':
                ii = 0
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                    args.task_name,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.patch_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    ii)
                args.setting = setting
                exp = Exp(args)  # set experiments
               
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting, test=1)
                torch.cuda.empty_cache()
