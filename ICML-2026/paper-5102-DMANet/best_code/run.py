import argparse
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    parser = argparse.ArgumentParser(description='TimesNet')
    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast,imputation]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='weather', help='model id')
    parser.add_argument('--model', type=str, default='DMANet',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./all_datasets/weather',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # model define
    parser.add_argument('--enc_in', type=int, default=21, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=21, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=21, help='output size')
    #
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='optimizer learning rate')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=2, help='expansion factor of model')
    parser.add_argument('--global_topk', type=int, default=3, help='num of selection of global frequencies')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--auxi_lambda', type=float, default=1, help='weight of auxilary function')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_c', type=float, default=0.5, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='conv',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--kernel_size', type=int, default=3, help='DWConv kernel_size')
    
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    # save
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results', type=str, default='./results/', help='location of results')
    parser.add_argument('--test_results', type=str, default='./test_results/', help='location of test results')
    parser.add_argument('--log_path', type=str, default='./result_long_term_forecast1.txt', help='log path')
    parser.add_argument('--output_pred', action='store_true', help='output true and pred', default=False)
    parser.add_argument('--output_vis', action='store_true', help='output visual figures', default=False)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
 
    parser.add_argument('--rec_lambda', type=float, default=0., help='weight of reconstruction function')
    parser.add_argument('--auxi_loss', type=str, default='MAE', help='loss function')
    parser.add_argument('--auxi_mode', type=str, default='fft', help='auxi loss mode, options: [fft, rfft]')
    parser.add_argument('--auxi_type', type=str, default='complex',
                        help='auxi loss type, options: [complex, mag, phase, mag-phase]')
    parser.add_argument('--module_first', type=int, default=1, help='calculate module first then mean ')
    parser.add_argument('--leg_degree', type=int, default=2, help='degree of legendre polynomial')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--top_k', type=int, default=5, help='for global freq')

    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5', help='device ids of multile gpus')
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2024, help="Randomization seed")

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        print("input error!!")
    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_pl{}_dm{}_dff{}_downl{}_el{}_window{}_windowc{}_batch{}_lr{}_kernel{}_auxi{}_eb{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.d_ff,
                args.down_sampling_layers,
                args.e_layers,
                args.down_sampling_window,
                args.down_sampling_c,
                args.batch_size,
                args.learning_rate,
                args.kernel_size,
                args.auxi_lambda,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_pl{}_dm{}_dff{}_downl{}_el{}_window{}_windowc{}_batch{}_lr{}_kernel{}_auxi{}_eb{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.down_sampling_layers,
            args.e_layers,
            args.down_sampling_window,
            args.down_sampling_c,
            args.batch_size,
            args.learning_rate,
            args.kernel_size,
            args.auxi_lambda,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()