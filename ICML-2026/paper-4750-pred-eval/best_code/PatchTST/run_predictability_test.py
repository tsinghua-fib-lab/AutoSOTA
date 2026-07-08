#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import numpy as np
import torch
from exp.exp_main import Exp_Main

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Predictability evaluation (MSE vs Coherence-based MSE)')

# run_longExp.py 
parser.add_argument('--is_training', type=int, default=0)  
parser.add_argument('--train_only', type=bool, default=False)
parser.add_argument('--model_id', type=str, required=True)
parser.add_argument('--model', type=str, required=True)


parser.add_argument('--data', type=str, required=True)
parser.add_argument('--root_path', type=str, default='./data/ETT/')
parser.add_argument('--data_path', type=str, default='.csv')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=96)

# DLinear
parser.add_argument('--ps_lambda', type=float, default=3.0, help='weight for ps_loss')
parser.add_argument('--use_ps_loss', type=int, default=0, help='whether to use ps_los')
parser.add_argument('--patch_len_threshold', type=int, default=24, help='patch length threshold')


# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--des', type=str, default='eval')
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--use_amp', action='store_true', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices', type=str, default='0')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

parser.add_argument('--outdir', type=str, default='./predictability_results')
parser.add_argument('--alpha_boundary', type=float, default=1)
parser.add_argument('--welch_win_frac', type=float, default=0.25)
parser.add_argument('--welch_overlap', type=float, default=0.5)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--limit_batches', type=int, default=None)

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in evaluation:')
print(args)

setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_psloss{}_{}_{}'.format(
    args.model_id,
    args.model,
    args.data,
    args.features,
    args.seq_len,
    args.label_len,
    args.pred_len,
    args.d_model,
    args.n_heads,
    args.e_layers,
    args.d_layers,
    args.d_ff,
    args.factor,
    args.embed,
    args.distil,
    0,  # use_ps_loss 
    args.des, 0
)

exp = Exp_Main(args)
exp.evaluate_predictability(
    setting=setting,
    load=True,
    alpha_boundary=args.alpha_boundary,
    welch_win_frac=args.welch_win_frac,
    welch_overlap=args.welch_overlap,
    workers=args.workers,
    outdir=args.outdir,
    limit_batches=args.limit_batches,
)

torch.cuda.empty_cache()
