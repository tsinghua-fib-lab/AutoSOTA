#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import numpy as np
import torch
from exp.exp_main import Exp_Main

fix_seed = 2025
random.seed(fix_seed); np.random.seed(fix_seed); torch.manual_seed(fix_seed)

parser = argparse.ArgumentParser(description='SCE: band-wise utilization (eta_m) + energy weights')

# —— Core parameters aligned with existing run_* ——
parser.add_argument('--is_training', type=int, default=0)
parser.add_argument('--train_only', type=bool, default=False)
parser.add_argument('--model_id', type=str, default='ETTh1_720_720')
parser.add_argument('--model', type=str, default='DLinear')

parser.add_argument('--data', type=str, default='ETTh1')
parser.add_argument('--root_path', type=str, default='./datasets')
parser.add_argument('--data_path', type=str, default='ETTh1.csv')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./DLinear/checkpoints/')

parser.add_argument('--seq_len', type=int, default=720)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=720)

# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
parser.add_argument('--ps_lambda', type=float, default=3.0, help='weight for ps_loss')
parser.add_argument('--use_ps_loss', type=int, default=0, help='whether to use ps_los')
parser.add_argument('--patch_len_threshold', type=int, default=24, help='patch length threshold')

# Model structure/training-independent hyperparameters (maintain compatibility)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--dec_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--factor', type=int, default=1)
parser.add_argument('--distil', action='store_false', default=True)
parser.add_argument('--dropout', type=float, default=0.05)
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--output_attention', action='store_true', default=False)
parser.add_argument('--do_predict', action='store_true', default=False)

# Run/Optimize (placeholder for compatibility with Exp_Main)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--des', type=str, default='Exp')
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--use_amp', action='store_true', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices', type=str, default='0')

# ——  Frequency band evaluation parameters ——
parser.add_argument('--channel', type=int, default=4)
parser.add_argument('--bands', type=str, default='auto:30')
parser.add_argument('--welch_win_frac', type=float, default=0.25)
parser.add_argument('--welch_overlap', type=float, default=0.5)
parser.add_argument('--include_dc', action='store_true', default=False)
parser.add_argument('--alpha_boundary', type=float, default=1.0)
parser.add_argument('--max_samples', type=int, default=30000)
parser.add_argument('--outdir', type=str, default='./predictability_bands_ECL')

# Weight loading control
parser.add_argument('--ckpt_file', type=str, default=None)
parser.add_argument('--no_load', action='store_true', default=False)

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids if id_ != '']
    if len(args.device_ids) > 0:
        args.gpu = args.device_ids[0]

print('Args in SCE bands evaluation:')
print(args)

# setting string
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
            args.use_ps_loss,
            args.des, 0)

exp = Exp_Main(args)
print(f'>>>>>>> SCE band utilization : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

paths = exp.evaluate_band_LUR(
    setting=setting,
    channel=args.channel,
    bands=args.bands,
    welch_win_frac=args.welch_win_frac,
    welch_overlap=args.welch_overlap,
    include_dc=args.include_dc,
    alpha_boundary=args.alpha_boundary,
    max_samples=args.max_samples,
    outdir=args.outdir,
    load=(not args.no_load),
    ckpt_file=args.ckpt_file,
)

print("[DONE]")
for k, v in paths.items():
    print(f"{k}: {v}")

torch.cuda.empty_cache()
