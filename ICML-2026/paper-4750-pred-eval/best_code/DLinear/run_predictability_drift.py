#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import numpy as np
import torch
from exp.exp_main import Exp_Main

fix_seed = 2025
random.seed(fix_seed); torch.manual_seed(fix_seed); np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Band-wise predictable energy (stacked) + right-axis MSE')

# —— Core parameters aligned with existing run_*
parser.add_argument('--is_training', type=int, default=0)
parser.add_argument('--train_only', type=bool, default=False)
parser.add_argument('--model_id', type=str, default='ETTh1_336_336')
parser.add_argument('--model', type=str, default='DLinear')

# data loader
parser.add_argument('--data', type=str, default='ETTh1')
parser.add_argument('--root_path', type=str, default='./datasets')
parser.add_argument('--data_path', type=str, default='ETTh1.csv')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./DLinear/checkpoints/')

# forecasting task
parser.add_argument('--seq_len', type=int, default=336)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=336)

# DLinear / PS-loss 
parser.add_argument('--individual', action='store_true', default=False)
parser.add_argument('--ps_lambda', type=float, default=3.0)
parser.add_argument('--use_ps_loss', type=int, default=0)
parser.add_argument('--patch_len_threshold', type=int, default=24)

# Model structure/training-independent hyperparameters (maintain compatibility)
parser.add_argument('--enc_in', type=int, default=321)
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

# GPU
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--use_amp', action='store_true', default=False)


# Evaluate common parameters (leave as is; outdir here is for overall evaluation)
parser.add_argument('--outdir', type=str, default='./predictability_results')
parser.add_argument('--alpha_boundary', type=float, default=1)

# Welch parameter
parser.add_argument('--welch_win_frac', type=float, default=0.25)
parser.add_argument('--welch_overlap', type=float, default=0.5)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--limit_batches', type=int, default=None)

# —— New: Frequency division stacking + dual y-axis required parameters ——
parser.add_argument('--channel', type=int, default=1)
parser.add_argument('--bands', type=str, default='auto:10',
                    help="auto:k 或 a-b,c-d,...（cycles/sample，Nyquist=0.5）")
parser.add_argument('--include_dc', action='store_true', default=False)
parser.add_argument('--max_samples', type=int, default=3000)
parser.add_argument('--outdir_bands', type=str, default='./predictability_bands_electricity_ch0',
                    help='Output directory for frequency stacked plots and CSV')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(i) for i in device_ids if i != '']
    args.gpu = args.device_ids[0] if len(args.device_ids) > 0 else args.gpu

print('Args in evaluation (bands dual-axis):')
print(args)

# —— setting
setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_psloss{}_{}_{}'.format(
    args.model_id, args.model, args.data, args.features,
    args.seq_len, args.label_len, args.pred_len,
    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
    args.factor, args.embed, args.distil, 0,  
    'Exp', 0
)

exp = Exp_Main(args)

print(f'>>>>>>> band-wise predictability + MSE : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
paths = exp.evaluate_band_predictability_dualaxis(
    setting=setting,
    channel=args.channel,
    bands=args.bands,
    welch_win_frac=args.welch_win_frac,
    welch_overlap=args.welch_overlap,
    include_dc=args.include_dc,
    max_samples=args.max_samples,
    outdir=args.outdir_bands,
    load = True,
)

for k, v in paths.items():
    print(f"{k}: {v}")

torch.cuda.empty_cache()
