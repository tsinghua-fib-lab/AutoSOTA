from torch import nn

def find_layers(module, layers=[nn.Conv2d, nn.Linear, nn.Embedding], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_ptq_arguments(**parser_kwargs):
    import argparse
    parser = argparse.ArgumentParser(**parser_kwargs)
    
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    
    ## Model
    parser.add_argument("--model_path", type=str, default='facebook/opt-125m', help='path of the model to be quantized')
    parser.add_argument("--save_model", action='store_true', help='Whether to save the fake-quantized model')
    
    ## Calib. Data
    parser.add_argument('--calib_data', type=str, default="c4", choices=["c4", "wikitext2"])
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--seqlen', type=int, default=2048, help='maximum sequence length')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')

    ## Quantization Configs
    parser.add_argument('--w_bits', type=int, default=3)
    parser.add_argument('--w_sym', action='store_true', help='Whether to perform symmetric weight quantization')
    parser.add_argument('--groupsize', type=int, default=-1, help='Group size for groupwise quantization. -1 means disabled (per-channel only).')

    ## aespa Options
    parser.add_argument('--block_v', action="store_true", help="Whether to apply block-wise objective for the value projection")
    parser.add_argument('--loss_option', type=str, default='global', choices=['local', 'global'], help='Global or local loss for hyperparameter search and Adaround')

    # Quantization Parameters Computation (scale and zero)
    parser.add_argument('--use_zfold', action='store_true', help="Whether to apply Z-Fold")

    # Integer Weight Optimization
    parser.add_argument('--order_option', type=str, default='spin', choices=['spin', 'act', 'none'], help='Whether to apply Hessian-based re-ordering')
    parser.add_argument('--comp_method', type=str, default='GPTQ', choices=['GPTAQ', 'GPTQ'], help='Compensation method')
    parser.add_argument('--learn_rounding', action='store_true', help='Whether to perform pre-computation-based weight-rounding policy learning')
    parser.add_argument('--blocksize', type=int, default=256, help='OPTQ block size')
    parser.add_argument('--clustersize', type=int, default=1, help='Number of columns per cluster (-1 disables GPTQ/1 conventional GPTQ)')

    # Hyperparams.
    parser.add_argument('--lr', type=float, default=0.015, help='learning rate for adaround training')
    parser.add_argument('--round_weight', type=float, default=1.0, help=' weight of rounding loss in adaround')
    parser.add_argument('--round_weight_qkv', type=float, default=1.5, help='rounding loss weight for QKV')
    parser.add_argument('--num_iters', type=int, default=2000, help='number of iterations for adaround training')
    
    parser.add_argument('--replace', type=float, default=1/2048, choices=[1.0, 1/2048], help='Value to be replaced for the Hessian diagonal elements corresponding to dead neurons')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')

    return parser.parse_args() 


def get_aespa_weight_quant_infos(args):
    qconfigs = {
        "w_bits": args.w_bits, "w_sym": args.w_sym
    }
    aespa_opts = {
        "block_v": args.block_v,
        'use_zfold': args.use_zfold, 
        "round_optim": {"learn_rounding": args.learn_rounding, "comp_method": args.comp_method, "loss_option": args.loss_option}
    }
    aespa_opts['round_optim']['order_option'] = args.order_option
    aespa_opts['round_optim']['blocksize'] = args.blocksize
    aespa_opts['round_optim']['clustersize'] = args.clustersize
    aespa_opts['round_optim']['groupsize'] = args.groupsize
    if args.learn_rounding:
        aespa_opts['round_optim']['lr'] = args.lr
        aespa_opts['round_optim']['round_weight'] = args.round_weight
        aespa_opts['round_optim']['round_weight_qkv'] = args.round_weight_qkv
        aespa_opts['round_optim']['num_iters'] = args.num_iters
        
    hyperparams = {"replace": args.replace, "percdamp": args.percdamp}
    
    return qconfigs, aespa_opts, hyperparams


def save_ppl_results(ppl_results, process_time, args):
    import os
    from pathlib import Path
    import csv

    data = [{**ppl_results, **{"time": process_time}}]
    output_dir = "results"
    if not os.path.exists(Path(output_dir)):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"aespa-{args.model_name}-w{args.w_bits}-{'sym' if args.w_sym else 'asym'}-{args.calib_data}_{args.nsamples}_{args.seqlen}_{args.seed}"
    if args.block_v:
        filename += "-block_v"
    if args.use_zfold:
        filename += "-zfold"
    if args.order_option != 'none':
        filename += "_order"
    if args.learn_rounding:
        filename += f"-learn_rounding-lr_{args.lr}_rw_{args.round_weight}_rwqkv_{args.round_weight_qkv}_niters_{args.num_iters}"
    filename += ".csv"
    with open(os.path.join(output_dir, filename), "w", newline='') as file:
        header = data[0].keys()
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def set_qmodel_dir(args):
    qmodel_dir = f"{args.model_name}-w{args.w_bits}-{'sym' if args.w_sym else 'asym'}-{args.calib_data}_{args.nsamples}_{args.seqlen}_{args.seed}"
    if args.block_v:
        qmodel_dir += "-block_v"
    if args.use_zfold:
        qmodel_dir += "-zfold"
    if args.act_order:
        qmodel_dir += "_act_order"
    if args.learn_rounding:
        qmodel_dir += f"-learn_rounding-lr_{args.lr}_rw_{args.round_weight}_rwqkv_{args.round_weight_qkv}_niters_{args.num_iters}"

    return qmodel_dir