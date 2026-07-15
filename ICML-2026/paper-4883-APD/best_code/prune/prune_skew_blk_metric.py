import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.data import get_loaders
from lib.util import check_sparsity, prepare_calibration_input_
from lib.eval import eval_ppl_wikitext
import argparse
import re
from peft import PeftModel
import torch


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparsity_ratio', type=float, default=0.5)
    parser.add_argument('--prune_n', type=int, default=0)
    parser.add_argument('--prune_m', type=int, default=0)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str)
    parser.add_argument('--cache', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--path', type=str, default=None, help='dataset path')
    parser.add_argument('--lora', action='store_true', help='use lora adapter or not')
    parser.add_argument('--lora_weights', type=str, default='', help='path of adapter weights')
    parser.add_argument('--save', action='store_true', help='save pruned model or not')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--m', type=float, default=2.0)

    return parser.parse_args()

def tensor_skew(x, eps=1e-8):
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)
    mu  = x.mean()
    m3  = torch.mean((x - mu) ** 3)
    std = x.std(unbiased=False)
    return torch.nan_to_num(m3 / (std**3 + eps))

def allocation(activations:dict, args):
    names, skews = [], []

    for layer in activations.keys():
        if isinstance(layer, nn.Linear):
            metric = activations[layer]['scores']
            s = tensor_skew(metric).detach().cpu().item()
            names.append(activations[layer]['name'])
            skews.append(s)

    if not skews:
        return {}
    
    print(skews)
    skews = np.array(skews, dtype=np.float64)

    skews = skews - np.mean(skews)
    delta_skewnss = skews.max() - skews.min()
    print(f'delta: {delta_skewnss}')
    skews = skews / (delta_skewnss + 1e-8)

    beta = math.log(args.m) / (skews.max() - skews.min()) * args.sparsity_ratio

    exp_term = np.exp(skews * beta)
    weights = 1 - exp_term / (exp_term.sum() + 1e-6) * len(skews) * (1 - args.sparsity_ratio)

    weights = {n: float(w) for n, w in zip(names, weights.tolist())}
    print(weights)
    return weights

def prune(model, dataloader, nsamples, global_sparsity, sparsitys, prune_n=0, prune_m=0):
    model.eval()
    device = next(model.parameters()).device


    name_to_linear = {name: mod
                      for name, mod in model.named_modules()
                      if isinstance(mod, nn.Linear)}
    linears = list(name_to_linear.values()) 
    activations = {} 
    hooks = []
    
    for layer in linears:
        activations[layer] = {
            'sum_abs': torch.zeros(layer.in_features), 
            'sum_sq': torch.zeros(layer.in_features),
        }

    def _hook_accumulate(layer, inp, out):
        inp_tensor = inp[0].detach()
        
        abs_act = inp_tensor.abs().mean(dim=1)
        
        activations[layer]['sum_abs'] += abs_act.sum(dim=0).to('cpu')
        activations[layer]['sum_sq'] += (inp_tensor.abs()**2).mean(dim=1).sum(dim=0).to('cpu')

    for layer in linears:
        hooks.append(layer.register_forward_hook(_hook_accumulate))

    print(f"Starting calibration with {nsamples} samples in batches...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"  Processing batch {i+1}/{len(dataloader)}...")
            model(batch.unsqueeze(0).to(device))
    print("Calibration finished.")

    for h in hooks:
        h.remove()

    layer_re = re.compile(r'layers\.(\d+)\.')

    for idx, (name, layer) in enumerate(name_to_linear.items()):
        if 'lm_head' in name:
            continue

        torch.cuda.empty_cache()
        final_mean_abs = activations[layer]['sum_abs'] / nsamples
        final_mean_sq = activations[layer]['sum_sq'] / nsamples
        
        am = final_mean_abs.to(device).clamp_min(1e-8)
        ar = torch.sqrt(final_mean_sq).to(device).clamp_min(1e-8)
        
        w = layer.weight.data.to(torch.float32)
        act_sum = torch.sqrt(am + ar)
        row_norms = torch.norm(w, p=1, dim=1, keepdim=True) + 1e-8
        scores = (w.abs() * act_sum.unsqueeze(0)) / row_norms
        activations[layer]['scores'] = scores
        activations[layer]['name'] = name

    sparsitys = allocation(activations, args)

    for idx, (name, layer) in enumerate(name_to_linear.items()):
        if 'lm_head' in name:
            continue
        if name in sparsitys.keys():
            sparsity = sparsitys[name]
        else:
            sparsity = global_sparsity

        torch.cuda.empty_cache()
        final_mean_abs = activations[layer]['sum_abs'] / nsamples
        final_mean_sq = activations[layer]['sum_sq'] / nsamples
        
        am = final_mean_abs.to(device).clamp_min(1e-8)
        ar = torch.sqrt(final_mean_sq).to(device).clamp_min(1e-8)
        
        w = layer.weight.data.to(torch.float32)
        act_sum = torch.sqrt(am + ar)
        row_norms = torch.norm(w, p=1, dim=1, keepdim=True) + 1e-8
        scores = (w.abs() * act_sum.unsqueeze(0)) / row_norms * 1e3

        W_mask = (torch.zeros_like(scores) == 1)

        if prune_n != 0:
            for ii in range(scores.shape[1]):
                if ii % prune_m == 0:
                    tmp = scores[:, ii:(ii + prune_m)].float()
                    W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
        else:
            sort_res = torch.sort(scores, dim=-1, stable=True)
            indices = sort_res[1][:, :int(scores.shape[1] * sparsity)]
            W_mask.scatter_(1, indices, True)

        layer.weight.data[W_mask] = 0
    return model


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="cuda:0",
    )
    
    if args.lora:
        model = PeftModel.from_pretrained(model,args.lora_weights,torch_dtype=torch.float16)
    model.seqlen = model.config.max_position_embeddings 
    return model

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = get_llm(args.model, cache_dir=args.cache)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device(args.device)


    # loading data
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed,seqlen=args.seqlen, tokenizer=tokenizer, path=args.path)
    inps, _ = prepare_calibration_input_(model, dataloader, device, args)


    prune(model.model, inps, args.nsamples, args.sparsity_ratio, args.prune_n, args.prune_m)


    # check sparsity ratio
    if not args.lora:
        sparsity_ratio = check_sparsity(model)
        print('sparsity ratio:', sparsity_ratio)


    torch.cuda.empty_cache()
    
    _, testloader = get_loaders(args.dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer, path=args.path)
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, args.device)
    
    if args.save:
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)

    return ppl_test



if __name__ == '__main__':
    args = config()

    main(args)



