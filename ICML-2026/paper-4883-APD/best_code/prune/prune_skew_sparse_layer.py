from collections import defaultdict
import sys
import os
import torch.nn as nn
import numpy as np
import transformers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.data import get_loaders
from lib.util import check_sparsity, prepare_calibration_input, prepare_calibration_input_
from lib.eval import eval_ppl_wikitext
import argparse
import re
from peft import PeftModel, PeftConfig 
import torch

def config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sparsity_ratio', type=float, default=0.5)
    parser.add_argument('--prune_n', type=int, default=0)
    parser.add_argument('--prune_m', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--blocksize', type=int, default=128)
    parser.add_argument('--damping', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str)
    parser.add_argument('--cache', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--path', type=str, default=None, help='dataset path')
    parser.add_argument('--lora', action='store_true', help='use lora adapter or not')
    parser.add_argument('--lora_weights', type=str, default='', help='path of adapter weights')
    parser.add_argument('--save', action='store_true', help='save pruned model or not')
    parser.add_argument('--cpu_store', action='store_true', help='Move H_diag to cpu to save gpu memory')
    parser.add_argument('--m', type=float, default=1.8)

    return parser.parse_args()

def tensor_skew(x, eps=1e-8):
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)
    mu  = x.mean()
    m3  = torch.mean((x - mu) ** 3)
    std = x.std(unbiased=False)
    return m3 / (std**3 + eps)

def allocation(model, args):
    skewness = defaultdict(list)
    layer_re = re.compile(r'layers\.(\d+)\.')

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            m = layer_re.search(name)
            if m:
                w = module.weight.abs().detach().to(torch.float32)
                s = tensor_skew(w)
                skewness[int(m.group(1))].append(s.detach().cpu().item())

    layer_idx = sorted(skewness.keys())
    skewness = np.array([np.mean(skewness[i]) for i in layer_idx])
    print(skewness)
    skewness = skewness - np.mean(skewness)
    delta_skewness = skewness.max() - skewness.min()
    skewness = skewness / (delta_skewness + 1e-8)

    beta = math.log(args.m) / (max(skewness) - min(skewness)) * args.sparsity_ratio

    weights = (1 - np.exp(skewness * beta) / (np.sum(np.exp(skewness * beta)) + 1e-8) * len(skewness) * (1 - args.sparsity_ratio)).tolist()

    return weights

def prune(model, dataloader, nsamples, sparsity, sparsitys, prune_n=0, prune_m=0, blocksize=128, damping=1e-5, cpu_store=False):
    model.eval()
    device = next(model.parameters()).device

    linears = {name: mod
                      for name, mod in model.named_modules()
                      if isinstance(mod, nn.Linear)}
    activations = {} 
    hooks = []
    
    for name, layer in linears.items():
        if cpu_store:
            activations[layer] = { # You can move H_diag to the GPU, but this will more than double the GPU memory usage
                'H_diag': torch.zeros((layer.in_features, layer.in_features), device='cpu'),
                'nsamples': 0
            }
        else:
            activations[layer] = {
                'H_diag': torch.zeros((layer.in_features, layer.in_features), device=device),
                'nsamples': 0
            }

    def _hook_accumulate(layer, inp, out):
        x = inp[0].detach().to(torch.float32)
        if isinstance(layer, nn.Linear) or isinstance(layer, transformers.Conv1D):
            if len(x.shape) == 3:
                x = x.reshape((-1, x.shape[-1]))
            x = x.t()

        state = activations[layer]
        if cpu_store:
            state['H_diag'] = (state['H_diag'] * (state['nsamples'] / (x.shape[0] + state['nsamples']))).to('cpu')
        else:
            state['H_diag'] *= state['nsamples'] / (x.shape[0] + state['nsamples'])
        state['nsamples'] += x.shape[0]
        x = math.sqrt(2 / state['nsamples']) * x.float()
        if cpu_store:
            state['H_diag'] += x.matmul(x.t()).to('cpu')
        else:
            state['H_diag'] += x.matmul(x.t())

    for name, layer in linears.items():
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
    for idx, (name, layer) in enumerate(linears.items()):
        m = layer_re.match(name)
        if m:
            layer_idx = int(m.group(1))
            layer_ratio = sparsitys[layer_idx]
        else:
            layer_ratio = sparsity

        w = layer.weight.data.to(torch.float32).clone()
        state = activations[layer]
        h = state['H_diag'].to(device)
        dead = torch.diag(h) == 0
        h[dead, dead] = 1
        
        losses = torch.zeros(layer.out_features, device=device)

        damp = damping * torch.mean(torch.diag(h))
        diag = torch.arange(layer.in_features, device=device)
        h[diag, diag] += damp
        h = torch.linalg.cholesky(h)
        h = torch.cholesky_inverse(h)
        h = torch.linalg.cholesky(h, upper=True)
        Hinv = h

        mask = None

        for i1 in range(0, layer.in_features, blocksize):
            i2 = min(i1 + blocksize, layer.in_features)
            count = i2 - i1

            W1 = w[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]



            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * layer_ratio)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                weight = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = weight.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (weight - q) ** 2 / d ** 2

                err1 = (weight - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            w[:, i1:i2] = Q1
            losses += torch.sum(Losses1, 1) / 2

            w[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        layer.weight.data = w.reshape(layer.weight.shape).to(layer.weight.data.dtype)
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
    model = get_llm(args.model, cache_dir=args.cache)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device(args.device)

    sparsitys = allocation(model.model, args)

    # loading data
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed,seqlen=args.seqlen, tokenizer=tokenizer, path=args.path)
    inps, _ = prepare_calibration_input_(model, dataloader, device, args)


    prune(model.model, inps, args.nsamples, args.sparsity_ratio, sparsitys, prune_n=args.prune_n, prune_m=args.prune_m, cpu_store=args.cpu_store, blocksize=args.blocksize, damping=args.damping)


    if not args.lora:
        sparsity_ratio = check_sparsity(model)
        print('sparsity ratio:', sparsity_ratio)

    torch.cuda.empty_cache()
    
    _, testloader = get_loaders(args.dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer, path=args.path)
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, device=args.device)
    
    if args.save:
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)

    return ppl_test



if __name__ == '__main__':
    args = config()
    main(args)