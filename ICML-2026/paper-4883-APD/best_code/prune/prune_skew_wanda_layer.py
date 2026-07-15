import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import math
from scipy.stats import skew
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
    parser.add_argument('--m', type=float, default=1.8)
    parser.add_argument('--load_in_8bit', action='store_true')

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

def prune(model, dataloader, nsamples, global_sparsity, ls_sparsity, prune_n=0, prune_m=0):
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
            'mean_sq': torch.zeros(layer.in_features),
            'nsamples': 0
        }

    def _hook_accumulate(layer, inp, out):
        inp_t = inp[0].detach().to(torch.float32)              
        if inp_t.dim() == 3:                                 
            inp_t = inp_t.reshape(-1, inp_t.shape[-1])     
        inp_t = inp_t.t()                            
        n_batch = inp_t.shape[1]

        state = activations[layer]
        n_prev, mean_sq = state["nsamples"], state["mean_sq"].to(device)

        mean_sq.mul_(n_prev / (n_prev + n_batch))
        mean_sq.add_(torch.norm(inp_t, p=2, dim=1).pow(2) / (n_prev + n_batch))

        state["nsamples"] += n_batch
        state["mean_sq"] = mean_sq.to('cpu')

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
        m = layer_re.match(name)
        if m and ls_sparsity:
            dec_idx = int(m.group(1))
            sparsity = ls_sparsity[dec_idx]
        else:
            sparsity = global_sparsity

        torch.cuda.empty_cache()
        state = activations[layer]
        act_l2 = torch.sqrt(state["mean_sq"].to(device).clamp_min(1e-8)) 
        w = layer.weight.data.to(torch.float32)
        scores = (w.abs() * act_l2.unsqueeze(0))
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


def get_llm(model_name, cache_dir="llm_weights", load_in_8bit=False):
    if args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            cache_dir=cache_dir, 
            low_cpu_mem_usage=True, 
            device_map="cuda:0",
            quantization_config=quantization_config
        )
    else:
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
    model = get_llm(args.model, cache_dir=args.cache, load_in_8bit=args.load_in_8bit)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device(args.device)

    sparsitys = allocation(model.model, args)

    # loading data
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed,seqlen=args.seqlen, tokenizer=tokenizer, path=args.path)
    inps, _ = prepare_calibration_input_(model, dataloader, device, args)


    prune(model.model, inps, args.nsamples, args.sparsity_ratio, sparsitys, args.prune_n, args.prune_m)


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



