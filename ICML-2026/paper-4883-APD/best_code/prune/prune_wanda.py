import sys
import os
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str)
    parser.add_argument('--cache', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--path', type=str, default=None, help='dataset path')
    parser.add_argument('--lora', action='store_true', help='use lora adapter or not')
    parser.add_argument('--lora_weights', type=str, default='', help='path of adapter weights')
    parser.add_argument('--save', action='store_true', help='save pruned model or not')

    return parser.parse_args()

def prune(model, dataloader, nsamples, sparsity, prune_n=0, prune_m=0):
    model.eval()
    device = next(model.parameters()).device

    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    activations = {} 
    hooks = []
    
    for layer in linears:
        activations[layer] = {
            'mean_sq': torch.zeros(layer.in_features, device=device),
            'nsamples': 0
        }

    def _hook_accumulate(layer, inp, out):
        inp_t = inp[0].detach().to(torch.float32)              
        if inp_t.dim() == 3:                                 
            inp_t = inp_t.reshape(-1, inp_t.shape[-1])     
        inp_t = inp_t.t()                            
        n_batch = inp_t.shape[1]

        state = activations[layer]
        n_prev, mean_sq = state["nsamples"], state["mean_sq"]

        mean_sq.mul_(n_prev / (n_prev + n_batch))
        mean_sq.add_(torch.norm(inp_t, p=2, dim=1).pow(2) / (n_prev + n_batch))

        state["nsamples"] += n_batch
        state["mean_sq"] = mean_sq

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

    for idx, layer in enumerate(linears):
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

    # loading data
    dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed,seqlen=args.seqlen, tokenizer=tokenizer, path=args.path)
    inps, _ = prepare_calibration_input_(model, dataloader, device, args)


    prune(model.model, inps, args.nsamples, args.sparsity_ratio, args.prune_n, args.prune_m)


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
