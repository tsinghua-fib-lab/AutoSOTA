import argparse
import time
import torch
import torch.nn as nn
from utils_alps import *
import numpy as np
import json

import sys
import os
from datautils import *
from utils_models import *
from utils_evaluations import llm_eval, lm_evaluate_parallel
import time

DEV = torch.device('cuda:0')

@torch.no_grad()
def llama_sequential(model, dataloader, dev, nsamples, model_name):
    print('Starting ...')


    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(torch.nn.Module):
        def __init__(self, module, model_name):
            super().__init__()
            self.module = module
            self.model_name = model_name
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "Llama-2" in self.model_name or "Llama-3" in self.model_name:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    
    layers[0] = Catcher(layers[0], model_name)

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    
    attention_mask = cache["attention_mask"]

    if "Llama-2" in model_name or "Llama-3" in model_name:
        position_ids = cache['position_ids']
    else:
        position_ids = None
    
    def run_llama_layer(layer, hidden, attention_mask, position_ids):
        if position_ids is not None and hasattr(model.model, "rotary_emb"):
            position_embeddings = model.model.rotary_emb(hidden, position_ids)
            return layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
        return layer(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]

    seqlen = model.seqlen

    print('Ready.')

    tot_params = 0
    tot_nnz = 0

    for i in range(len(layers)):


        layer = layers[i].to(dev)
        full = find_layers(layer)


        sequential = [list(full.keys())]
        scd = {}
        print('----')

        for names in sequential:
            subset = {n: full[n] for n in names}


            for name in subset:
                
                if args.method == 'ALPS':
                    scd[name] = ALPS_prune(subset[name], nsamples=nsamples, seqlen=seqlen, uncertainty_set=args.uncertainty_set, type_H = args.type_H)
                else:
                    raise Exception

            def add_batch(name):
                def tmp(_, inp, out):
                    scd[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                hidden = inps[j].unsqueeze(0)
                outs[j] = run_llama_layer(layer, hidden, attention_mask, position_ids)
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name, flush=True)
               
                gamma = args.gamma

                if args.method == 'ALPS':
                    scd[name].ALPS_admm(sp=args.sp, nm_n=args.nm_n, nm_m=args.nm_m, rho=0.1, gamma=gamma)

                d1 = scd[name].layer.weight.data.shape[0]
                d2 = scd[name].layer.weight.data.shape[1]
                nnz = len( (scd[name].layer.weight.data.abs() > 0).nonzero(as_tuple=True)[0])
                tot_params += d1*d2
                tot_nnz += nnz



                scd[name].free()
        for j in range(args.nsamples):
            if "Llama-2" in model_name or "Llama-3" in model_name:
                hidden = inps[j].unsqueeze(0)
                outs[j] = run_llama_layer(layer, hidden, attention_mask, position_ids)
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        
        layers[i] = layer.cpu()
        del layer
        del scd 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    


    return tot_nnz/tot_params

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model', type=str,
    help='Llama model to load; pass `meta-llama/Llama-2-7b-hf`.'
)
parser.add_argument(
    '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
    help='Where to extract calibration data from.'
)

parser.add_argument(
    '--method', type=str, choices=['ALPS'],
    help='Method to run.'
)

parser.add_argument(
    '--sp', type=float, 
    help='Sparsity level'
)

parser.add_argument(
    '--seed',
    type=int, default=0, help='Seed for sampling the calibration data.'
)

parser.add_argument(
    '--nm_n',
    type=int, default=0, help='N for N:M'
)

parser.add_argument(
    '--nm_m',
    type=int, default=0, help='M for N:M'
)

parser.add_argument(
    '--nsamples', type=int, default=128,
    help='Number of calibration data samples.'
)

parser.add_argument(
    "--nsamples_val", type=int, default=128, help="Number of calibration data samples."
)

parser.add_argument(
    "--gamma",
    type=float,
    default=0.01,
    help="gamma as defined in RobOP.",
)

parser.add_argument(
    "--delete_previous_results", type = int, default=0, help="Whether to erase the previous results"
)

parser.add_argument(
    "--uncertainty_set", type = str, default="baseline", help="Either cte, baseline, trace, or eigh"
)

parser.add_argument(
    "--type_H", type = int, default=32, help="Numerical Precision of the Hessian"
)

parser.add_argument(
    "--checkpoints_dir", type = str, default="./model_checkpoints", help="Path to the model checkpoints"
)

parser.add_argument(
    "--dset_dir", type = str, default="./datasets", help="Path to the datasets"
)

parser.add_argument(
    "--save_dir", type = str, default="./results_RobOP", help="Saving Directory"
)


def create_path_results(gamma, uncertainty_set, model, sp, nm_n, nm_m, type_H, seed, dataset, nsamples, save_dir):
    to_add_type_H = f"_f{type_H}" if type_H!=32 else ""
    to_add_seed = f"_s{seed}" if seed!=0 else ""
    to_add_dataset = f"_{dataset}" if dataset!="c4" else ""
    to_add_nsamples = f"_{nsamples}" if nsamples!=128 else ""

    if nm_n == 0:
        path_results = f"{save_dir}/RobOP_ALPS_{gamma}_{uncertainty_set}{to_add_type_H}{to_add_dataset}{to_add_nsamples}{to_add_seed}/results_{model}_{sp}.json"
    else:
        path_results = f"{save_dir}/RobOP_ALPS_{gamma}_{uncertainty_set}{to_add_type_H}{to_add_dataset}{to_add_nsamples}{to_add_seed}/results_{model}_{nm_n}_{nm_m}.json"
    return path_results


if __name__ == '__main__':

    args = parser.parse_args()

    checkpoints_dir = args.checkpoints_dir
    dset_dir = args.dset_dir

    path_results = create_path_results(args.gamma, args.uncertainty_set, args.model, args.sp, args.nm_n, args.nm_m, args.type_H, args.seed, args.dataset, args.nsamples, args.save_dir)

    print("Path results:", path_results, flush=True)

    test_new_val = args.nsamples_val != 128
    test_existing_res = False
    to_add_nsamples_val = f"_{args.nsamples_val}" if args.nsamples_val != 128 else ""
    if os.path.exists(path_results):
        with open(path_results) as f:
            results_sparsity = json.load(f)
        test_existing_res = True
        test_new_val = test_new_val and f"val_ppl_{args.dataset}{to_add_nsamples_val}" not in results_sparsity

    if not(os.path.exists(path_results)) or args.delete_previous_results or test_new_val:
        path_results_end = "/".join(path_results.split("/")[:-1])
        results_sparsity = {}
        if os.path.exists(path_results):
            with open(path_results) as f:
                results_sparsity = json.load(f)
        model, criterion, l_layers_to_prune_per_block, l_blocks = get_model(args.model, 0, pretrained=True, checkpoints_dir=checkpoints_dir)
        model.to(DEV)
        model.eval()
        os.makedirs(path_results_end, exist_ok = True)
        test_attempt_success = False

        for attempt in range(10):
            if not test_attempt_success:
                try:
                    dataloader, testloader = get_loaders(
                        args.dataset, nsamples=2*args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, cache_dir = args.dset_dir
                    )
                    test_attempt_success = True
                except Exception as e:
                    print(e)
                    time.sleep(1.0)
                    pass
        if not test_attempt_success:
            print("Error, couldn't load the dataset")
            exit()

        dataloader = dataloader[:args.nsamples]

        tick = time.time()
        if args.method != 'Dense' and args.sp != 0.0:
            sp = llama_sequential(model, dataloader, DEV, nsamples=args.nsamples, model_name = args.model)
            for n, p in model.named_parameters():
                print(n, torch.mean((p == 0).float()))
                if 'down_proj' in n:
                    break
        else:
            sp = args.sp
        runtime = time.time() - tick
        print(time.time() - tick)
        sp = 1 - sp
        results_sparsity["time_pruning"] = time.time() - tick
        results_sparsity["Sparsity"] = sp

        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.config.output_attentions = True
            
        for eval_dataset in ["c4", "wikitext2", "ptb"]:
            if eval_dataset == args.dataset:
                if f"train_ppl_{eval_dataset}" not in results_sparsity:
                    train_ppl = llm_eval(model, eval_dataset, dset_dir, checkpoints_dir, args.model, DEV, args.nsamples, is_train=True, do_validation = True, percent_val = 0.5, seed = args.seed, nsamples_val = args.nsamples_val)
                    results_sparsity[f"train_ppl_{eval_dataset}"] = train_ppl
                else:
                    train_ppl = results_sparsity[f"train_ppl_{eval_dataset}"]
                if args.nm_n == 0:
                    print(f"Training perplexity on {eval_dataset} for sparsity = {args.sp}: {train_ppl:3f}")
                else:
                    print(f"Training perplexity on {eval_dataset} for sparsity = {args.nm_n}:{args.nm_m}: {train_ppl:3f}")
                if f"val_ppl_{eval_dataset}{to_add_nsamples_val}" not in results_sparsity:
                    val_ppl = llm_eval(model, eval_dataset, dset_dir, checkpoints_dir, args.model, DEV, args.nsamples, is_train=False, is_val = True, do_validation = True, percent_val = 0.5, seed = args.seed, nsamples_val = args.nsamples_val)
                    results_sparsity[f"val_ppl_{eval_dataset}{to_add_nsamples_val}"] = val_ppl
                else:
                    val_ppl = results_sparsity[f"val_ppl_{eval_dataset}{to_add_nsamples_val}"]
                if args.nm_n == 0:
                    print(f"Validation perplexity on {eval_dataset} for sparsity = {args.sp}: {val_ppl:3f}")
                else:
                    print(f"Validation perplexity on {eval_dataset} for sparsity = {args.nm_n}:{args.nm_m}: {val_ppl:3f}")
            else:
                if f"val_ppl_{eval_dataset}{to_add_nsamples_val}" not in results_sparsity:
                    val_ppl = llm_eval(model, eval_dataset, dset_dir, checkpoints_dir, args.model, DEV, args.nsamples_val, is_train=True, do_validation = False, percent_val = 0.0, seed = args.seed)
                    results_sparsity[f"val_ppl_{eval_dataset}{to_add_nsamples_val}"] = val_ppl
                else:
                    val_ppl = results_sparsity[f"val_ppl_{eval_dataset}{to_add_nsamples_val}"]
                if args.nm_n == 0:
                    print(f"Val perplexity on {eval_dataset} for sparsity = {args.sp}: {val_ppl:3f}")
                else:
                    print(f"Val perplexity on {eval_dataset} for sparsity = {args.nm_n}:{args.nm_m}: {val_ppl:3f}")

            if f"test_ppl_{eval_dataset}" not in results_sparsity:
                test_ppl = llm_eval(model, eval_dataset, dset_dir, checkpoints_dir, args.model, DEV, nsamples=1, is_train=False)
                results_sparsity[f"test_ppl_{eval_dataset}"] = test_ppl
            else:
                test_ppl = results_sparsity[f"test_ppl_{eval_dataset}"]
            if args.nm_n == 0:
                print(f"Test perplexity on {eval_dataset} for sparsity = {args.sp}: {test_ppl:3f}")
            else:
                print(f"Test perplexity on {eval_dataset} for sparsity = {args.nm_n}:{args.nm_m}: {test_ppl:3f}")

        if not test_existing_res:
            str_tasks = "openbookqa,winogrande,piqa"
            results_0_shot = lm_evaluate_parallel(model, str_tasks, 0, False, args.model, checkpoints_dir, dset_dir)
            l_tasks = str_tasks.split(",")
            for name_task in l_tasks:
                results_sparsity[f"test_acc_{name_task}_0_shot"] = results_0_shot[f"test_acc_{name_task}_0_shot"]
        model.config.use_cache = use_cache

        os.makedirs(path_results_end, exist_ok = True)
        with open(path_results, 'w') as f:
            json.dump(results_sparsity, f)

