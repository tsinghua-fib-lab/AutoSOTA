import os
import torch.nn as nn
from datautils import get_loaders
import torch
from tqdm import tqdm
import copy
from utils_models import find_layers
import json
import transformers
# Compatibility for lm-eval 0.4.9.2
if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = transformers.AutoModelForSeq2SeqLM

from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
from datautils import get_tokenizer
from lm_eval.utils import load_yaml_config
import torch.multiprocessing as mp
import numpy as np


@torch.no_grad()
def llm_eval(model, name_dataset, dset_dir, tokenizers_dir, arch, device, nsamples, is_train=True, seed=0, do_validation=False, is_val=False, percent_val=0.5, nsamples_val=None):
    print('Evaluating ...')
    if nsamples_val is None:
        if do_validation:
            multiplier_samples = 1/(1-percent_val)
        else:
            multiplier_samples = 1.0
        nsamples_val = int(multiplier_samples*nsamples) - nsamples
    # trainenc, testenc = get_llm_loaders(name_dataset, dset_dir, tokenizers_dir, nsamples=nsamples + nsamples_val, seqlen=model.seqlen, arch=arch, seed=seed)
    trainenc, testenc = get_loaders(name_dataset, nsamples=nsamples + nsamples_val, seed=seed, seqlen=model.seqlen, model=arch, cache_dir=dset_dir)
    
    if is_train:
        testenc = torch.hstack([x[0] for x in trainenc[:nsamples]])
    elif is_val:
        if do_validation:
            testenc = torch.hstack([x[0] for x in trainenc[nsamples:]])
        else:
            print("Error, do_validation must be True when is_val is True")
            import ipdb; ipdb.set_trace()
    else:
        testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "Llama" in arch:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(device)
        if "Llama-3" in arch or "Llama-2" in arch:
            model.model.rotary_emb = model.model.rotary_emb.to(device)

    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device="cpu"
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    class Catcher(torch.nn.Module):
        def __init__(self, module, arch):
            super().__init__()
            self.module = module
            self.arch = arch

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if "Llama-2" in self.arch or "Llama-3" in self.arch:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0], arch)
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(device)
        try:
            model(batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if model.model.norm is not None:
        model.model.norm = model.model.norm.cpu()
    if "Llama-3" in arch or "Llama-2" in arch:
        model.model.rotary_emb = model.model.rotary_emb.cpu()

    torch.cuda.empty_cache()

    attention_mask = cache['attention_mask']

    if "Llama-2" in arch or "Llama-3" in arch:
        position_ids = cache['position_ids']
    else:
        position_ids = None

    position_embeddings = None
    
    def run_decoder_layer(layer, hidden, attention_mask, position_ids=None, position_embeddings=None):
        if ("Llama-2" in arch or "Llama-3" in arch) and position_ids is not None:
            position_embeddings = model.model.rotary_emb(hidden, position_ids)

            return layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

        if position_ids is not None:
            return layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        return layer(
            hidden,
            attention_mask=attention_mask,
        )[0]

    outs = torch.zeros_like(inps)

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(device)

        for j in range(nsamples):
            hidden = inps[j].unsqueeze(0).to(device)

            outs[j] = run_decoder_layer(
                layer,
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            ).to("cpu")
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)
    if "Llama-3" in arch or "Llama-2" in arch:
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    testenc = testenc.to(device)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0).to(device)
        if "Llama" in arch:
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
        else:
            if model.model.decoder.final_layer_norm is not None:
                hidden_states = model.model.decoder.final_layer_norm(hidden_states)
            if model.model.decoder.project_out is not None:
                hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    model.config.use_cache = use_cache
    return ppl.item()

def lm_evaluate(model, tasks, num_fewshot, device, arch, tokenizers_dir, dset_dir, results_0_shot, d_completed_0_shot, check_integrity=False):
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model.to(device)
    tokenizer = get_tokenizer(arch, tokenizers_dir)
    wrapped_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto", max_batch_size=512)

    print(f"Selected Tasks: {tasks}")

    task_manager = TaskManager("INFO")
    task_manager.prefix_data_path = dset_dir
    task_list = tasks.split(",")
    task_names = task_manager.match_tasks(task_list)
    for task in [task for task in task_list if task not in task_names]:
        if os.path.isfile(task):
            config = load_yaml_config(task)
            task_names.append(config)
    task_missing = [
        task for task in task_list if task not in task_names and "*" not in task
    ]  # we don't want errors if a wildcard ("*") task name was used

    if task_missing:
        missing = ", ".join(task_missing)
        print(
            f"Tasks were not found: {missing}\n"
        )
        raise ValueError(
            f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
        )

    results = evaluator.simple_evaluate(
        model=wrapped_model,
        # model_args=args.model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size="auto",
        bootstrap_iters=0,
        device=device,
        check_integrity=check_integrity,
        task_manager=task_manager,
    )
    l_tasks = tasks.split(",")
    for name_task in l_tasks:
        results_0_shot[f"test_acc_{name_task}_0_shot"] = results["results"][name_task]['acc,none']
    model.to("cpu")
    d_completed_0_shot[int(device.split(":")[1])] = True
    return results

def lm_evaluate_parallel(model, tasks, num_fewshot, test_distributed, arch, tokenizers_dir, dset_dir, check_integrity=False):
    processes = []
    if test_distributed and torch.cuda.is_available() and torch.cuda.device_count()>1:
        mp.set_start_method('spawn', force=True)
        manager = mp.Manager()
        results_0_shot = manager.dict()
        d_completed_0_shot = manager.dict()
        world_size = torch.cuda.device_count()
        for rank in range(world_size):
            d_completed_0_shot[rank] = False
        l_tasks = tasks.replace(",winogrande", "").split(",")
        n_tasks_per_gpu = int(np.ceil(len(l_tasks)/(world_size-1)))
        for rank in range(world_size):
            if rank==world_size-1:
                str_sub_tasks = "winogrande"
            else:
                str_sub_tasks = ",".join(l_tasks[rank*n_tasks_per_gpu:(rank+1)*n_tasks_per_gpu])
            print("----------------------------------------------", flush=True)
            print(f"Tasks on gpu {rank}:", str_sub_tasks, flush=True)
            print("----------------------------------------------", flush=True)
            p = mp.Process(target=lm_evaluate, args=(model, str_sub_tasks, num_fewshot, f"cuda:{rank}", arch, tokenizers_dir, dset_dir, results_0_shot, d_completed_0_shot))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        eval_successful = True
        for rank in range(world_size):
            print(f"Pruning on cuda:{rank}: {d_completed_0_shot[rank]}", flush=True)
            eval_successful = eval_successful and d_completed_0_shot[rank]
        if not(eval_successful):
            import ipdb;ipdb.set_trace
        results_0_shot = dict(results_0_shot)
    else:
        results_0_shot = {}
        d_completed_0_shot = {}
        rank = 0
        world_size = 1
        lm_evaluate(model, tasks, num_fewshot, "cuda:0" if torch.cuda.is_available() else "cpu", arch, tokenizers_dir, dset_dir, results_0_shot, d_completed_0_shot)
        eval_successful = True
    return results_0_shot
