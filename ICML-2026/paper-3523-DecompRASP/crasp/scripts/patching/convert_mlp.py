import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from itertools import product
import re
from copy import deepcopy
from collections import defaultdict
from transformers import GPT2LMHeadModel
from patching_data import get_tokenizer_and_dataset_for_task
from train_new_models import customCollator, customBCECollator, customTokenizer
from pruning_model import OptimalQueryBiasVectors, PruningModelWithHooksForQK
import matplotlib.pyplot as plt
from patching_utils import *
import time
import glob
from pathlib import Path
import json
import yaml
from convert_to_code import convert_keys_to_int

@torch.no_grad()
def trace_mlp(hooked_model, converted_mlp, path, input_ids, position_ids ,logger):
    model = hooked_model.model

    pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
    split_nodes = re.findall(pattern, path) # ['mlp-1', 'attn_output-1-1', 'wte']
    split_nodes = split_nodes[len(split_nodes)-1:0:-1]
    prod = None
    for i, node in enumerate(split_nodes): # outer to inner
        if node.startswith("attn_output"):
            _, layer, head = node.split("-")
            layer, head = int(layer), int(head)
            A = hooked_model.activations[f"attn_weights-{layer}"][:, head]
            prod = A @ prod
        elif node == "wte":
            prod = F.one_hot(input_ids, num_classes=model.config.vocab_size).float()
        elif node == "wpe":
            prod = F.one_hot(position_ids, num_classes=model.config.max_position_embeddings).float()
        elif node.startswith("mlp"):
            primitive, _ = converted_mlp["-".join(split_nodes[i::-1])]
            prod = get_mlp_primitives(prod, primitive=primitive)
        else:
            raise RuntimeError("node not recognized", node)
    
    assert prod.dim() == 3

    return prod

@torch.no_grad()
def trace_mlp_multi_source(hooked_model, converted_mlp, path, input_ids, position_ids ,logger):
    model = hooked_model.model
    config = hooked_model.config
    # 'mlp-1': ['attn_output-1-0-attn_output-0-0-wte', ''attn_output-1-0-mlp-0', 'wte']  'mlp-0': ['attn_output-0-0-wte', 'wte']

    if path.startswith("mlp"):
        prods = []
        for mlp_inp in config[int(path[4:])]["mlp"]:
            prods.append(trace_mlp_multi_source(hooked_model, converted_mlp, mlp_inp, input_ids, position_ids, logger))
        if path in converted_mlp:
            primitive, _ = converted_mlp[path]
            prod = get_mlp_primitives_multi_source(prods, primitive=primitive)
            return prod
        else:
            return prods
    
    else:
        pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
        split_nodes = re.findall(pattern, path) 
        split_nodes = split_nodes[len(split_nodes)-1::-1]
        prod = None
        for i, node in enumerate(split_nodes): # outer to inner
            if node.startswith("attn_output"):
                _, layer, head = node.split("-")
                layer, head = int(layer), int(head)
                A = hooked_model.activations[f"attn_weights-{layer}"][:, head]
                prod = A @ prod
            elif node == "wte":
                prod = F.one_hot(input_ids, num_classes=model.config.vocab_size).float()
            elif node == "wpe":
                prod = F.one_hot(position_ids, num_classes=model.config.max_position_embeddings).float()
            elif node.startswith("mlp"):
                prod = trace_mlp_multi_source(hooked_model, converted_mlp, node, input_ids, position_ids, logger)
                if isinstance(prod, list):
                    raise NotImplementedError("cannot trace through unconverted MLP, if you're running `visualize_mlp_multi_source`, nested branching not implemented as it's complex")
            else:
                raise RuntimeError("node not recognized", node)
        
        assert prod.dim() == 3

        return prod

def convert_mlp(
        hooked_model: PruningModelWithHooksForQK,
        oa_vecs: OptimalQueryBiasVectors,
        orig_model: GPT2LMHeadModel,
        dataloader: DataLoader,
        logger,
):
    primitive_set = [
        "no_op", "diff",
        ("sharpen", 2), ("sharpen", 3), ("sharpen", 5), "harden", "erase",
        ("ABbalance", 0.5), ("ABbalance", 0.05), ("ABbalance", 0.01),
        ("01balance", 0.5), ("01balance", 0.05), ("01balance", 0.01),
        ("forall", 0.95), ("forall", 0.9), ("forall", 0.85), ("forall", 0.8), ("forall", 0.75), ("forall", 0.7), 
        "exists", "equal"
        ]    # exists should be at the end
    converted_mlp = {}  # path: (primitive:str, C:FloatTensor)

    if hasattr(dataloader.dataset, "BCE"):
        use_BCE = dataloader.dataset.BCE
    else:
        use_BCE = False
    tokenizer = dataloader.dataset.tokenizer
    num_layers = len(hooked_model.model.transformer.h)
    d_model = hooked_model.model.config.hidden_size
    config = hooked_model.config
    device = hooked_model.device

    for layer in range(num_layers):
        for mlp_inp in config[layer]["mlp"]:
            path = f"mlp-{layer}-{mlp_inp}"
            logger("\nconverting", path)

            pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
            split_nodes = re.findall(pattern, mlp_inp)
            if not all(not node.startswith("mlp") or ("-".join(split_nodes[i:]) in converted_mlp) for i, node in enumerate(split_nodes)):
                logger("Unable to convert MLP: Dependency on unconverted MLP")
                continue

            all_inp_depend = []
            all_mlp_out = []
            for i, batch in enumerate(dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                bz, seq_len = batch["input_ids"].size()

                test_tensor = torch.randn(bz, seq_len, d_model, device=device, requires_grad=True)
                test_tensor_grad = None
                def capture_grad(grad):
                    nonlocal test_tensor_grad
                    test_tensor_grad = grad
                handle = test_tensor.register_hook(capture_grad)

                mlp_out = None
                def temp_hook(module, input, output):
                    nonlocal mlp_out
                    mlp_out = hooked_model.activations[path]
                    # test_tensor.data.copy_(mlp_out.data)
                    hooked_model.activations[path] = test_tensor
                handle2 = hooked_model.model.transformer.h[layer].mlp.register_forward_hook(temp_hook)
                
                logits = hooked_model(masks=torch.ones((1, 1), device=device), oa_vecs=oa_vecs, **batch).logits

                if not use_BCE:
                    loss = F.cross_entropy(logits[:, :-1].flatten(end_dim=1), labels[:, 1:].flatten())
                else:
                    mask = (batch["input_ids"] != tokenizer.pad_token_id) & (batch["input_ids"] != tokenizer.eos_token_id)    
                    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
                    loss = loss[mask].mean()
                loss.backward()

                masks = test_tensor_grad.abs().sum(dim=-1) > 1e-5

                handle.remove()
                handle2.remove()

                input_dependent = trace_mlp(hooked_model, converted_mlp, path, batch["input_ids"], batch["position_ids"], logger)

                all_inp_depend.append(input_dependent[masks])
                all_mlp_out.append(mlp_out[masks])
                
                if sum(item.size(0) for item in all_inp_depend) > 20000: # big enough to ensure full rank
                    break
            
            all_inp_depend = torch.cat(all_inp_depend, dim=0)
            all_mlp_out = torch.cat(all_mlp_out, dim=0)
            
            assert torch.allclose(all_inp_depend.sum(dim=1), torch.ones(all_inp_depend.size(0), device=device), atol=1e-3)

            with torch.no_grad():
                best_acc = 0
                best_primitive = None
                best_C = None

                temp_primitive_set = []
                for p in primitive_set:
                    if p == "exists":
                        for i in range(all_inp_depend.size(-1)):
                            temp_primitive_set.append((p, i))
                    elif p == "equal":
                        if all_inp_depend.size(-1) >= 6:
                            temp_primitive_set.append((p, list(range(all_inp_depend.size(-1)-4))))  # for 0,1 enough, but should add much more possibilities for a,b,c,d...
                    elif p[0] == "01balance":
                        if all_inp_depend.size(-1) == 6 and "attn_output" in path:
                            temp_primitive_set.append(p+(0,))
                    elif p[0] == "ABbalance":
                        if all_inp_depend.size(-1) == 8 and "attn_output" in path:
                            temp_primitive_set.append(p+(0,))
                    elif p == "diff":
                        if "attn_output" in path:
                            if all_inp_depend.size(-1) == 8:    # a, b
                                temp_primitive_set.append((p, 2, 3))
                            if all_inp_depend.size(-1) == 6:    # 0, 1
                                temp_primitive_set.append((p, 0, 1))
                    else:
                        temp_primitive_set.append(p)

                for primitive in temp_primitive_set:
                    if type(primitive) == tuple and (primitive[0] in ["exists", "equal"]) and best_acc >= 0.9:
                        break

                    primitive_func = partial(get_mlp_primitives, primitive=primitive)

                    Y = primitive_func(all_inp_depend)

                    try:
                        C = torch.linalg.pinv(Y) @ all_mlp_out  
                    except:
                        # torch._C._LinAlgError: linalg.svd: The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated singular values (error code: 155)
                        logger("primitive", primitive, "failed to compute inverse")
                        continue
                    # results = torch.linalg.lstsq(Y.cpu(), all_mlp_out.cpu(), rcond=1e-5, driver="gelsy")
                    # logger("rank", results.rank.item())   # very annoying and mysterious bug, rank is much smaller when running on remote machine than my laptop, even when rcond is fixed. so I'm instead using the less recommended matrix inverse
                    # C = results.solution.to(device)  # "For CUDA input, the only valid driver is ‘gels’, which assumes that A is full-rank."
                    assert not C.isnan().any().item()


                    recon_error = (Y @ C - all_mlp_out).pow(2).mean()
                    FVU = (recon_error / all_mlp_out.var(dim=0).mean()).item()

                    if FVU < 0.6:
                        total_num = 0
                        match_num = 0
                        correct_num = 0
                        for i, batch in enumerate(dataloader):
                            batch = {k: v.to(device) for k, v in batch.items()}
                            labels = batch.pop("labels")

                            target_logits = orig_model(**batch).logits

                            hooked_model(masks=torch.ones((1, 1), device=device), oa_vecs=oa_vecs, **batch)

                            input_dependent = trace_mlp(hooked_model, converted_mlp, path, batch["input_ids"], batch["position_ids"], logger)
                            Y = primitive_func(input_dependent)

                            recon_mlp_out = Y @ C.unsqueeze(0)

                            def temp_hook(module, input, output):
                                hooked_model.activations[path] = recon_mlp_out

                            handle = hooked_model.model.transformer.h[layer].mlp.register_forward_hook(temp_hook)
                            logits = hooked_model(masks=torch.ones((1, 1), device=device), oa_vecs=oa_vecs, **batch).logits
                            handle.remove()

                            if not use_BCE:
                                shift_target_logits = target_logits[:, :-1]
                                shift_logits = logits[:, :-1]
                                shift_labels = labels[:, 1:]
                                target_predictions = shift_target_logits.argmax(dim=-1)
                                predictions = shift_logits.argmax(dim=-1)

                                match = ((predictions == target_predictions) | (shift_labels == -100)).all(dim=1)
                                match_num += match.sum().item()
                                correct = ((predictions == shift_labels) | (shift_labels == -100)).all(dim=1)
                                correct_num += correct.sum().item()
                                total_num += correct.numel()
                            else:
                                mask = (batch["input_ids"] == tokenizer.pad_token_id) | (batch["input_ids"] == tokenizer.eos_token_id)
                                target_predictions = (target_logits > 0).long()
                                predictions = (logits > 0).long()

                                match = ((predictions == target_predictions).all(dim=-1) | mask).all(dim=1)
                                match_num += match.sum().item()

                                correct = ((predictions == labels).all(dim=-1) | mask).all(dim=1)
                                correct_num += correct.sum().item()
                                total_num += correct.numel()

                            if total_num > 2000:
                                break
                        
                        acc_match = match_num / total_num
                        acc_task = correct_num / total_num
                    
                    else:
                        acc_match, acc_task = 0, 0
                        
                    logger("primitive", primitive, f"\tFVU: {FVU:.4f}, Acc (match): {acc_match:.3f}, Acc (task): {acc_task:.3f}")
                    adjusted_acc = acc_match if primitive != "no_op" else acc_match + 0.01

                    if adjusted_acc > best_acc:
                        best_acc = adjusted_acc
                        best_primitive = primitive
                        best_C = C
                    
                    if primitive == "no_op" and acc_match > 0.92:
                        break
                
                logger("best primitive:", best_primitive)

                if best_acc < 0.9:
                    logger("Unable to convert MLP: Low Acc")
                    # return converted_mlp
                else:
                    converted_mlp[path] = (best_primitive, best_C)

    return converted_mlp


def convert_mlp_multi_source(
        hooked_model: PruningModelWithHooksForQK,
        oa_vecs: OptimalQueryBiasVectors,
        orig_model: GPT2LMHeadModel,
        dataloader: DataLoader,
        logger,
):
    # when split_mlp=False
    primitive_set = ["erase", "keep_one", "combine"]    # exists should be at the end
    converted_mlp = {}  # path: (primitive:str, C:FloatTensor)
    if hasattr(dataloader.dataset, "BCE"):
        use_BCE = dataloader.dataset.BCE
    else:
        use_BCE = False
    tokenizer = dataloader.dataset.tokenizer

    num_layers = len(hooked_model.model.transformer.h)
    d_model = hooked_model.model.config.hidden_size
    config = hooked_model.config
    device = hooked_model.device

    pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
    def check_dependency(config, layer):
        lis = []
        for mlp_inp in config[layer]["mlp"]:
            if "mlp" in mlp_inp:
                inp_layer = int(re.findall(pattern, mlp_inp)[-1][4:])
                lis.append(check_dependency(config, inp_layer))
            else:
                lis.append(True)
        return all(lis)
                

    for layer in range(num_layers):
        if len(config[layer]["mlp"]) == 0:
            continue
        path = f"mlp-{layer}"
        logger("\nconverting", path)
        
        if not check_dependency(config, layer):
            logger("Unable to convert MLP: Dependency on unconverted MLP")
            continue
        
        all_inp_depend = []
        all_mlp_out = []
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            bz, seq_len = batch["input_ids"].size()

            test_tensor = torch.randn(bz, seq_len, d_model, device=device, requires_grad=True)
            test_tensor_grad = None
            def capture_grad(grad):
                nonlocal test_tensor_grad
                test_tensor_grad = grad
            handle = test_tensor.register_hook(capture_grad)

            mlp_out = None
            def temp_hook(module, input, output):
                nonlocal mlp_out
                mlp_out = hooked_model.activations[path]
                hooked_model.activations[path] = test_tensor
            handle2 = hooked_model.model.transformer.h[layer].mlp.register_forward_hook(temp_hook)
            
            logits = hooked_model(masks=torch.ones((1, 1), device=device), oa_vecs=oa_vecs, **batch).logits

            if not use_BCE:
                loss = F.cross_entropy(logits[:, :-1].flatten(end_dim=1), labels[:, 1:].flatten())
            else:
                mask = (batch["input_ids"] != tokenizer.pad_token_id) & (batch["input_ids"] != tokenizer.eos_token_id)    
                loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
                loss = loss[mask].mean()
            loss.backward()

            masks = test_tensor_grad.abs().sum(dim=-1) > 1e-5

            handle.remove()
            handle2.remove()

            input_dependents = trace_mlp_multi_source(hooked_model, converted_mlp, path, batch["input_ids"], batch["position_ids"], logger)

            all_inp_depend.append([input_dependent[masks] for input_dependent in input_dependents])
            all_mlp_out.append(mlp_out[masks])

            if sum(item.size(0) for item in all_mlp_out) > 200000: # big enough to ensure full rank, for unique_bigram_copy, this matters a lot
                break
        
        all_inp_depend = [torch.cat(lis, dim=0) for lis in zip(*all_inp_depend)]
        all_mlp_out = torch.cat(all_mlp_out, dim=0)

        assert all(torch.allclose(item.sum(dim=1), torch.ones(item.size(0), device=device), atol=1e-3) for item in all_inp_depend)

        with torch.no_grad():
            best_acc = 0
            best_primitive = None
            best_C = None

            temp_primitive_set = []
            for p in primitive_set:
                if p == "keep_one":
                    for i in range(len(all_inp_depend)):
                        temp_primitive_set.append((p, i))
                elif p == "combine" and len(all_inp_depend) == 1:
                    pass
                else:
                    temp_primitive_set.append(p)

            for primitive in temp_primitive_set:

                primitive_func = partial(get_mlp_primitives_multi_source, primitive=primitive)

                Y = primitive_func(all_inp_depend)
                logger("Y dim", Y.size(1))

                if Y.size(1) > 10000:
                    logger("primitive", primitive, "Y dim is too large, computing C takes long time")
                    continue
                s_time = time.time()
                try:
                    C = torch.linalg.pinv(Y) @ all_mlp_out
                except:
                    logger("primitive", primitive, "error occurs when computing C, skip")
                    continue
                
                assert not C.isnan().any().item()
                logger(f"C calculation completed {time.time() - s_time:.2f}s")

                recon_error = (Y @ C - all_mlp_out).pow(2).mean()
                FVU = (recon_error / all_mlp_out.var(dim=0).mean()).item()

                if FVU < 0.6:
                    total_num = 0
                    match_num = 0
                    correct_num = 0
                    for i, batch in enumerate(dataloader):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        labels = batch.pop("labels")

                        target_logits = orig_model(**batch).logits

                        hooked_model(masks=torch.ones((1, 1), device=device), oa_vecs=oa_vecs, **batch)

                        input_dependent = trace_mlp_multi_source(hooked_model, converted_mlp, path, batch["input_ids"], batch["position_ids"], logger)
                        Y = primitive_func(input_dependent)

                        recon_mlp_out = Y @ C.unsqueeze(0)

                        def temp_hook(module, input, output):
                            hooked_model.activations[path] = recon_mlp_out

                        handle = hooked_model.model.transformer.h[layer].mlp.register_forward_hook(temp_hook)
                        logits = hooked_model(masks=torch.ones((1, 1), device=device), oa_vecs=oa_vecs, **batch).logits
                        handle.remove()

                        if not use_BCE:
                            shift_target_logits = target_logits[:, :-1]
                            shift_logits = logits[:, :-1]
                            shift_labels = labels[:, 1:]
                            target_predictions = shift_target_logits.argmax(dim=-1)
                            predictions = shift_logits.argmax(dim=-1)

                            match = ((predictions == target_predictions) | (shift_labels == -100)).all(dim=1)
                            match_num += match.sum().item()
                            correct = ((predictions == shift_labels) | (shift_labels == -100)).all(dim=1)
                            correct_num += correct.sum().item()
                            total_num += correct.numel()
                        else:
                            mask = (batch["input_ids"] == tokenizer.pad_token_id) | (batch["input_ids"] == tokenizer.eos_token_id)
                            target_predictions = (target_logits > 0).long()
                            predictions = (logits > 0).long()

                            match = ((predictions == target_predictions).all(dim=-1) | mask).all(dim=1)
                            match_num += match.sum().item()

                            correct = ((predictions == labels).all(dim=-1) | mask).all(dim=1)
                            correct_num += correct.sum().item()
                            total_num += correct.numel()

                        if total_num > 2000:
                            break
                    
                    acc_match = match_num / total_num
                    acc_task = correct_num / total_num
                
                else:
                    acc_match, acc_task = 0, 0
                    
                logger("primitive", primitive, f"\tFVU: {FVU:.4f}, Acc (match): {acc_match:.3f}, Acc (task): {acc_task:.3f}")

                if acc_match > best_acc:
                    best_acc = acc_match
                    best_primitive = primitive
                    best_C = C
            
            logger("best primitive:", best_primitive)

            if best_acc < 0.9:
                logger("ERROR: unable to convert MLP")
                return converted_mlp
            
            converted_mlp[path] = (best_primitive, best_C)

    return converted_mlp