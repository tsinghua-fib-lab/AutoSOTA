import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from itertools import product
import re
import json
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from transformers import GPT2LMHeadModel
from patching_data import get_tokenizer_and_dataset_for_task
from train_new_models import customCollator
from pruning_model import OptimalQueryBiasVectors, PruningModelWithHooksForQK
import matplotlib.pyplot as plt
from convert_mlp import get_mlp_primitives, get_mlp_primitives_multi_source, trace_mlp, trace_mlp_multi_source
from sklearn.cluster import KMeans


def get_QK_for_head(model: GPT2LMHeadModel, layer_idx, qkv_type, head_idx):
    attn_layer = model.transformer.h[layer_idx].attn
    w_matrix = attn_layer.c_attn.weight.data
    k_offset = attn_layer.embed_dim
    head_dim = attn_layer.head_dim

    W_q = w_matrix[:, head_idx*head_dim : (head_idx+1)*head_dim].clone()
    W_k = w_matrix[:, k_offset+head_idx*head_dim : k_offset+(head_idx+1)*head_dim].clone()  # W_k for head i

    if qkv_type == "q":
        return W_q
    elif qkv_type == "k":
        return W_k      # d_model, d_head

def get_OV_for_head(model: GPT2LMHeadModel, layer_idx, head_idx):
    attn_layer = model.transformer.h[layer_idx].attn
    w_matrix = attn_layer.c_attn.weight.data
    v_offset = attn_layer.embed_dim * 2
    head_dim = attn_layer.head_dim

    W_v = w_matrix[:, v_offset+head_idx*head_dim : v_offset+(head_idx+1)*head_dim].clone()  # d_model, d_head
    W_o = attn_layer.c_proj.weight.data[head_idx*head_dim : (head_idx+1)*head_dim].clone()   # d_head, d_model

    return W_v, W_o # already transposed

def get_LN_matrix_for_node(model: GPT2LMHeadModel, oa_vecs: OptimalQueryBiasVectors, layer_idx, qkv_type, head_idx, activation_name=None):
    if qkv_type == "v":
        assert activation_name is not None
        LN_var = oa_vecs.LN_var.data[oa_vecs.to_LN_idx[(layer_idx, "v", head_idx, activation_name)]].exp()
        denom = (LN_var + model.transformer.h[layer_idx].ln_1.eps).sqrt()
        gamma = model.transformer.h[layer_idx].ln_1.weight.data
    elif qkv_type == "mlp":
        assert activation_name is not None
        LN_var = oa_vecs.LN_var.data[oa_vecs.to_LN_idx[(layer_idx, "mlp", activation_name)]].exp()
        denom = (LN_var + model.transformer.h[layer_idx].ln_2.eps).sqrt()
        gamma = model.transformer.h[layer_idx].ln_2.weight.data
    elif qkv_type == "lm_head":
        LN_var = oa_vecs.LN_var.data[oa_vecs.to_LN_idx[("lm_head",)]].exp()
        denom = (LN_var + model.transformer.ln_f.eps).sqrt()
        gamma = model.transformer.ln_f.weight.data
    else:
        LN_var = oa_vecs.LN_var.data[oa_vecs.to_LN_idx[(layer_idx, qkv_type, head_idx)]].exp()
        denom = (LN_var + model.transformer.h[layer_idx].ln_1.eps).sqrt()
        gamma = model.transformer.h[layer_idx].ln_1.weight.data

    d_model = model.config.hidden_size
    mean_op = torch.eye(d_model) - torch.ones(d_model, d_model) / d_model

    W_ln = mean_op.to(gamma.device) @ torch.diag(gamma) / denom.to(gamma.device)
    return W_ln  # already transposed

def get_attn_weights_for_head(hooked_model: PruningModelWithHooksForQK, layer_idx, head_idx):
    return hooked_model.activations[f"attn_weights-{layer_idx}"][:, head_idx]

def get_product_for_one_side(hooked_model, oa_vecs: OptimalQueryBiasVectors, converted_mlp, path, input_ids, position_ids):
    model = hooked_model.model
    d_model = model.config.hidden_size

    pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
    split_nodes = re.findall(pattern, path) # ['attn_output-2-0', 'attn_output-1-1', 'wte']
    split_nodes = split_nodes[::-1].copy()
    dep_prod = None
    indep_prod = None
    for i, node in enumerate(split_nodes):
        if node.startswith("attn_output"):
            _, layer, head = node.split("-")
            layer, head = int(layer), int(head)
            A = get_attn_weights_for_head(hooked_model, layer, head).squeeze(0)
            dep_prod = A @ dep_prod

            past_path = "-".join(split_nodes[i-1::-1])
            W_ln = get_LN_matrix_for_node(model, oa_vecs, layer, "v", head, past_path)
            W_v, W_o = get_OV_for_head(model, layer, head)
            indep_prod = indep_prod @ W_ln @ W_v @ W_o

        elif node == "wte":
            dep_prod = F.one_hot(input_ids, num_classes=model.config.vocab_size).float()
            indep_prod = model.transformer.wte.weight.data     # vocab_size, d_model
        elif node == "wpe":
            dep_prod = F.one_hot(position_ids, num_classes=model.config.max_position_embeddings).float()
            indep_prod = model.transformer.wpe.weight.data

        elif node.startswith("mlp"):
            layer = int(node.split("-")[1])
            node_path = "-".join(split_nodes[i::-1])
            if node_path in converted_mlp:
                primitive, C = converted_mlp[node_path]
                dep_prod = get_mlp_primitives(dep_prod, primitive)
                indep_prod = C
            else:
                dep_prod = hooked_model.activations[node_path].squeeze(0)
                indep_prod = torch.eye(d_model).to(model.device)
        else:
            raise RuntimeError("node not recognized", node)
    
    # assert indep_prod.dim() == 2 and dep_prod.dim() == 2

    return dep_prod, indep_prod

# todo recursively for mlp multi source
def get_product_for_one_side_multi_source(hooked_model, oa_vecs: OptimalQueryBiasVectors, converted_mlp, path, input_ids, position_ids):
    model = hooked_model.model
    d_model = model.config.hidden_size
    config = hooked_model.config
    # 'mlp-1': ['attn_output-1-0-attn_output-0-0-wte', ''attn_output-1-0-mlp-0', 'wte']  'mlp-0': ['attn_output-0-0-wte', 'wte']

    if path.startswith("mlp"):
        if path in converted_mlp:
            prods = []
            for mlp_inp in config[int(path[4:])]["mlp"]:
                prods.append(get_product_for_one_side_multi_source(hooked_model, oa_vecs, converted_mlp, mlp_inp, input_ids, position_ids)[0])
            primitive, C = converted_mlp[path]
            prod = get_mlp_primitives_multi_source(prods, primitive=primitive)
            return prod, C
        else:
            dep_prod = hooked_model.activations[path].squeeze(0)
            indep_prod = torch.eye(d_model).to(model.device)
            return dep_prod, indep_prod

    else:  
        pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
        split_nodes = re.findall(pattern, path) # ['attn_output-2-0', 'attn_output-1-1', 'wte']
        dep_prod = None
        indep_prod = None
        for i, node in enumerate(split_nodes):
            if node.startswith("attn_output"):
                _, layer, head = node.split("-")
                layer, head = int(layer), int(head)
                A = get_attn_weights_for_head(hooked_model, layer, head).squeeze(0)
                if dep_prod is not None:
                    dep_prod = dep_prod @ A
                else:
                    dep_prod = A

                past_path = "-".join(split_nodes[i+1:])
                W_ln = get_LN_matrix_for_node(model, oa_vecs, layer, "v", head, past_path)
                W_v, W_o = get_OV_for_head(model, layer, head)
                if indep_prod is not None:
                    indep_prod = W_ln @ W_v @ W_o @ indep_prod
                else:
                    indep_prod = W_ln @ W_v @ W_o

            elif node == "wte":
                t = F.one_hot(input_ids, num_classes=model.config.vocab_size).float()
                if dep_prod is not None:
                    dep_prod = dep_prod @ t
                else:
                    dep_prod = t
                if indep_prod is not None:
                    indep_prod = model.transformer.wte.weight.data @ indep_prod
                else:
                    indep_prod = model.transformer.wte.weight.data     # vocab_size, d_model
            elif node == "wpe":
                p = F.one_hot(position_ids, num_classes=model.config.max_position_embeddings).float()
                if dep_prod is not None:
                    dep_prod = dep_prod @ p
                else:
                    dep_prod = p
                if indep_prod is not None:
                    indep_prod = model.transformer.wpe.weight.data @ indep_prod
                else:
                    indep_prod = model.transformer.wpe.weight.data

            elif node.startswith("mlp"):
                dep_p, indep_p = get_product_for_one_side_multi_source(hooked_model, oa_vecs, converted_mlp, node, input_ids, position_ids)
                dep_prod = dep_prod @ dep_p
                indep_prod = indep_p @ indep_prod
            else:
                raise RuntimeError("node not recognized", node)
        
        # assert indep_prod.dim() == 2 and dep_prod.dim() == 2

        return dep_prod, indep_prod

def get_product_for_one_side_for_head(hooked_model, oa_vecs: OptimalQueryBiasVectors, converted_mlp, attn_layer_idx, attn_head_idx, qk_type, path, input_ids, position_ids):
    get_product_func = get_product_for_one_side if hasattr(oa_vecs, "MLPs") else get_product_for_one_side_multi_source
    dep_prod, indep_prod = get_product_func(hooked_model, oa_vecs, converted_mlp, path, input_ids, position_ids)
    
    W_ln = get_LN_matrix_for_node(hooked_model.model, oa_vecs, attn_layer_idx, qk_type, attn_head_idx)
    W_q_or_k = get_QK_for_head(hooked_model.model, attn_layer_idx, qk_type, attn_head_idx)
    indep_prod = indep_prod @ W_ln @ W_q_or_k

    return dep_prod, indep_prod    # all already transposed

def get_product_for_one_side_for_unembed(hooked_model, oa_vecs: OptimalQueryBiasVectors, converted_mlp, path, input_ids, position_ids):
    
    if path != "vocab_bias":
        get_product_func = get_product_for_one_side if hasattr(oa_vecs, "MLPs") else get_product_for_one_side_multi_source
        dep_prod, indep_prod = get_product_func(hooked_model, oa_vecs, converted_mlp, path, input_ids, position_ids)
    else:
        indep_prod = oa_vecs.output_vertex_oa.data[oa_vecs.to_out_oa_idx[("lm_head",)]].unsqueeze(0)
        dep_prod = None
    
    W_ln = get_LN_matrix_for_node(hooked_model.model, oa_vecs, None, "lm_head", None)
    indep_prod = indep_prod @ W_ln @ hooked_model.model.lm_head.weight.data.T

    return dep_prod, indep_prod    # all already transposed


# if convert fails, we try manul inspection via patterns 
@torch.no_grad()
def visualize_mlp(
        complete_path, # path ends with unembed
        hooked_model: PruningModelWithHooksForQK,
        oa_vecs: OptimalQueryBiasVectors,
        dataloader: DataLoader,
        converted_mlp: dict[str, tuple], # path: (primitive:str, C:FloatTensor)
        logger,
        cache_num = 1000,
):
    
    pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
    split_nodes = re.findall(pattern, complete_path) # ['lm_head', 'mlp-1', 'attn_output-1-1', 'wte']
    mlp_path_idx = None
    for i, node in enumerate(split_nodes):
        if node.startswith("mlp") and ("-".join(split_nodes[i:]) not in converted_mlp):
            mlp_path_idx = i
    if mlp_path_idx is None:
        return None, None   # should use another function in this case
    mlp_path = "-".join(split_nodes[mlp_path_idx:])
    mlp_layer = int(split_nodes[mlp_path_idx][4:])

    num_layers = len(hooked_model.model.transformer.h)
    d_model = hooked_model.model.config.hidden_size
    config = hooked_model.config
    tokenizer = dataloader.dataset.tokenizer
    device = hooked_model.device

    logger("\ninspecting", complete_path)

    all_inp_depend = []
    all_mlp_out = []

    in_between_mlp = False
    A_lis = []
    for node in split_nodes[:mlp_path_idx]:
        if node.startswith("mlp") and not in_between_mlp:
            in_between_mlp = True
        elif node.startswith("attn_output") and in_between_mlp:
            _, layer, head = node.split("-")
            layer, head = int(layer), int(head)
            A_lis.append((layer, head))

    mean_A_max = []
    pbar = tqdm(total=cache_num)
    for i, batch in enumerate(dataloader):
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}

        mlp_out = None
        def temp_hook(module, input, output):
            nonlocal mlp_out
            mlp_out = hooked_model.activations[mlp_path]
        handle = hooked_model.model.transformer.h[mlp_layer].mlp.register_forward_hook(temp_hook)
                
        hooked_model(masks=torch.ones((1, 1), device=device), oa_vecs=oa_vecs, **batch)

        masks = (batch["input_ids"] != tokenizer.pad_token_id) & (batch["input_ids"] != tokenizer.eos_token_id)

        handle.remove()
        input_dependent = trace_mlp(hooked_model, converted_mlp, mlp_path, batch["input_ids"], batch["position_ids"], logger)

        input_dependent = input_dependent[masks]
        mlp_out = mlp_out[masks]
        # remove duplicates (within batch)
        dist = torch.cdist(input_dependent.unsqueeze(0), input_dependent.unsqueeze(0)).squeeze(0)
        selected_ids = (dist < 1e-3).float().argmax(dim=1).unique(sorted=False)
        all_inp_depend.append(input_dependent[selected_ids])
        all_mlp_out.append(mlp_out[selected_ids])
        
        if A_lis:
            A_aggregated = None
            for layer, head in A_lis:
                A = get_attn_weights_for_head(hooked_model, layer, head)
                if A_aggregated is None:
                    A_aggregated = A
                else:
                    A_aggregated = A_aggregated @ A
            mean_A_max.append(A_aggregated[masks].max(dim=-1)[0].mean())

        pbar.update(all_inp_depend[-1].size(0))
        if sum(item.size(0) for item in all_inp_depend) > cache_num:
            break
    pbar.close()
    
    if mean_A_max:
        mean_A_max = torch.stack(mean_A_max).mean().item()
        if mean_A_max < 0.9:
            logger("WARNING: results are not reliable")
 
    all_inp_depend = torch.cat(all_inp_depend, dim=0)
    all_mlp_out = torch.cat(all_mlp_out, dim=0)
    # logger("all_inp_depend", all_inp_depend.size(), "nan", all_inp_depend.isnan().any())
    # logger("all_mlp_out", all_mlp_out.size(), "nan", all_mlp_out.isnan().any())

    assert torch.allclose(all_inp_depend.sum(dim=1), torch.ones(all_inp_depend.size(0), device=device), atol=1e-3)

    i = mlp_path_idx - 1
    while i >= 0:
        node = split_nodes[i]
        if node == "lm_head":
            W_ln = get_LN_matrix_for_node(hooked_model.model, oa_vecs, None, "lm_head", None)
            all_mlp_out = all_mlp_out @ W_ln @ hooked_model.model.lm_head.weight.data.T
        elif node.startswith("attn_output"):
            _, layer, head = node.split("-")
            layer, head = int(layer), int(head)
            past_path = "-".join(split_nodes[i+1:])
            W_ln = get_LN_matrix_for_node(hooked_model.model, oa_vecs, layer, "v", head, past_path)
            W_v, W_o = get_OV_for_head(hooked_model.model, layer, head)
            all_mlp_out = all_mlp_out @ W_ln @ W_v @ W_o
        elif node.startswith("mlp"):
            layer = int(node.split("-")[1])
            inp_path = "-".join(split_nodes[i+1:])

            LN_var = hooked_model.oa_vecs.LN_var[hooked_model.oa_vecs.to_LN_idx[(layer, "mlp", inp_path)]].exp()
            input_activation = hooked_model.linear_layer_norm(hooked_model.ln_2[layer], all_mlp_out.unsqueeze(0), LN_var)

            all_mlp_out = hooked_model.oa_vecs.MLPs[f"{layer} {inp_path}"](input_activation).squeeze(0)
        else:
            raise RuntimeError("invalid node", node)
        i -= 1
        
    return all_inp_depend, all_mlp_out
    
    # for unembed and attn-product:
    #   mlp-end: ok
    #   mlp-attn-end: ok
    #   mlp-mlp-attn-end: ok
    #   mlp-attn-mlp-end: 
    #     if attn is one-hot (attn in between need to be one-hot, but not for previous layer): ok (need to remove A)
    #     else: makes no sense (raise a warning)

@torch.no_grad()
def visualize_mlp_logits(
    complete_path, # path ends with unembed
    hooked_model: PruningModelWithHooksForQK,
    oa_vecs: OptimalQueryBiasVectors,
    dataloader: DataLoader,
    converted_mlp: dict[str, tuple], # path: (primitive:str, C:FloatTensor)
    logger,
):
    assert complete_path.startswith("lm_head")
    mlp_in, mlp_logits = visualize_mlp(complete_path, hooked_model, oa_vecs, dataloader, converted_mlp, logger)
    if not hasattr(dataloader.dataset, "BCE") or not dataloader.dataset.BCE:
        mlp_logits = mlp_logits - mlp_logits.topk(dim=1, k=2)[0][:, 1:2]

    # for each out token, make 10 bins
    inp_cache = []  # from highest to lowest (most negative)
    out_cache = []

    num_bins = 20
    num_per_bin = 10
    max_v = mlp_logits.clamp(min=0).max(dim=0)[0]
    bin_edges = max_v.unsqueeze(0) * (torch.arange(1, num_bins//2+1, dtype=torch.float, device=mlp_logits.device) / (num_bins//2)).unsqueeze(1)
    
    for i in range(num_bins//2-1, -1, -1):
        topk_indices = mlp_logits.masked_fill(mlp_logits > bin_edges[i:i+1], -100000).topk(k=num_per_bin, dim=0, largest=True)[1]    # k, vocab_size
        inp_cache.append(mlp_in[topk_indices].clone())    # k, vocab, in_vocab
        out_cache.append(mlp_logits[topk_indices].clone())  # k, vocab, vocab

    min_v = mlp_logits.clamp(max=0).min(dim=0)[0]
    bin_edges = min_v.unsqueeze(0) * (torch.arange(1, num_bins//2+1, dtype=torch.float, device=mlp_logits.device) / (num_bins//2)).unsqueeze(1)
    
    for i in range(num_bins//2):
        topk_indices = mlp_logits.masked_fill(mlp_logits < bin_edges[i:i+1], 100000).topk(k=num_per_bin, dim=0, largest=False)[1]    # k, vocab_size
        inp_cache.append(mlp_in[topk_indices].clone())    # k, vocab, in_vocab
        out_cache.append(mlp_logits[topk_indices].clone())  # k, vocab, vocab
    
    return inp_cache, out_cache


@torch.no_grad()
def visualize_mlp_multi_source(
        complete_path, # path ends with unembed
        hooked_model: PruningModelWithHooksForQK,
        oa_vecs: OptimalQueryBiasVectors,
        dataloader: DataLoader,
        converted_mlp: dict[str, tuple], # path: (primitive:str, C:FloatTensor)
        logger,
        cache_num = 1000,
):
    
    pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
    split_nodes = re.findall(pattern, complete_path) # ['lm_head', 'attn_output-1-1', 'mlp-0']
    
    mlp_path = split_nodes[-1]
    mlp_layer = int(split_nodes[-1][4:])

    num_layers = len(hooked_model.model.transformer.h)
    d_model = hooked_model.model.config.hidden_size
    config = hooked_model.config
    tokenizer = dataloader.dataset.tokenizer
    device = hooked_model.device

    logger("\ninspecting", complete_path)

    all_inp_depend = []
    all_mlp_out = []

    pbar = tqdm(total=cache_num)
    for i, batch in enumerate(dataloader):
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}

        mlp_out = None
        def temp_hook(module, input, output):
            nonlocal mlp_out
            mlp_out = hooked_model.activations[mlp_path]
        handle = hooked_model.model.transformer.h[mlp_layer].mlp.register_forward_hook(temp_hook)
                
        hooked_model(masks=torch.ones((1, 1), device=device), oa_vecs=oa_vecs, **batch)

        masks = (batch["input_ids"] != tokenizer.pad_token_id) & (batch["input_ids"] != tokenizer.eos_token_id)

        handle.remove()
        input_dependents = trace_mlp_multi_source(hooked_model, converted_mlp, mlp_path, batch["input_ids"], batch["position_ids"], logger)
        
        input_dependents = [input_dependent[masks] for input_dependent in input_dependents]
        mlp_out = mlp_out[masks]
        # remove duplicates (within batch)
        cat_input_dependents = torch.cat(input_dependents, dim=1)
        dist = torch.cdist(cat_input_dependents.unsqueeze(0), cat_input_dependents.unsqueeze(0)).squeeze(0)
        selected_ids = (dist < 1e-3).float().argmax(dim=1).unique(sorted=False)
        
        all_inp_depend.append([input_dependent[selected_ids] for input_dependent in input_dependents])
        all_mlp_out.append(mlp_out[selected_ids])

        pbar.update(all_mlp_out[-1].size(0))
        if sum(item.size(0) for item in all_mlp_out) > cache_num:
            break
    pbar.close()
    
    # we don't compute attn between MLPs, could be very complex to code a general solution. 
    # logger("WARNING: results are not reliable")
 
    all_inp_depend = [torch.cat(lis, dim=0) for lis in zip(*all_inp_depend)]
    all_mlp_out = torch.cat(all_mlp_out, dim=0)
    # logger("all_inp_depend", all_inp_depend.size(), "nan", all_inp_depend.isnan().any())
    # logger("all_mlp_out", all_mlp_out.size(), "nan", all_mlp_out.isnan().any())

    assert all(torch.allclose(item.sum(dim=1), torch.ones(item.size(0), device=device), atol=1e-3) for item in all_inp_depend)

    i = len(split_nodes) - 2
    while i >= 0:
        node = split_nodes[i]
        if node == "lm_head":
            W_ln = get_LN_matrix_for_node(hooked_model.model, oa_vecs, None, "lm_head", None)
            all_mlp_out = all_mlp_out @ W_ln @ hooked_model.model.lm_head.weight.data.T
        elif node.startswith("attn_output"):
            _, layer, head = node.split("-")
            layer, head = int(layer), int(head)
            past_path = "-".join(split_nodes[i+1:])
            W_ln = get_LN_matrix_for_node(hooked_model.model, oa_vecs, layer, "v", head, past_path)
            W_v, W_o = get_OV_for_head(hooked_model.model, layer, head)
            all_mlp_out = all_mlp_out @ W_ln @ W_v @ W_o
        else:
            raise RuntimeError("invalid node", node)
        i -= 1
    
    all_inp_depend = {inp_v_path: mlp_inp for inp_v_path, mlp_inp in zip(config[int(mlp_path[4:])]["mlp"], all_inp_depend)}

    return all_inp_depend, all_mlp_out

@torch.no_grad()
def visualize_mlp_logits_multi_source(
    complete_path, # path ends with unembed
    hooked_model: PruningModelWithHooksForQK,
    oa_vecs: OptimalQueryBiasVectors,
    dataloader: DataLoader,
    converted_mlp: dict[str, tuple], # path: (primitive:str, C:FloatTensor)
    logger,
):
    assert complete_path.startswith("lm_head")
    mlp_ins, mlp_logits = visualize_mlp_multi_source(complete_path, hooked_model, oa_vecs, dataloader, converted_mlp, logger)
    if not hasattr(dataloader.dataset, "BCE") or not dataloader.dataset.BCE:
        mlp_logits = mlp_logits - mlp_logits.topk(dim=1, k=2)[0][:, 1:2]

    # for each out token, make 10 bins
    inp_cache = defaultdict(list)  # from highest to lowest (most negative)
    out_cache = []

    num_bins = 20
    num_per_bin = 10
    max_v = mlp_logits.clamp(min=0).max(dim=0)[0]
    bin_edges = max_v.unsqueeze(0) * (torch.arange(1, num_bins//2+1, dtype=torch.float, device=mlp_logits.device) / (num_bins//2)).unsqueeze(1)
    
    for i in range(num_bins//2-1, -1, -1):
        topk_indices = mlp_logits.masked_fill(mlp_logits > bin_edges[i:i+1], -100000).topk(k=num_per_bin, dim=0, largest=True)[1]    # k, vocab_size
        for k, mlp_in in mlp_ins.items():
            inp_cache[k].append(mlp_in[topk_indices].clone())  # k, vocab, in_vocab
        out_cache.append(mlp_logits[topk_indices].clone())  # k, vocab, vocab

    min_v = mlp_logits.clamp(max=0).min(dim=0)[0]
    bin_edges = min_v.unsqueeze(0) * (torch.arange(1, num_bins//2+1, dtype=torch.float, device=mlp_logits.device) / (num_bins//2)).unsqueeze(1)
    
    for i in range(num_bins//2):
        topk_indices = mlp_logits.masked_fill(mlp_logits < bin_edges[i:i+1], 100000).topk(k=num_per_bin, dim=0, largest=False)[1]    # k, vocab_size
        for k, mlp_in in mlp_ins.items():
            inp_cache[k].append(mlp_in[topk_indices].clone())  # k, vocab, in_vocab
        out_cache.append(mlp_logits[topk_indices].clone())  # k, vocab, vocab
    
    return inp_cache, out_cache

@torch.no_grad()
def visualize_mlp_attn_products(
    qk_complete_path: tuple[str, str], # path ends with qk product
    attn_layer_idx, 
    attn_head_idx,
    hooked_model: PruningModelWithHooksForQK,
    oa_vecs: OptimalQueryBiasVectors,
    dataloader: DataLoader,
    converted_mlp: dict[str, tuple], # path: (primitive:str, C:FloatTensor)
    logger,
):
    num_layers = len(hooked_model.model.transformer.h)
    d_model = hooked_model.model.config.hidden_size
    config = hooked_model.config
    device = hooked_model.device

    logger("\ninspecting", qk_complete_path)

    q_complete_path, k_complete_path =  qk_complete_path
    q_all_inp_depend, q_all_mlp_out = visualize_mlp(q_complete_path, hooked_model, oa_vecs, dataloader, converted_mlp, logger)
    if q_all_inp_depend is None:
        q_all_inp_depend, q_all_mlp_out = visualize_convertable_path(q_complete_path, hooked_model, oa_vecs, dataloader, converted_mlp, logger)
    W_ln = get_LN_matrix_for_node(hooked_model.model, oa_vecs, attn_layer_idx, "q", attn_head_idx)
    W_q = get_QK_for_head(hooked_model.model, attn_layer_idx, "q", attn_head_idx)
    q_all_mlp_out = q_all_mlp_out @ W_ln @ W_q
    
    k_all_inp_depend, k_all_mlp_out = visualize_mlp(k_complete_path, hooked_model, oa_vecs, dataloader, converted_mlp, logger)
    if k_all_inp_depend is None:
        k_all_inp_depend, k_all_mlp_out = visualize_convertable_path(k_complete_path, hooked_model, oa_vecs, dataloader, converted_mlp, logger)
    W_ln = get_LN_matrix_for_node(hooked_model.model, oa_vecs, attn_layer_idx, "k", attn_head_idx)
    W_k = get_QK_for_head(hooked_model.model, attn_layer_idx, "k", attn_head_idx)
    k_all_mlp_out = k_all_mlp_out @ W_ln @ W_k

    # over entire Q @ K.T  (2000 by 2000)
    all_attn_logits = q_all_mlp_out @ k_all_mlp_out.T
    all_attn_logits -= all_attn_logits.mean(dim=1, keepdim=True)
    
    # 1: for entire matrix, make some bins
    num_bins = 20
    num_per_bin = 10
    num_q_cluster = 5

    q_inp_cache = [[] for i in range(num_bins)]  # from highest to lowest (most negative)  # k, num_q_cluster, in_vocab
    k_inp_cache = [[] for i in range(num_bins)]
    out_cache = [[] for i in range(num_bins)]

    kmeans = KMeans(n_clusters=num_q_cluster, random_state=0).fit(q_all_inp_depend.numpy(force=True))
    cluster_labels = torch.tensor(kmeans.labels_, device=device)
    
    for c_idx in range(num_q_cluster+1):
        if c_idx == 0:
            attn_logits = all_attn_logits
            selected_q_all_inp_depend = q_all_inp_depend
        else:
            attn_logits = all_attn_logits[cluster_labels == c_idx-1]
            selected_q_all_inp_depend = q_all_inp_depend[cluster_labels == c_idx-1]
            if attn_logits.numel() < num_per_bin:
                continue
        max_v = attn_logits.clamp(min=0).max()
        bin_edges = max_v * torch.arange(1, num_bins//2+1, dtype=torch.float, device=device) / (num_bins//2)
        
        for i in range(num_bins//2-1, -1, -1):
            topk_values, topk_indices = attn_logits.masked_fill(attn_logits > bin_edges[i], -100000).view(-1).topk(k=num_per_bin, largest=True)
            q_indices = topk_indices // k_all_mlp_out.size(0)
            k_indices = topk_indices % k_all_mlp_out.size(0)
            q_inp_cache[num_bins//2-1-i].append(selected_q_all_inp_depend[q_indices].clone()) # k, in_vocab
            k_inp_cache[num_bins//2-1-i].append(k_all_inp_depend[k_indices].clone()) # k, in_vocab
            out_cache[num_bins//2-1-i].append(topk_values)  # k

        min_v = attn_logits.clamp(max=0).min()
        bin_edges = min_v * torch.arange(1, num_bins//2+1, dtype=torch.float, device=device) / (num_bins//2)
        
        for i in range(num_bins//2):
            topk_values, topk_indices = attn_logits.masked_fill(attn_logits < bin_edges[i], 100000).view(-1).topk(k=num_per_bin, largest=False)
            q_indices = topk_indices // k_all_mlp_out.size(0)
            k_indices = topk_indices % k_all_mlp_out.size(0)
            
            q_inp_cache[num_bins//2+i].append(selected_q_all_inp_depend[q_indices].clone()) # k, in_vocab
            k_inp_cache[num_bins//2+i].append(k_all_inp_depend[k_indices].clone()) # k, in_vocab
            out_cache[num_bins//2+i].append(topk_values)  # k
    
    q_inp_cache = [torch.stack(item, dim=1) for item in q_inp_cache]
    k_inp_cache = [torch.stack(item, dim=1) for item in k_inp_cache]
    out_cache = [torch.stack(item, dim=1) for item in out_cache]
    return q_inp_cache, k_inp_cache, out_cache
    

@torch.no_grad()
def visualize_convertable_path(
        complete_path, # path ends with unembed
        hooked_model: PruningModelWithHooksForQK,
        oa_vecs: OptimalQueryBiasVectors,
        dataloader: DataLoader,
        converted_mlp: dict[str, tuple], # path: (primitive:str, C:FloatTensor)
        logger,
):
    device = hooked_model.device
    tokenizer = dataloader.dataset.tokenizer
    get_product_func = get_product_for_one_side if hasattr(oa_vecs, "MLPs") else get_product_for_one_side_multi_source  # TODO: haven't check if multi source is okay
    
    all_inp_depend = []
    all_mlp_out = []

    for i, batch in enumerate(dataloader):
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
       
        hooked_model(masks=torch.ones((1, 1), device=device), oa_vecs=oa_vecs, **batch)

        dep_prod, indep_prod = get_product_func(hooked_model, oa_vecs, converted_mlp, complete_path, batch["input_ids"], batch["position_ids"])    # if input_ids 2 dimension, dep_prod 3 dim, and indep_prod 2 dim
    
        masks = (batch["input_ids"] != tokenizer.pad_token_id) & (batch["input_ids"] != tokenizer.eos_token_id)

        input_dependent = dep_prod[masks]
        mlp_out = input_dependent @ indep_prod
        # remove duplicates (within batch)
        dist = torch.cdist(input_dependent.unsqueeze(0), input_dependent.unsqueeze(0)).squeeze(0)
        selected_ids = (dist < 1e-3).float().argmax(dim=1).unique(sorted=False)
        all_inp_depend.append(input_dependent[selected_ids])
        all_mlp_out.append(mlp_out[selected_ids])
        
        if sum(item.size(0) for item in all_inp_depend) > 500:
            break
    
    all_inp_depend = torch.cat(all_inp_depend, dim=0)
    all_mlp_out = torch.cat(all_mlp_out, dim=0)
    # logger("all_inp_depend", all_inp_depend.size(), "nan", all_inp_depend.isnan().any())
    # logger("all_mlp_out", all_mlp_out.size(), "nan", all_mlp_out.isnan().any())

    assert torch.allclose(all_inp_depend.sum(dim=1), torch.ones(all_inp_depend.size(0), device=device), atol=1e-3)
    return all_inp_depend, all_mlp_out
    
