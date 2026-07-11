import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from itertools import product
import re
from copy import deepcopy
from collections import defaultdict, OrderedDict
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
from patching_data import get_tokenizer_and_dataset_for_task
from train_new_models import customCollator, customBCECollator
from patching_utils import get_mlp_primitives, get_mlp_primitives_multi_source


# {0: {'k': {0: ['wte', 'wpe'], 1: ['wte', 'wpe'], 2: ['wte', 'wpe'], 3: ['wte', 'wpe']}, 'q': {0: ['wte', 'wpe'], 1: ['wte', 'wpe'], 2: ['wte', 'wpe'], 3: ['wte', 'wpe']}, 'v': {0: ['wte', 'wpe'], 1: ['wte', 'wpe'], 2: ['wte', 'wpe'], 3: ['wte', 'wpe']}, 'mlp': ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3']}, 
#  1: {'k': {0: ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3', 'mlp-0'], 
#            1: ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3', 'mlp-0'], 
#            2: ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3', 'mlp-0'], 
#            3: ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3', 'mlp-0']}, 
#     'q': {0: ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3', 'mlp-0'], 
#           1: ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3', 'mlp-0'], 
#           2: ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3', 'mlp-0'], 
#           3: ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3', 'mlp-0']}, 
#     'v': {0: ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3', 'mlp-0'], 
#           1: ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3', 'mlp-0'], 
#           2: ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3', 'mlp-0'], 
#           3: ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3', 'mlp-0']}, 
#     'mlp': ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3', 'attn_output-1-0', 'attn_output-1-1', 'attn_output-1-2', 'attn_output-1-3', 'mlp-0']}, 
# 'lm_head': ['wte', 'wpe', 'attn_output-0-0', 'attn_output-0-1', 'attn_output-0-2', 'attn_output-0-3', 'attn_output-1-0', 'attn_output-1-1', 'attn_output-1-2', 'attn_output-1-3', 'mlp-0', 'mlp-1']}

class MaskSampler(nn.Module):
    # rewritten version, and linearize LN
    def __init__(self, config):
        super().__init__()
        
        param_idx = 0
        mapping_to_param_idx = {}
        input_vertex = set()
        output_vertex = set()
        for k1 in config:
            if type(config[k1]) == dict:
                for k2 in config[k1]:
                    if type(config[k1][k2]) == dict:
                        for k3 in config[k1][k2]:
                            if type(config[k1][k2][k3]) == list:
                                output_vertex.add((k1, k2, k3))
                                for item in config[k1][k2][k3]:
                                    mapping_to_param_idx[(k1, k2, k3, item)] = param_idx
                                    param_idx += 1
                                    input_vertex.add(item)
                            else:
                                raise RuntimeError
                    elif type(config[k1][k2]) == list:
                        output_vertex.add((k1, k2))
                        for item in config[k1][k2]:
                            mapping_to_param_idx[(k1, k2, item)] = param_idx
                            param_idx += 1
                            input_vertex.add(item)
                    else:
                        raise RuntimeError
            elif type(config[k1]) == list:
                output_vertex.add((k1,))
                for item in config[k1]:
                    mapping_to_param_idx[(k1, item)] = param_idx
                    param_idx += 1
                    input_vertex.add(item)
            else:
                raise RuntimeError
            
        self.config = config
        self.mapping_to_param_idx = mapping_to_param_idx
        self.sample_params = nn.Parameter(torch.ones(param_idx))
        self.input_vertex = list(input_vertex)
        self.output_vertex = list(output_vertex)

        assert not any("attention_bias" in item for item in input_vertex)

        num_head_per_layer = []
        for i in range(len(config)-1):
            num_head_per_layer.append(len(config[i]["k"]))
        self.num_head_per_layer = num_head_per_layer

        self.window_function = lambda x: x * (1-x)

    def sample_masks(self, bz):

        prob = F.sigmoid(self.sample_params).unsqueeze(0).expand(bz, -1)
        unif = torch.rand_like(prob)
        window_size = self.window_function(prob).detach()
        prob = window_size * prob + (1 - window_size) * prob.detach()
        masks = ((prob - unif) / window_size + 0.5).clamp(0, 1)

        masks = self.prune_dangling_edges(masks)  

        n_samples = ((masks < 1-1e-3) & (masks > 1e-3)).sum(dim=0).float()
        grad_wts = torch.where(n_samples < 1, 0, bz / n_samples)
        masks = grad_wts * masks + (1 - grad_wts) * masks.detach()
        
        return masks


    def prune_dangling_edges(self, masks):
        with torch.device(masks.device):
            # reachability to input
            with torch.no_grad():
                bz = masks.size(0)
                reachable_to_input = {"wte": torch.ones(bz).bool(), "wpe": torch.ones(bz).bool()}
                for layer, num_head in enumerate(self.num_head_per_layer):
                    for head in range(num_head):
                        reach_head = []
                        for act in ["q", "k", "v"]:
                            reach = [torch.zeros(bz).bool()]
                            for input_v in self.config[layer][act][head]:
                                reach.append( (masks[:, self.mapping_to_param_idx[(layer, act, head, input_v)]] > 0) & reachable_to_input[input_v] )
                            reach_head.append(torch.stack(reach).any(dim=0))

                        reachable_to_input[f"attn_output-{layer}-{head}"] = torch.stack(reach_head).all(dim=0)
                    
                    reach_mlp = [torch.zeros(bz).bool()]
                    for input_v in self.config[layer]["mlp"]:
                        reach_mlp.append( (masks[:, self.mapping_to_param_idx[(layer, "mlp", input_v)]] > 0) & reachable_to_input[input_v] )
                    reach_mlp = torch.stack(reach_mlp).any(dim=0)
                    reachable_to_input[f"mlp-{layer}"] = reach_mlp

            # remove edges connected to dangling vertices, 1st time using input reachability
            for layer, num_head in enumerate(self.num_head_per_layer):
                for head in range(num_head):
                    dangling_out = ~reachable_to_input[f"attn_output-{layer}-{head}"]
                    for activation in ["q", "k", "v"]:
                        for input_v in self.config[layer][activation][head]:
                            dangling = dangling_out | ~reachable_to_input[input_v]
                            masks[:, self.mapping_to_param_idx[(layer, activation, head, input_v)]].masked_fill_(dangling, 0)

                for input_v in self.config[layer]["mlp"]:
                    dangling_out = ~reachable_to_input[f"mlp-{layer}"]
                    dangling = dangling_out | ~reachable_to_input[input_v]
                    masks[:, self.mapping_to_param_idx[(layer, "mlp", input_v)]].masked_fill_(dangling, 0)
            
            for input_v in self.config["lm_head"]:
                dangling = ~reachable_to_input[input_v]
                masks[:, self.mapping_to_param_idx[("lm_head", input_v)]].masked_fill_(dangling, 0)


            with torch.no_grad():
                reachable_to_output = defaultdict(lambda : torch.zeros(bz).bool())
                for input_v in self.config["lm_head"]:
                    reachable_to_output[input_v] = masks[:, self.mapping_to_param_idx[("lm_head", input_v)]] > 0
                for layer in range(len(self.num_head_per_layer)-1, -1, -1):
                    for input_v in self.config[layer]["mlp"]:
                        reach_inp = (masks[:, self.mapping_to_param_idx[(layer, "mlp", input_v)]] > 0) & reachable_to_output[f"mlp-{layer}"]
                        reachable_to_output[input_v] = reachable_to_output[input_v] | reach_inp

                    for head in range(self.num_head_per_layer[layer]):
                        reach_head = reachable_to_output[f"attn_output-{layer}-{head}"]
                        for activation in ["q", "k", "v"]:
                            for input_v in self.config[layer][activation][head]:
                                reach_inp = (masks[:, self.mapping_to_param_idx[(layer, activation, head, input_v)]] > 0) & reach_head
                                reachable_to_output[input_v] = reachable_to_output[input_v] | reach_inp
            
            # remove edges connected to dangling vertices, 2nd time using output reachability
            for layer, num_head in enumerate(self.num_head_per_layer):
                for head in range(num_head):
                    dangling_out = ~reachable_to_output[f"attn_output-{layer}-{head}"]
                    for activation in ["q", "k", "v"]:
                        for input_v in self.config[layer][activation][head]:
                            dangling = dangling_out | ~reachable_to_output[input_v]
                            masks[:, self.mapping_to_param_idx[(layer, activation, head, input_v)]].masked_fill_(dangling, 0)

                for input_v in self.config[layer]["mlp"]:
                    dangling_out = ~reachable_to_output[f"mlp-{layer}"]
                    dangling = dangling_out | ~reachable_to_output[input_v]
                    masks[:, self.mapping_to_param_idx[(layer, "mlp", input_v)]].masked_fill_(dangling, 0)
            
            for input_v in self.config["lm_head"]:
                dangling = ~reachable_to_output[input_v]
                masks[:, self.mapping_to_param_idx[("lm_head", input_v)]].masked_fill_(dangling, 0)

        return masks
    
    def get_penalty(self, node_reg_coef: float):
        edge_reg = F.sigmoid(self.sample_params).sum()

        if node_reg_coef == 0:
            return edge_reg, (edge_reg.item(), 0)
        else:
            raise NotImplementedError

    def sample_binary_masks(self, bz, threshold=0):
        masks = (self.sample_params > threshold).float().unsqueeze(0).expand(bz, -1)
        masks = self.prune_dangling_edges(masks)  
        return masks


class OptimalAblationVectors(nn.Module):
    def __init__(self, input_vertex, output_vertex, LN_vertex, MLP_vertex, model_config, init_var):
        super().__init__()
        d_model = model_config.hidden_size

        self.input_vertex = input_vertex
        self.input_vertex_oa = nn.Parameter(torch.zeros(len(input_vertex), d_model))
        self.to_in_oa_idx = {item: i for i, item in enumerate(input_vertex)}

        self.output_vertex = output_vertex
        self.output_vertex_oa = nn.Parameter(torch.zeros(len(output_vertex), d_model))
        self.to_out_oa_idx = {item: i for i, item in enumerate(output_vertex)}

        assert init_var.size(0) == len(LN_vertex)
        self.LN_vertex = LN_vertex
        self.LN_var = nn.Parameter(init_var)
        self.to_LN_idx = {item: i for i, item in enumerate(LN_vertex)}

        self.MLP_vertex = MLP_vertex
        if MLP_vertex is not None and len(MLP_vertex) > 0:
            inner_dim = model_config.n_inner if model_config.n_inner is not None else 4 * model_config.hidden_size
            self.MLPs = nn.ModuleDict({f"{item[0]} {item[2]}": GPT2MLP(inner_dim, model_config)  for item in MLP_vertex})    # item: (0, 'mlp', 'attn_output-1-0-attn_output-0-0-wte')


class PruningModelWithHooks:
    def __init__(self, model, config, mapping_to_param_idx, logger=None, linear_LN=True):
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        self.full_config = deepcopy(config)
        self.mapping_to_param_idx = mapping_to_param_idx
        self.logger = logger
        self.device = self.model.device
        self.linear_LN = linear_LN
        
        self.hooks = []

        self.activations = {}

        self.ln_1 = {}
        self.ln_2 = {}
        self.hooks.append( self.model.transformer.wte.register_forward_hook(partial(
            self.save_activation_hook, activation_type="wte", layer=None
        )) )
        self.hooks.append( self.model.transformer.wpe.register_forward_hook(partial(
            self.save_activation_hook, activation_type="wpe", layer=None
        )) )
        self.hooks.append( self.model.lm_head.register_forward_hook(
            self.lm_head_hook
        ) )
        self.ln_f = self.model.transformer.ln_f
        for layer in range(len(self.model.transformer.h)):
            self.ln_1[layer] = self.model.transformer.h[layer].ln_1
            self.ln_2[layer] = self.model.transformer.h[layer].ln_2
            self.hooks.append( self.model.transformer.h[layer].attn.c_attn.register_forward_hook(partial(
                self.c_attn_hook, layer=layer
            )) )
            self.hooks.append( self.model.transformer.h[layer].attn.c_proj.register_forward_hook(partial(
                self.save_activation_hook, activation_type="attn_output", layer=layer
            )) )
            self.hooks.append( self.model.transformer.h[layer].mlp.register_forward_hook(partial(
                self.mlp_hook, layer=layer
            )) )
        
        self.masks = None
        self.oa_vecs = None

    def set_mask_sampler(self, mask_sampler: MaskSampler):
        self.logger("set mask sampler, should be done with training")
        self.mask_sampler = mask_sampler

    def __call__(self, *args, masks=None, oa_vecs: OptimalAblationVectors =None, **kwargs):
        if masks is not None:
            self.masks = masks
            self.oa_vecs = oa_vecs
        else:
            assert hasattr(self, "mask_sampler") and self.mask_sampler is not None
            assert self.oa_vecs is not None
            if args:
                bz = args[0].size(0)
            else:
                bz = kwargs["input_ids"].size(0)
            self.masks = self.mask_sampler.sample_binary_masks(bz)
            
        return self.model(*args, **kwargs)

    def linear_layer_norm(self, module: nn.LayerNorm, input, scalar, bias=True):    # default value for bias is different from later models
        out = (input - input.mean(dim=-1, keepdim=True)) / (scalar + module.eps).sqrt() * module.weight.view(1, 1, -1)
        if bias:    
            out = out + module.bias.view(1, 1, -1)
        return out
    
    def save_activation_hook(self, module, input, output, activation_type, layer):
        if activation_type in ["wte", "wpe"]:
            self.activations[activation_type] = self.model.transformer.drop(output.detach())
        elif activation_type == "attn_output":
            input = input[0]
            bz, seq_len = input.shape[:-1]
            d_model = module.weight.size(-1)
            num_heads, head_dim = self.model.transformer.h[layer].attn.num_heads, self.model.transformer.h[layer].attn.head_dim
            attention_by_head = input.view(bz * seq_len, num_heads, head_dim)
            attention_by_head_output = attention_by_head.transpose(0, 1) @ module.weight.view(num_heads, head_dim, d_model)  # num_head, bz*seq_len, d_model
            attention_by_head_output = self.model.transformer.h[layer].attn.resid_dropout(attention_by_head_output)
            attention_by_head_output = attention_by_head_output.view(num_heads, bz, seq_len, d_model).unbind(dim=0)

            # assert (((sum(attention_by_head_output) + module.bias) - output).abs() < 1e-3).all(), ((sum(attention_by_head_output) + module.bias) - output)
            for head, attn in enumerate(attention_by_head_output):
                self.activations[f"{activation_type}-{layer}-{head}"] = attn
        else:
            raise NotImplementedError()

    def c_attn_hook(self, module, input, output, layer):
        # input.shape == (bsz, seqlen, hidden_size)
        # output.shape == torch.vstack((bsz, seqlen, embed_dim), (bsz, seqlen, embed_dim), (bsz, seqlen, embed_dim))
        # output = query_states, key_states, value_states

        num_heads = self.model.transformer.h[layer].attn.num_heads
        head_dim = self.model.transformer.h[layer].attn.head_dim
        d_model = input[0].size(-1)

        input_activations = []
        for i, (activation, head) in enumerate(product(["q", "k", "v"], range(num_heads))):
            summed_act = torch.zeros_like(input[0])
            for activation_name in self.full_config[layer][activation][head]:
                coef = self.masks[:, self.mapping_to_param_idx[(layer, activation, head, activation_name)]]
                first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
                oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
                second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                                oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
                summed_act = summed_act + first_term + second_term.unsqueeze(1)
            oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[(layer, activation, head)]]
            summed_act = summed_act + oa.view(1, 1, -1)

            if self.linear_LN:
                LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[(layer, activation, head)]].exp()
                input_activations.append(self.linear_layer_norm(self.ln_1[layer], summed_act, LN_var))
            else:
                input_activations.append(self.ln_1[layer](summed_act))
        input_activations = torch.stack(input_activations)

        output_activations = input_activations.flatten(start_dim=1, end_dim=2) @ \
                             module.weight.view(d_model, num_heads * 3, head_dim).transpose(0, 1)
        output_activations = output_activations.transpose(0, 1).contiguous().view(*input[0].size()[:2], num_heads * head_dim * 3)
        output_activations += module.bias.view(1, 1, -1)
        
        return output_activations
    
    def mlp_hook(self, module, input, output, layer):
        summed_act = torch.zeros_like(input[0])
        for activation_name in self.full_config[layer]["mlp"]:
            coef = self.masks[:, self.mapping_to_param_idx[(layer, "mlp", activation_name)]]

            first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
            oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
            second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                                oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
            summed_act = summed_act + first_term + second_term.unsqueeze(1)
        oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[(layer, "mlp")]]
        summed_act = summed_act + oa.view(1, 1, -1)

        if self.linear_LN:
            LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[(layer, "mlp")]].exp()
            input_activation = self.linear_layer_norm(self.ln_2[layer], summed_act, LN_var)
        else:
            input_activation = self.ln_2[layer](summed_act)

        output = module.forward(input_activation)
        self.activations[f"mlp-{layer}"] = output   # dropout is included
        return output

    def lm_head_hook(self, module, input, output):
        summed_act = torch.zeros_like(input[0])
        for activation_name in self.full_config["lm_head"]:
            coef = self.masks[:, self.mapping_to_param_idx[("lm_head", activation_name)]]

            first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
            oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
            second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                                oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
            summed_act = summed_act + first_term + second_term.unsqueeze(1)
        oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[("lm_head",)]]
        summed_act = summed_act + oa.view(1, 1, -1)

        if self.linear_LN:
            LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[("lm_head",)]].exp()
            input_activation = self.linear_layer_norm(self.ln_f, summed_act, LN_var)
        else:
            input_activation = self.ln_f(summed_act)

        output = module.forward(input_activation)
        return output

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()



# {0: {'k': {0: ['wte', 'wpe'], 1: ['wte', 'wpe'], 2: ['wte', 'wpe'], 3: ['wte', 'wpe']}, 'q': {0: ['wte', 'wpe'], 1: ['wte', 'wpe'], 2: ['wte', 'wpe'],
#  3: ['wte', 'wpe']}, 'v': {0: ['wte', 'wpe'], 1: ['wte', 'wpe'], 2: ['wte', 'wpe'], 3: ['wte', 'wpe']}, 'mlp': ['wte', 'wpe', 'attn_output-0-0-wte', 'attn_output-0-0-wpe', 'attn_output
# -0-1-wte', 'attn_output-0-1-wpe', 'attn_output-0-2-wte', 'attn_output-0-2-wpe', 'attn_output-0-3-wte', 'attn_output-0-3-wpe', 'attention_bias-0']}, 1: {'k': {0: ['wte', 'wpe', 'attn_ou
# tput-0-0-wte', 'attn_output-0-0-wpe', 'attn_output-0-1-wte', 'attn_output-0-1-wpe', 'attn_output-0-2-wte', 'attn_output-0-2-wpe', 'attn_output-0-3-wte', 'attn_output-0-3-wpe', 'attenti
# on_bias-0', 'mlp-0'], 1: ['wte', 'wpe', 'attn_output-0-0-wte', 'attn_output-0-0-wpe', 'attn_output-0-1-wte', 'attn_output-0-1-wpe', 'attn_output-0-2-wte', 'attn_output-0-2-wpe', 'attn_
# output-0-3-wte', 'attn_output-0-3-wpe', 'attention_bias-0', 'mlp-0'], 2: ['wte', 'wpe', 'attn_output-0-0-wte', 'attn_output-0-0-wpe', 'attn_output-0-1-wte', 'attn_output-0-1-wpe', 'att
# n_output-0-2-wte', 'attn_output-0-2-wpe', 'attn_output-0-3-wte', 'attn_output-0-3-wpe', 'attention_bias-0', 'mlp-0'], 3: ['wte', 'wpe', 'attn_output-0-0-wte', 'attn_output-0-0-wpe', 'a
# ttn_output-0-1-wte', 'attn_output-0-1-wpe', 'attn_output-0-2-wte', 'attn_output-0-2-wpe', 'attn_output-0-3-wte', 'attn_output-0-3-wpe', 'attention_bias-0', 'mlp-0']}, 'q': {0: ['wte', 
# 'wpe', 'attn_output-0-0-wte', 'attn_output-0-0-wpe', 'attn_output-0-1-wte', 'attn_output-0-1-wpe', 'attn_output-0-2-wte', 'attn_output-0-2-wpe', 'attn_output-0-3-wte', 'attn_output-0-3
# -wpe', 'attention_bias-0', 'mlp-0'], 1: ['wte', 'wpe', 'attn_output-0-0-wte', 'attn_output-0-0-wpe', 'attn_output-0-1-wte', 'attn_output-0-1-wpe', 'attn_output-0-2-wte', 'attn_output-0
# -2-wpe', 'attn_output-0-3-wte', 'attn_output-0-3-wpe', 'attention_bias-0', 'mlp-0'], 2: ['wte', 'wpe', 'attn_output-0-0-wte', 'attn_output-0-0-wpe', 'attn_output-0-1-wte', 'attn_output
# -0-1-wpe', 'attn_output-0-2-wte', 'attn_output-0-2-wpe', 'attn_output-0-3-wte', 'attn_output-0-3-wpe', 'attention_bias-0', 'mlp-0'], 3: ['wte', 'wpe', 'attn_output-0-0-wte', 'attn_outp
# ut-0-0-wpe', 'attn_output-0-1-wte', 'attn_output-0-1-wpe', 'attn_output-0-2-wte', 'attn_output-0-2-wpe', 'attn_output-0-3-wte', 'attn_output-0-3-wpe', 'attention_bias-0', 'mlp-0']}, 'v
# ': {0: ['wte', 'wpe', 'attn_output-0-0-wte', 'attn_output-0-0-wpe', 'attn_output-0-1-wte', 'attn_output-0-1-wpe', 'attn_output-0-2-wte', 'attn_output-0-2-wpe', 'attn_output-0-3-wte', '
# attn_output-0-3-wpe', 'attention_bias-0', 'mlp-0'], 1: ['wte', 'wpe', 'attn_output-0-0-wte', 'attn_output-0-0-wpe', 'attn_output-0-1-wte', 'attn_output-0-1-wpe', 'attn_output-0-2-wte',
#  'attn_output-0-2-wpe', 'attn_output-0-3-wte', 'attn_output-0-3-wpe', 'attention_bias-0', 'mlp-0'], 2: ['wte', 'wpe', 'attn_output-0-0-wte', 'attn_output-0-0-wpe', 'attn_output-0-1-wte
# ', 'attn_output-0-1-wpe', 'attn_output-0-2-wte', 'attn_output-0-2-wpe', 'attn_output-0-3-wte', 'attn_output-0-3-wpe', 'attention_bias-0', 'mlp-0'], 3: ['wte', 'wpe', 'attn_output-0-0-w
# te', 'attn_output-0-0-wpe', 'attn_output-0-1-wte', 'attn_output-0-1-wpe', 'attn_output-0-2-wte', 'attn_output-0-2-wpe', 'attn_output-0-3-wte', 'attn_output-0-3-wpe', 'attention_bias-0'
# , 'mlp-0']}, 'mlp': ['wte', 'wpe', 'attn_output-0-0-wte', 'attn_output-0-0-wpe', 'attn_output-0-1-wte', 'attn_output-0-1-wpe', 'attn_output-0-2-wte', 'attn_output-0-2-wpe', 'attn_outpu
# t-0-3-wte', 'attn_output-0-3-wpe', 'attn_output-1-0-wte', 'attn_output-1-0-wpe', 'attn_output-1-0-attn_output-0-0-wte', 'attn_output-1-0-attn_output-0-0-wpe', 'attn_output-1-0-attn_out
# put-0-1-wte', 'attn_output-1-0-attn_output-0-1-wpe', '

class MaskSamplerFullPaths(MaskSampler):
    def __init__(self, config: dict, incl_mlp):
        super().__init__(config)

        output_vertex = set(self.output_vertex)
        v_output_vertex = set()
        mlp_output_vertex = set()
        for k1 in config:
            if type(config[k1]) == dict:
                for k2 in config[k1]:
                    if k2 == "v":
                        for k3 in config[k1][k2]:
                            output_vertex.remove((k1, k2, k3))
                            for item in config[k1][k2][k3]:
                                v_output_vertex.add((k1, k2, k3, item))
                    elif incl_mlp and k2 == "mlp":
                        output_vertex.remove((k1, k2))
                        for item in config[k1][k2]:
                            mlp_output_vertex.add((k1, k2, item))
        
        self.output_vertex = list(output_vertex)
        self.all_output_vertex = list(output_vertex.union(v_output_vertex).union(mlp_output_vertex))
        self.mlp_output_vertex = list(mlp_output_vertex)
        self.incl_mlp = incl_mlp

        self.sample_params.data = torch.ones_like(self.sample_params.data)
    
    def prune_dangling_edges(self, masks):
        with torch.device(masks.device):
            with torch.no_grad():
                bz = masks.size(0)
                reachable_to_input = {"wte": torch.ones(bz).bool(), "wpe": torch.ones(bz).bool()}
                for layer, num_head in enumerate(self.num_head_per_layer):
                    for head in range(num_head):
                        reach_q = [torch.zeros(bz).bool()]
                        for input_v in self.config[layer]["q"][head]:
                            reach_q.append( (masks[:, self.mapping_to_param_idx[(layer, "q", head, input_v)]] > 0) & reachable_to_input[input_v] )
                        reach_q = torch.stack(reach_q).any(dim=0)

                        reach_k = [torch.zeros(bz).bool()]
                        for input_v in self.config[layer]["k"][head]:
                            reach_k.append( (masks[:, self.mapping_to_param_idx[(layer, "k", head, input_v)]] > 0) & reachable_to_input[input_v] )
                        reach_k = torch.stack(reach_k).any(dim=0)

                        for input_v in self.config[layer]["v"][head]:
                            reach_v = (masks[:, self.mapping_to_param_idx[(layer, "v", head, input_v)]] > 0) & reachable_to_input[input_v]
                            reach_attn = reach_q & reach_k & reach_v
                            reachable_to_input[f"attn_output-{layer}-{head}-{input_v}"] = reach_attn
                    
                    if not self.incl_mlp:
                        reach_mlp = [torch.zeros(bz).bool()]
                        for input_v in self.config[layer]["mlp"]:
                            reach_mlp.append( (masks[:, self.mapping_to_param_idx[(layer, "mlp", input_v)]] > 0) & reachable_to_input[input_v] )
                        reach_mlp = torch.stack(reach_mlp).any(dim=0)
                        reachable_to_input[f"mlp-{layer}"] = reach_mlp
                    else:
                        for input_v in self.config[layer]["mlp"]:
                            reach_mlp = (masks[:, self.mapping_to_param_idx[(layer, "mlp", input_v)]] > 0) & reachable_to_input[input_v]
                            reachable_to_input[f"mlp-{layer}-{input_v}"] = reach_mlp
            
            # print("reach input", {k: v[0].item() for k, v in reachable_to_input.items()})

            # remove edges connected to dangling vertices
            for layer, num_head in enumerate(self.num_head_per_layer):
                for head in range(num_head):
                    dangling_head = [torch.ones(bz).bool()]
                    for input_v in self.config[layer]["v"][head]:
                        v_name = f"attn_output-{layer}-{head}-{input_v}"
                        dangling_out = ~reachable_to_input[v_name]
                        dangling_head.append(dangling_out)

                        dangling = dangling_out | ~reachable_to_input[input_v]
                        masks[:, self.mapping_to_param_idx[(layer, "v", head, input_v)]].masked_fill_(dangling, 0)

                    dangling_out = torch.stack(dangling_head).all(dim=0)
                    for activation in ["q", "k"]:
                        for input_v in self.config[layer][activation][head]:
                            dangling = dangling_out | ~reachable_to_input[input_v]
                            masks[:, self.mapping_to_param_idx[(layer, activation, head, input_v)]].masked_fill_(dangling, 0)

                for input_v in self.config[layer]["mlp"]:
                    v_name = f"mlp-{layer}" if not self.incl_mlp else f"mlp-{layer}-{input_v}"
                    dangling_out = ~reachable_to_input[v_name]
                    dangling = dangling_out | ~reachable_to_input[input_v]
                    masks[:, self.mapping_to_param_idx[(layer, "mlp", input_v)]].masked_fill_(dangling, 0)
            
            for input_v in self.config["lm_head"]:
                dangling = ~reachable_to_input[input_v]
                masks[:, self.mapping_to_param_idx[("lm_head", input_v)]].masked_fill_(dangling, 0)

            with torch.no_grad():
                reachable_to_output = defaultdict(lambda : torch.zeros(bz).bool())
                for input_v in self.config["lm_head"]:
                    reachable_to_output[input_v] = masks[:, self.mapping_to_param_idx[("lm_head", input_v)]] > 0
                for layer in range(len(self.num_head_per_layer)-1, -1, -1):
                    for input_v in self.config[layer]["mlp"]:
                        reach_inp = (masks[:, self.mapping_to_param_idx[(layer, "mlp", input_v)]] > 0) & (reachable_to_output[f"mlp-{layer}"] if not self.incl_mlp else reachable_to_output[f"mlp-{layer}-{input_v}"])
                        reachable_to_output[input_v] = reachable_to_output[input_v] | reach_inp

                    for head in range(self.num_head_per_layer[layer]):
                        reach_head = [torch.zeros(bz).bool()]
                        for input_v in self.config[layer]["v"][head]:
                            reach_head.append( reachable_to_output[f"attn_output-{layer}-{head}-{input_v}"] )
                            reach_inp = (masks[:, self.mapping_to_param_idx[(layer, "v", head, input_v)]] > 0) & reachable_to_output[f"attn_output-{layer}-{head}-{input_v}"]
                            reachable_to_output[input_v] = reachable_to_output[input_v] | reach_inp
                        reach_head = torch.stack(reach_head).any(dim=0)
                        for input_v in self.config[layer]["q"][head]:
                            reach_inp = (masks[:, self.mapping_to_param_idx[(layer, "q", head, input_v)]] > 0) & reach_head
                            reachable_to_output[input_v] = reachable_to_output[input_v] | reach_inp
                        for input_v in self.config[layer]["k"][head]:
                            reach_inp = (masks[:, self.mapping_to_param_idx[(layer, "k", head, input_v)]] > 0) & reach_head
                            reachable_to_output[input_v] = reachable_to_output[input_v] | reach_inp
            # print("reach output", {k: v[0].item() for k, v in reachable_to_output.items()})

            # remove edges connected to dangling vertices
            for layer, num_head in enumerate(self.num_head_per_layer):
                for head in range(num_head):
                    dangling_head = [torch.ones(bz).bool()]
                    for input_v in self.config[layer]["v"][head]:
                        v_name = f"attn_output-{layer}-{head}-{input_v}"
                        dangling_out = ~reachable_to_output[v_name]
                        dangling_head.append(dangling_out)

                        dangling = dangling_out | ~reachable_to_output[input_v]
                        masks[:, self.mapping_to_param_idx[(layer, "v", head, input_v)]].masked_fill_(dangling, 0)

                    dangling_out = torch.stack(dangling_head).all(dim=0)
                    for activation in ["q", "k"]:
                        for input_v in self.config[layer][activation][head]:
                            dangling = dangling_out | ~reachable_to_output[input_v]
                            masks[:, self.mapping_to_param_idx[(layer, activation, head, input_v)]].masked_fill_(dangling, 0)

                for input_v in self.config[layer]["mlp"]:
                    v_name = f"mlp-{layer}" if not self.incl_mlp else f"mlp-{layer}-{input_v}"
                    dangling_out = ~reachable_to_output[v_name]
                    dangling = dangling_out | ~reachable_to_output[input_v]
                    masks[:, self.mapping_to_param_idx[(layer, "mlp", input_v)]].masked_fill_(dangling, 0)
            
            for input_v in self.config["lm_head"]:
                dangling = ~reachable_to_output[input_v]
                masks[:, self.mapping_to_param_idx[("lm_head", input_v)]].masked_fill_(dangling, 0)

        return masks
        

# {0: {'k': {0: ['wte', 'wpe']}, 'q': {0: ['wte', 'wpe']}, 'v': {0: ['wte']}, 'mlp': ['wte', 'attn_output-0-0-wte']}, 
#  1: {'k': {0: ['wte', 'attn_output-0-0-wte', 'mlp-0-wte', 'mlp-0-attn_output-0-0-wte']}, 
#      'q': {0: ['wte', 'attn_output-0-0-wte', 'mlp-0-wte', 'mlp-0-attn_output-0-0-wte']}, 
#      'v': {0: ['attn_output-0-0-wte', 'mlp-0-wte', 'mlp-0-attn_output-0-0-wte']}, 
#      'mlp': ['attn_output-1-0-attn_output-0-0-wte', 'attn_output-1-0-mlp-0-wte', 'attn_output-1-0-mlp-0-attn_output-0-0-wte']}, 
# 'lm_head': ['attn_output-1-0-attn_output-0-0-wte', 'attn_output-1-0-mlp-0-wte', 'attn_output-1-0-mlp-0-attn_output-0-0-wte', 'mlp-1-attn_output-1-0-attn_output-0-0-wte', 'mlp-1-attn_output-1-0-mlp-0-wte', 'mlp-1-attn_output-1-0-mlp-0-attn_output-0-0-wte']}

class PruningModelWithHooksFullPaths:
    def __init__(self, model, config, mapping_to_param_idx, incl_mlp, logger=None):
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.config = deepcopy(config)
        self.mapping_to_param_idx = mapping_to_param_idx
        
        self.incl_mlp = incl_mlp
        self.logger = logger
        self.device = self.model.device
        
        self.hooks = []

        self.activations = {}

        self.ln_1 = {}
        self.ln_2 = {}
        self.hooks.append( self.model.transformer.wte.register_forward_hook(partial(
            self.save_activation_hook, activation_type="wte", layer=None
        )) )
        self.hooks.append( self.model.transformer.wpe.register_forward_hook(partial(
            self.save_activation_hook, activation_type="wpe", layer=None
        )) )
        self.hooks.append( self.model.lm_head.register_forward_hook(
            self.lm_head_hook
        ) )
        self.ln_f = self.model.transformer.ln_f
        for layer in range(len(self.model.transformer.h)):
            self.ln_1[layer] = self.model.transformer.h[layer].ln_1
            self.ln_2[layer] = self.model.transformer.h[layer].ln_2
            self.hooks.append( self.model.transformer.h[layer].attn.register_forward_pre_hook(partial(
                self.attn_pre_hook, layer=layer
            ), with_kwargs=False) )
            self.hooks.append( self.model.transformer.h[layer].attn.c_attn.register_forward_hook(partial(
                self.c_attn_hook, layer=layer
            )) )
            self.hooks.append( self.model.transformer.h[layer].attn.c_proj.register_forward_hook(partial(
                self.save_activation_hook, activation_type="attn_output", layer=layer
            )) )
            self.hooks.append( self.model.transformer.h[layer].mlp.register_forward_hook(partial(
                self.mlp_hook, layer=layer
            )) )
        
        self.masks = None
        self.oa_vecs = None

    def set_mask_sampler(self, mask_sampler: MaskSamplerFullPaths):
        self.logger("set mask sampler, should be done with training")
        self.mask_sampler = mask_sampler

    def __call__(self, *args, masks=None, oa_vecs: OptimalAblationVectors =None, **kwargs):
        if masks is not None:
            self.masks = masks
            self.oa_vecs = oa_vecs
        else:
            assert hasattr(self, "mask_sampler") and self.mask_sampler is not None
            assert self.oa_vecs is not None
            if args:
                bz = args[0].size(0)
            else:
                bz = kwargs["input_ids"].size(0)
            self.masks = self.mask_sampler.sample_binary_masks(bz)
        return self.model(*args, **kwargs)

    def linear_layer_norm(self, module: nn.LayerNorm, input, scalar, bias=False):
        out = (input - input.mean(dim=-1, keepdim=True)) / (scalar + module.eps).sqrt() * module.weight.view(1, 1, -1)
        if bias:    # let output vertex oa absorb this
            out = out + module.bias.view(1, 1, -1)
        return out

    def save_activation_hook(self, module, input, output, activation_type, layer):
        if activation_type in ["wte", "wpe"]:
            self.activations[activation_type] = output.detach()
        elif activation_type == "attn_output":
            if not hasattr(self, "activation_name_to_keep") or self.activation_name_to_keep is None:
                return None
            
            input = input[0]
            bz, seq_len = input.shape[:-1]
            d_model = module.weight.size(-1)
            num_heads, head_dim = self.model.transformer.h[layer].attn.num_heads, self.model.transformer.h[layer].attn.head_dim
            attention_by_head = input.view(bz * seq_len, num_heads, head_dim)
            attention_by_head_output = attention_by_head.transpose(0, 1) @ module.weight.view(num_heads, head_dim, d_model)  # num_head, bz*seq_len, d_model
            attention_by_head_output = attention_by_head_output.view(num_heads, bz, seq_len, d_model).unbind(dim=0)

            # assert (((sum(attention_by_head_output) + module.bias) - output).abs() < 1e-3).all(), ((sum(attention_by_head_output) + module.bias) - output)
            
            for head, attn in enumerate(attention_by_head_output):
                if self.activation_name_to_keep[head] is not None:
                    self.activations[f"{activation_type}-{layer}-{head}-{self.activation_name_to_keep[head]}"] = attn
        else:
            raise NotImplementedError()
        
    def attn_pre_hook(self, module, input, layer):
        self.current_layer = layer
        activation_names_to_keep_per_head = [
            [
                self.config[layer]["v"][head][i]
                if i < len(self.config[layer]["v"][head])
                else None
                for head in range(self.model.transformer.h[layer].attn.num_heads)
            ]
            for i in range(max([len(self.config[layer]["v"][head])
                           for head in range(self.model.transformer.h[layer].attn.num_heads)]))
        ]
        for activation_name_to_keep in activation_names_to_keep_per_head:
            self.activation_name_to_keep = activation_name_to_keep
            module.forward(*input)
        self.activation_name_to_keep = None

    def c_attn_hook(self, module, input, output, layer):
        # input.shape == (bsz, seqlen, hidden_size)
        # output.shape == torch.vstack((bsz, seqlen, embed_dim), (bsz, seqlen, embed_dim), (bsz, seqlen, embed_dim))
        # output = query_states, key_states, value_states
        if not hasattr(self, "activation_name_to_keep") or self.activation_name_to_keep is None:
            return None
        assert self.current_layer == layer

        num_heads = self.model.transformer.h[layer].attn.num_heads
        head_dim = self.model.transformer.h[layer].attn.head_dim
        d_model = input[0].size(-1)

        input_activations = []
        for activation in ["q", "k"]:
            for head in range(num_heads):
                summed_act = torch.zeros_like(input[0])
                for activation_name in self.config[layer][activation][head]:
                    coef = self.masks[:, self.mapping_to_param_idx[(layer, activation, head, activation_name)]]

                    first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
                    oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
                    second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                                oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
                    summed_act = summed_act + first_term + second_term.unsqueeze(1)
                oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[(layer, activation, head)]]
                summed_act = summed_act + oa.view(1, 1, -1)

                LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[(layer, activation, head)]].exp()
                input_activations.append(self.linear_layer_norm(self.ln_1[layer], summed_act, LN_var))
                # input_activations.append(self.ln_1[layer](summed_act))
        
        for head in range(num_heads):
            if hasattr(self, "activation_name_to_keep") and self.activation_name_to_keep is not None and self.activation_name_to_keep[head] is not None:
                activation_name = self.activation_name_to_keep[head]
                coef = self.masks[:, self.mapping_to_param_idx[(layer, "v", head, activation_name)]]
                
                first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
                oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
                second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                            oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
                summed_act = first_term + second_term.unsqueeze(1)

                LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[(layer, "v", head, activation_name)]].exp()
                input_act = self.linear_layer_norm(self.ln_1[layer], summed_act, LN_var, bias=False)    # avoid adding bias term multiple times, and leave this to output oa vec

                # scalar = self.oa_vecs.v_edge_scalar[self.oa_vecs.to_v_scalar_idx[(layer, "v", head, activation_name)]].exp()  # to cancel scaling effect of LN
                # input_act = self.ln_1[layer](summed_act) * scalar
            else:
                input_act = torch.zeros_like(input[0])
            input_activations.append(input_act)

    
        input_activations = torch.stack(input_activations)

        output_activations = input_activations.flatten(start_dim=1, end_dim=2) @ \
                             module.weight.view(d_model, num_heads * 3, head_dim).transpose(0, 1)
        output_activations = output_activations.transpose(0, 1).contiguous().view(*input[0].size()[:2], num_heads * head_dim * 3)
        bias = module.bias.clone()
        bias[num_heads*head_dim*2:] = 0  # avoid adding bias multiple times
        output_activations += bias.view(1, 1, -1)
        
        return output_activations
    
    def mlp_hook(self, module, input, output, layer):
        if len(self.config[layer]["mlp"]) == 0:
            return None
        
        if not self.incl_mlp:
            summed_act = torch.zeros_like(input[0])
            for activation_name in self.config[layer]["mlp"]:
                coef = self.masks[:, self.mapping_to_param_idx[(layer, "mlp", activation_name)]]

                first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
                oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
                second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                                    oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
                summed_act = summed_act + first_term + second_term.unsqueeze(1)

            oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[(layer, "mlp")]]
            summed_act = summed_act + oa.view(1, 1, -1)

            LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[(layer, "mlp")]].exp()
            input_activation = self.linear_layer_norm(self.ln_2[layer], summed_act, LN_var)
            # input_activation = self.ln_2[layer](summed_act)

            output = module.forward(input_activation)
            self.activations[f"mlp-{layer}"] = output
            return output
        
        else:
            for activation_name in self.config[layer]["mlp"]:
                coef = self.masks[:, self.mapping_to_param_idx[(layer, "mlp", activation_name)]]
                first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
                oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
                second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                                    oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
                summed_act = first_term + second_term.unsqueeze(1)

                LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[(layer, "mlp", activation_name)]].exp()
                input_activation = self.linear_layer_norm(self.ln_2[layer], summed_act, LN_var)

                output = self.oa_vecs.MLPs[f"{layer} {activation_name}"](input_activation)
                self.activations[f"mlp-{layer}-{activation_name}"] = output

            return None

    def lm_head_hook(self, module, input, output):
        summed_act = torch.zeros_like(input[0])
        for activation_name in self.config["lm_head"]:
            coef = self.masks[:, self.mapping_to_param_idx[("lm_head", activation_name)]]

            first_term = self.activations[activation_name] * coef.view(-1, 1, 1)
            oa = self.oa_vecs.input_vertex_oa[self.oa_vecs.to_in_oa_idx[activation_name]]
            second_term = oa.unsqueeze(0) * ((1 - coef) * (coef < 0.001).float()).unsqueeze(1) + \
                                oa.unsqueeze(0).detach() * ((1 - coef) * (coef >= 0.001).float()).unsqueeze(1)
            summed_act = summed_act + first_term + second_term.unsqueeze(1)

        oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[("lm_head",)]]
        summed_act = summed_act + oa.view(1, 1, -1)

        LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[("lm_head",)]].exp()
        input_activation = self.linear_layer_norm(self.ln_f, summed_act, LN_var)
        # input_activation = self.ln_f(summed_act)

        output = module.forward(input_activation)
        return output

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# {0: {'v': {0: ['wte']}, 'mlp': ['wte', 'attn_output-0-0-wte'], 'qk': {0: [('wpe', 'wpe')]}}, 
#  1: {'v': {0: ['attn_output-0-0-wte']}, 
#      'mlp': [], 
#      'qk': {0: [('wte', 'wte'), ('wte', 'attn_output-0-0-wte'), ('wte', 'mlp-0'), ('attn_output-0-0-wte', 'wte'), ('attn_output-0-0-wte', 'attn_output-0-0-wte'), ('attn_output-0-0-wte', 'mlp-0'), ('mlp-0', 'wte'), ('mlp-0', 'attn_output-0-0-wte'), ('mlp-0', 'mlp-0')]}}, 
# 'lm_head': ['attn_output-1-0-attn_output-0-0-wte']}

class MaskSamplerForQK(MaskSampler):
    def __init__(self, config: dict):
        nn.Module.__init__(self)

        param_idx = 0
        mapping_to_param_idx = {}
        key_names = set()
        for k1 in config:
            if type(config[k1]) == dict:
                for k2 in config[k1]:
                    if k2 == "qk" or k2 == "k":
                        for k3 in config[k1][k2]:
                            for item in config[k1][k2][k3]:
                                mapping_to_param_idx[(k1, k3, item)] = param_idx
                                param_idx += 1
                                if k2 == "k":
                                    key_names.add((k1, k3, item))
        
        self.key_names = list(key_names)
        self.config = config
        self.mapping_to_param_idx = mapping_to_param_idx
        self.sample_params = nn.Parameter(torch.ones(param_idx))

        num_head_per_layer = []
        for i in range(len(config)-1):
            num_head_per_layer.append(len(config[i]["k"]))
        self.num_head_per_layer = num_head_per_layer

        self.window_function = lambda x: x * (1-x)

    def prune_dangling_edges(self, masks):
        return masks

class OptimalQueryBiasVectors(nn.Module):
    def __init__(self, key_names, d_head, oa_vecs: OptimalAblationVectors):
        super().__init__()
        self.key_names = key_names
        self.q_bias_term = nn.Parameter(torch.zeros(len(key_names), d_head))
        self.to_q_bias = {item: i for i, item in enumerate(key_names)}

        self.output_vertex = [item for item in oa_vecs.output_vertex if len(item) <= 2] # only need for mlp (if not split) and lm_head
        self.to_out_oa_idx = {item: i for i, item in enumerate(self.output_vertex)}
        output_vertex_oa = torch.zeros(len(self.output_vertex), oa_vecs.output_vertex_oa.size(1))
        for item in self.output_vertex:
            output_vertex_oa[self.to_out_oa_idx[item]] = oa_vecs.output_vertex_oa[oa_vecs.to_out_oa_idx[item]].data
        self.output_vertex_oa = nn.Parameter(output_vertex_oa)

        self.LN_vertex = oa_vecs.LN_vertex
        self.LN_var = oa_vecs.LN_var
        self.to_LN_idx = oa_vecs.to_LN_idx

        self.MLP_vertex = oa_vecs.MLP_vertex
        if hasattr(oa_vecs, "MLPs"):
            self.MLPs = oa_vecs.MLPs


class PruningModelWithHooksForQK:
    def __init__(self, model, config, mapping_to_param_idx, incl_mlp, logger):
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.config = deepcopy(config)
        self.mapping_to_param_idx = mapping_to_param_idx
        
        self.incl_mlp = incl_mlp
        self.logger = logger
        self.device = self.model.device
        
        self.hooks = []

        self.activations = {}

        self.ln_1 = {}
        self.ln_2 = {}
        self.hooks.append( self.model.transformer.wte.register_forward_hook(partial(
            self.save_activation_hook, activation_type="wte", layer=None
        )) )
        self.hooks.append( self.model.transformer.wpe.register_forward_hook(partial(
            self.save_activation_hook, activation_type="wpe", layer=None
        )) )
        self.hooks.append( self.model.lm_head.register_forward_hook(
            self.lm_head_hook
        ) )
        self.ln_f = self.model.transformer.ln_f
        for layer in range(len(self.model.transformer.h)):
            self.ln_1[layer] = self.model.transformer.h[layer].ln_1
            self.ln_2[layer] = self.model.transformer.h[layer].ln_2
            self.hooks.append( self.model.transformer.h[layer].attn.register_forward_pre_hook(partial(
                self.attn_pre_hook, layer=layer
            ), with_kwargs=False) )
            self.hooks.append( self.model.transformer.h[layer].mlp.register_forward_hook(partial(
                self.mlp_hook, layer=layer
            )) )
        
        self.masks = None
        self.oa_vecs = None

    def set_mask_sampler(self, mask_sampler: MaskSamplerFullPaths):
        self.logger("set mask sampler, should be done with training")
        self.mask_sampler = mask_sampler
    
    def set_converted_mlp(self, converted_mlp: dict):
        self.logger("set converted mlp, should be done with training")
        self.converted_mlp = converted_mlp
        self.variables = {}
    
    def remove_converted_mlp(self):
        del self.converted_mlp
        del self.variables

    def __call__(self, *args, masks=None, oa_vecs: OptimalAblationVectors =None, **kwargs):
        if masks is not None:
            self.masks = masks
            self.oa_vecs = oa_vecs
        else:
            assert hasattr(self, "mask_sampler") and self.mask_sampler is not None
            assert self.oa_vecs is not None
            if args:
                bz = args[0].size(0)
            else:
                bz = kwargs["input_ids"].size(0)
            self.masks = self.mask_sampler.sample_binary_masks(bz)
        return self.model(*args, **kwargs)

    def linear_layer_norm(self, module: nn.LayerNorm, input, scalar, bias=False):
        out = (input - input.mean(dim=-1, keepdim=True)) / (scalar + module.eps).sqrt() * module.weight.view(1, 1, -1)
        if bias:    # let output vertex oa absorb this
            out = out + module.bias.view(1, 1, -1)
        return out

    def save_activation_hook(self, module, input, output, activation_type, layer):
        if activation_type in ["wte", "wpe"]:
            self.activations[activation_type] = output.detach()
            if hasattr(self, "converted_mlp"):
                assert isinstance(input, tuple)
                self.variables[activation_type] = F.one_hot(input[0], 
                        num_classes=self.model.config.vocab_size if activation_type == "wte" else self.model.config.max_position_embeddings
                        ).float()
        else:
            raise NotImplementedError()
        
    def attn_pre_hook(self, module, input, layer):
        self.current_layer = layer
        activation_names_to_keep_per_head = [
            [
                self.config[layer]["v"][head][i]
                if i < len(self.config[layer]["v"][head])
                else None
                for head in range(self.model.transformer.h[layer].attn.num_heads)
            ]
            for i in range(max([len(self.config[layer]["v"][head])
                           for head in range(self.model.transformer.h[layer].attn.num_heads)]))
        ]
        for activation_name_to_keep in activation_names_to_keep_per_head:
            self.activation_name_to_keep = activation_name_to_keep
            self.attention_forward(module, layer=layer)
        self.activation_name_to_keep = None

    def attention_forward(self, module, layer=None):
        assert self.current_layer == layer
        assert hasattr(self, "activation_name_to_keep") and self.activation_name_to_keep is not None

        num_heads = module.num_heads
        head_dim = module.head_dim
        split_size = module.split_size
        batch_size, seq_len, d_model = self.activations["wte"].size()
        device = self.activations["wte"].device

        # to compute v
        input_activations = []
        for head in range(num_heads):
            if self.activation_name_to_keep[head] is not None:
                activation_name = self.activation_name_to_keep[head]
                input_act = self.activations[activation_name]

                LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[(layer, "v", head, activation_name)]].exp()
                input_act = self.linear_layer_norm(self.ln_1[layer], input_act, LN_var, bias=False)    # avoid adding bias term multiple times, and leave this to output oa vec
            else:
                input_act = torch.zeros_like(self.activations["wte"])
            input_activations.append(input_act)

        input_activations = torch.stack(input_activations)

        output_activations = input_activations.flatten(start_dim=1, end_dim=2) @ \
                             module.c_attn.weight[:, split_size*2:].view(d_model, num_heads, head_dim).transpose(0, 1)
        value_states = output_activations.transpose(0, 1).contiguous().view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        # output_activations += module.bias[module.split_size*2:].view(1, 1, -1)

        attn_weights = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=device)
        for head in range(num_heads):
            for activation_name_to_keep_q, activation_name_to_keep_k in self.config[layer]["qk"][head]:
                LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[(layer, "q", head)]].exp()
                q_act = self.linear_layer_norm(self.ln_1[layer], self.activations[activation_name_to_keep_q], LN_var, bias=False)
                LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[(layer, "k", head)]].exp()
                k_act = self.linear_layer_norm(self.ln_1[layer], self.activations[activation_name_to_keep_k], LN_var, bias=False)

                query_states = q_act @ module.c_attn.weight[:, head*head_dim: (head+1)*head_dim].unsqueeze(0)
                key_states = k_act @ module.c_attn.weight[:, split_size+head*head_dim: split_size+(head+1)*head_dim].unsqueeze(0)
                product = torch.matmul(query_states, key_states.transpose(-1, -2))
                 
                coef = self.masks[:, self.mapping_to_param_idx[(layer, head, (activation_name_to_keep_q, activation_name_to_keep_k))]]
                first_term = product * coef.view(-1, 1, 1)
                attn_weights[:, head, :, :] += first_term

            for k_name in self.config[layer]["k"][head]:
                k_act = self.activations[k_name]
                LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[(layer, "k", head)]].exp()
                k_act = self.linear_layer_norm(self.ln_1[layer], k_act, LN_var, bias=False)
                key_states = k_act @ module.c_attn.weight[:, split_size+head*head_dim: split_size+(head+1)*head_dim].unsqueeze(0)

                q_bias_term = self.oa_vecs.q_bias_term[self.oa_vecs.to_q_bias[(layer, head, k_name)]]
                coef = self.masks[:, self.mapping_to_param_idx[(layer, head, k_name)]].view(-1, 1, 1)
                q_bias_term = q_bias_term.view(1, 1, -1) * coef * (coef > 0.999).float() + \
                    q_bias_term.view(1, 1, -1).detach() * coef * (coef <= 0.999).float()
                product = torch.matmul(q_bias_term, key_states.transpose(-1, -2))

                attn_weights[:, head, :, :] += product

                
        if module.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value_states.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # if only "normal" attention layer implements causal mask
        query_length, key_length = seq_len, seq_len # query_states.size(-2), key_states.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value_states.dtype)
        # attn_weights = module.attn_dropout(attn_weights)  # let's remove all dropout since we do not consider it (an additional scalar) when computing heatmap
        self.activations[f"attn_weights-{layer}"] = attn_weights.detach()

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)

        attention_by_head = attn_output.contiguous().view(batch_size * seq_len, num_heads, head_dim)
        attention_by_head_output = attention_by_head.transpose(0, 1) @ module.c_proj.weight.view(num_heads, head_dim, d_model)  # num_head, bz*seq_len, d_model
        # attention_by_head_output = module.resid_dropout(attention_by_head_output)
        attention_by_head_output = attention_by_head_output.view(num_heads, batch_size, seq_len, d_model).unbind(dim=0)

        for head, attn in enumerate(attention_by_head_output):
            if self.activation_name_to_keep[head] is not None:
                self.activations[f"attn_output-{layer}-{head}-{self.activation_name_to_keep[head]}"] = attn
                if hasattr(self, "converted_mlp") and self.activation_name_to_keep[head] in self.variables:
                    self.variables[f"attn_output-{layer}-{head}-{self.activation_name_to_keep[head]}"] = \
                        self.activations[f"attn_weights-{layer}"][:, head] @ self.variables[self.activation_name_to_keep[head]]

        return None
    
    def mlp_hook(self, module, input, output, layer):
        if len(self.config[layer]["mlp"]) == 0:
            return None
        
        if not self.incl_mlp:
            if hasattr(self, "converted_mlp") and f"mlp-{layer}" in self.converted_mlp:
                prods = [self.variables[activation_name] for activation_name in self.config[layer]["mlp"]]
                primitive, C = self.converted_mlp[f"mlp-{layer}"]
                Y = get_mlp_primitives_multi_source(prods, primitive=primitive)
                self.variables[f"mlp-{layer}"] = Y
                output = Y @ C.unsqueeze(0)
            else:
                summed_act = torch.zeros_like(input[0])
                for activation_name in self.config[layer]["mlp"]:
                    summed_act = summed_act + self.activations[activation_name]

                oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[(layer, "mlp")]]
                summed_act = summed_act + oa.view(1, 1, -1)

                LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[(layer, "mlp")]].exp()
                input_activation = self.linear_layer_norm(self.ln_2[layer], summed_act, LN_var)

                output = module.forward(input_activation)
            self.activations[f"mlp-{layer}"] = output
            return output
        
        else:
            for activation_name in self.config[layer]["mlp"]:
                if hasattr(self, "converted_mlp") and f"mlp-{layer}-{activation_name}" in self.converted_mlp:
                    primitive, C = self.converted_mlp[f"mlp-{layer}-{activation_name}"]
                    Y = get_mlp_primitives(self.variables[activation_name], primitive=primitive)
                    self.variables[f"mlp-{layer}-{activation_name}"] = Y
                    output = Y @ C.unsqueeze(0)
                else:
                    LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[(layer, "mlp", activation_name)]].exp()
                    input_activation = self.linear_layer_norm(self.ln_2[layer], self.activations[activation_name], LN_var)

                    output = self.oa_vecs.MLPs[f"{layer} {activation_name}"](input_activation)
                self.activations[f"mlp-{layer}-{activation_name}"] = output

            return None

    def lm_head_hook(self, module, input, output):
        summed_act = torch.zeros_like(input[0])
        for activation_name in self.config["lm_head"]:
            summed_act = summed_act + self.activations[activation_name]

        oa = self.oa_vecs.output_vertex_oa[self.oa_vecs.to_out_oa_idx[("lm_head",)]]
        summed_act = summed_act + oa.view(1, 1, -1)

        LN_var = self.oa_vecs.LN_var[self.oa_vecs.to_LN_idx[("lm_head",)]].exp()
        input_activation = self.linear_layer_norm(self.ln_f, summed_act, LN_var)

        output = module.forward(input_activation)
        return output

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def convert_config_fullpaths_(current_config, num_heads_per_layer):
    # no attention_bias, no ln_1, ln_2
    num_layers = len(num_heads_per_layer)
    for layer in range(num_layers):
        for head in range(num_heads_per_layer[layer]):
            for attn_act in ["k", "q", "v"]:
                current_config[layer][attn_act][head] = [
                    act for act in ["wte", "wpe"] if act in current_config[layer][attn_act][head]
                ] + \
                [f"attn_output-{l}-{h}-{prev_path}"
                    for l in range(layer)
                    for h in range(num_heads_per_layer[l])
                    for prev_path in current_config[l]["v"][h]
                if f"attn_output-{l}-{h}" in current_config[layer][attn_act][head]] + \
                [f"mlp-{l}"
                    for l in range(layer)
                    if f"mlp-{l}" in current_config[layer][attn_act][head]]
        current_config[layer]["mlp"] = [
                act for act in ["wte", "wpe"] if act in current_config[layer]["mlp"]
                ] + \
                [f"attn_output-{l}-{h}-{prev_path}"
                    for l in range(layer + 1)
                    for h in range(num_heads_per_layer[l])
                    for prev_path in current_config[l]["v"][h]
                    if f"attn_output-{l}-{h}" in current_config[layer]["mlp"]] + \
                [f"mlp-{l}"
                for l in range(layer)
                if f"mlp-{l}" in current_config[layer]["mlp"]]
    current_config["lm_head"] = [
                act for act in ["wte", "wpe"] if act in current_config["lm_head"]
                ] + \
                [f"attn_output-{l}-{h}-{prev_path}"
                    for l in range(num_layers)
                    for h in range(num_heads_per_layer[l])
                    for prev_path in current_config[l]["v"][h]
                    if f"attn_output-{l}-{h}" in current_config["lm_head"]] + \
                [f"mlp-{l}"
                for l in range(num_layers)
                if f"mlp-{l}" in current_config["lm_head"]]
    

def convert_config_fullpaths_incl_mlp_(current_config, num_heads_per_layer):
    # no attention_bias, no ln_1, ln_2
    num_layers = len(num_heads_per_layer)
    for layer in range(num_layers):
        for head in range(num_heads_per_layer[layer]):
            for attn_act in ["k", "q", "v"]:
                current_config[layer][attn_act][head] = [
                    act for act in ["wte", "wpe"] if act in current_config[layer][attn_act][head]
                ] + \
                [f"attn_output-{l}-{h}-{prev_path}"
                    for l in range(layer)
                    for h in range(num_heads_per_layer[l])
                    for prev_path in current_config[l]["v"][h]
                if f"attn_output-{l}-{h}" in current_config[layer][attn_act][head]] + \
                [f"mlp-{l}-{prev_path}"
                    for l in range(layer)
                    for prev_path in current_config[l]["mlp"]
                    if f"mlp-{l}" in current_config[layer][attn_act][head]]
        current_config[layer]["mlp"] = [
                act for act in ["wte", "wpe"] if act in current_config[layer]["mlp"]
                ] + \
                [f"attn_output-{l}-{h}-{prev_path}"
                    for l in range(layer + 1)
                    for h in range(num_heads_per_layer[l])
                    for prev_path in current_config[l]["v"][h]
                    if f"attn_output-{l}-{h}" in current_config[layer]["mlp"]] + \
                [f"mlp-{l}-{prev_path}"
                    for l in range(layer)
                    for prev_path in current_config[l]["mlp"]
                    if f"mlp-{l}" in current_config[layer]["mlp"]]
    current_config["lm_head"] = [
                act for act in ["wte", "wpe"] if act in current_config["lm_head"]
                ] + \
                [f"attn_output-{l}-{h}-{prev_path}"
                    for l in range(num_layers)
                    for h in range(num_heads_per_layer[l])
                    for prev_path in current_config[l]["v"][h]
                    if f"attn_output-{l}-{h}" in current_config["lm_head"]] + \
                [f"mlp-{l}-{prev_path}"
                    for l in range(num_layers)
                    for prev_path in current_config[l]["mlp"]
                    if f"mlp-{l}" in current_config["lm_head"]]


def convert_full_paths_config_to_prune_inside_kq_config_(current_config, num_heads_per_layer):
    for layer in range(len(num_heads_per_layer)):
        current_config[layer]["qk"] = {}
        for head in range(num_heads_per_layer[layer]):
            current_config[layer]["qk"][head] = [
                (path_in_q, path_in_k)
                for path_in_q in current_config[layer]["q"][head]
                for path_in_k in current_config[layer]["k"][head]
            ]
        del current_config[layer]["q"]


def convert_mask_to_config_(masks, config, mapping_to_param_idx):
    assert masks.dim() == 1
    assert ((masks != 0) & (masks != 1)).sum().item() == 0
    for k1 in config:
        if type(config[k1]) == dict:
            for k2 in config[k1]:
                if type(config[k1][k2]) == dict:
                    for k3 in config[k1][k2]:
                        new_lis = []
                        for item in config[k1][k2][k3]:
                            if masks[mapping_to_param_idx[(k1, k2, k3, item)]].item() == 1:
                                new_lis.append(item)
                        config[k1][k2][k3] = new_lis
                elif type(config[k1][k2]) == list:
                    new_lis = []
                    for item in config[k1][k2]:
                        if masks[mapping_to_param_idx[(k1, k2, item)]].item() == 1:
                            new_lis.append(item)
                    config[k1][k2] = new_lis
        elif type(config[k1]) == list:
            new_lis = []
            for item in config[k1]:
                if masks[mapping_to_param_idx[(k1, item)]].item() == 1:
                    new_lis.append(item)
            config[k1] = new_lis

def convert_mask_to_config_qk_(masks, config, mapping_to_param_idx):
    assert masks.dim() == 1
    assert ((masks != 0) & (masks != 1)).sum().item() == 0
    for k1 in config:
        if type(config[k1]) == dict:
            for k2 in config[k1]:
                if k2 == "qk" or k2 == "k":
                    for k3 in config[k1][k2]:
                        new_lis = []
                        for item in config[k1][k2][k3]:
                            if masks[mapping_to_param_idx[(k1, k3, item)]].item() == 1:
                                new_lis.append(item)
                        config[k1][k2][k3] = new_lis

# TODO debug, cascaded
# {0: {'k': {0: [], 1: []}, 'v': {0: ['wte'], 1: []}, 'mlp': ['wte', 'attn_output-0-0-wte'], 'qk': {0: [], 1: []}}, 1: {'k': {0: [], 1: []}, 'v': {0: [], 1: []}, 'mlp': ['mlp-0-wte'], 'qk': {0: [], 1: []}}, 2: {'k': {0: [], 1: []}, 'v': {0: [], 1: ['mlp-1-mlp-0-wte']}, 'mlp': [], 'qk': {0: [], 1: []}}, 3: {'k': {0: [], 1: []}, 'v': {0: [], 1: []}, 'mlp': [], 'qk': {0: [], 1: []}}, 'lm_head': ['attn_output-2-1-mlp-1-mlp-0-wte']}
def remove_other_edges_after_QK_pruning_(config, split_mlp, logger=print):
    nodes_needed = set()
    for k1 in config:
        if type(config[k1]) == dict:    # k1=layer
            for k2 in config[k1]:
                if type(config[k1][k2]) == dict: # k2=k/qk/v
                    for k3 in config[k1][k2]:
                        if type(config[k1][k2][k3]) == list:
                            for item in config[k1][k2][k3]:
                                if type(item) != str:
                                    nodes_needed.add(item[0])
                                    nodes_needed.add(item[1])
                                else:
                                    nodes_needed.add(item)
                        else:
                            raise RuntimeError
                elif type(config[k1][k2]) == list:  # k2=mlp
                    for item in config[k1][k2]:
                        nodes_needed.add(item)
                else:
                    raise RuntimeError
        elif type(config[k1]) == list:  # k1=lm_head
            for item in config[k1]:
                nodes_needed.add(item)
        else:
            raise RuntimeError
    
    # remove not needed nodes
    for k1 in config:
        if type(config[k1]) == dict:    # k1=layer
            for k2 in config[k1]:
                if k2 == "v":
                    for k3 in config[k1][k2]:   # k3=head
                        assert type(config[k1][k2][k3]) == list
                        new_lis = []
                        for item in config[k1][k2][k3]:
                            node = f"attn_output-{k1}-{k3}-{item}"
                            if node in nodes_needed:
                                new_lis.append(item)
                            else:
                                logger(node, "is removed")
                        config[k1][k2][k3] = new_lis
                elif k2 == "mlp":
                    if split_mlp:
                        new_lis = []
                        for item in config[k1][k2]:
                            node = f"mlp-{k1}-{item}"
                            if node in nodes_needed:
                                new_lis.append(item)
                            else:
                                logger(node, "is removed")
                        config[k1][k2] = new_lis
                    else:
                        node = f"mlp-{k1}"
                        if node not in nodes_needed and len(config[k1][k2]) > 0:
                            config[k1][k2] = []
                            logger(node, "is removed")

@torch.no_grad()
def capture_LN_var(dataset, model, collator, device, ignore_ids):
    num_layers = len(model.transformer.h)
    batch_size = 64
    hooks = []
    var = torch.zeros(num_layers*2+1, device=device)
    def save_hook(module, input, output, idx):
        var[idx] += input[0].var(dim=-1)[~mask].mean()
    
    hooks.append( model.transformer.ln_f.register_forward_hook(partial(
            save_hook, idx=-1
        )) )
    for layer in range(num_layers):
        hooks.append( model.transformer.h[layer].ln_1.register_forward_hook(partial(
            save_hook, idx=layer*2
        )) )
        hooks.append( model.transformer.h[layer].ln_2.register_forward_hook(partial(
            save_hook, idx=layer*2+1
        )) )
    
    inputs = []
    step_idx = 0
    for item in dataset:
        inputs.append(item)
        if len(inputs) == batch_size:
            batch = collator(inputs)
            batch = {k: v.to(device) for k, v in batch.items()}
            mask = torch.stack([batch["input_ids"] == i for i in ignore_ids]).any(dim=0)
            batch.pop("labels")
            model(**batch)

            inputs = []
            step_idx += 1
            if step_idx == 10:
                break
    for hook in hooks:
        hook.remove()

    var /= 10
    return var
