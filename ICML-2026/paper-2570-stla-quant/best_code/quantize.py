import math
import functools
from typing import Any
import torch
import torch.nn as nn

from aespa import Aespa
from quantizer import MinMaxQuantizer
from utils import * 
from quant_utils import *

from model_utils import get_transformer_blocks

QKV_NAMES = {"query": "self_attn.q_proj", "key": "self_attn.k_proj", "value": "self_attn.v_proj"}

@torch.no_grad()
def aespa_fwrd(model, calib_data, qconfigs, aespa_opts: dict, hyperparams: dict):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    transformer_blocks, model_type = get_transformer_blocks(model)
    org_dtype = next(iter(model.parameters())).dtype

    # cache inputs of the first Transformer block for fast quantization
    inps, block_kwargs = input_caching(model, calib_data, dev)
    # Initialize FP inputs for dXXT calculation
    fp_inps = inps.clone()
    rot_emb = get_rot_emb(model.config, block_kwargs['position_ids']) if model_type == "llama" else None

    # quantize each Transformer block
    quantizers = {}
    round_optim_opts = aespa_opts['round_optim']
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads if hasattr(model.config, 'num_key_value_heads') else n_heads
    for idx_block in range(len(transformer_blocks)):
        print('-' * 50)
        print(f'>>>> Quantize {idx_block+1}-th Transformer Block.... ({idx_block+1}/{len(transformer_blocks)})')
        transformer_block = transformer_blocks[idx_block].to(device=dev, dtype=torch.float32)

        fp_layers = find_layers(transformer_block, layers=[nn.Linear])
        wrappers = {}
        for name, fp_layer in fp_layers.items():
            wrappers[name] = Aespa(fp_layer)
            wrappers[name].i, wrappers[name].name = idx_block, name
            wrappers[name].quantizer = MinMaxQuantizer()
            # Only for hyperparameter initialization
            wrappers[name].quantizer.configure(qconfigs['w_bits'], per_channel=True, sym=qconfigs['w_sym'])
            wrappers[name].quantizer.find_params(wrappers[name].layer.weight.data)

        print(f">>> Compute Hessian")
        compute_Hessian(transformer_block, wrappers, inps, fp_inps, block_kwargs, n_heads, n_kv_heads, rot_emb, aespa_opts['block_v'], round_optim_opts["learn_rounding"], round_optim_opts['comp_method'])
        # if model_type == "llama":
        #     compute_Hessian_llama(transformer_block, wrappers, inps, block_kwargs, model.config.num_attention_heads, model.config.num_key_value_heads, round_optim_opts["learn_rounding"], aespa_opts['block_v'], rot_emb)
        # else:
        #     compute_Hessian(transformer_block, wrappers, inps, block_kwargs, model.config.num_attention_heads, model.config.num_key_value_heads, rot_emb, aespa_opts['block_v'], round_optim_opts["learn_rounding"])

        # print(f">>> Refine qparams using Hessian")
        # refine_qparams_with_hessian(wrappers, idx_block, model_type, aespa_opts['use_zfold'], hyperparams, model.config)

        print(f">>> Optimize integer weights")
        if round_optim_opts["learn_rounding"]:
            print('+---------------------------+----------------+-----------------+-----------------+')
            print('|           Layer           |   iterations   |   Recon. Loss   |   Round. Loss   |')
            print('+===========================+================+=================+=================+')
        for name, wrapper in wrappers.items():
            wrapper.quant(round_optim_opts, hyperparams)
            quantizers['%d.%s' % (idx_block, name)] = wrapper.quantizer
            wrapper.free()

        print(f">>> Update inputs for the next Transformer block")
        for j in range(len(inps)):
            inps[j] = transformer_block(inps[j].unsqueeze(0), **block_kwargs)[0]
        
        transformer_blocks[idx_block] = transformer_block.to(device="cpu", dtype=org_dtype)
        del transformer_block
        del wrappers 
        torch.cuda.empty_cache()
    
    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def input_caching(model, calib_data, dev):
    inps = torch.zeros((len(calib_data), model.seqlen, model.config.hidden_size), dtype=torch.float32, device=dev)
    block_kwargs = {}

    transformer_blocks, model_type = get_transformer_blocks(model)
    if model_type == "llama":
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    elif model_type == "opt":
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if model.model.decoder.project_in is not None:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        if model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    elif model_type == "bloom":
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    else:
        raise NotImplementedError(f"{model_type} models are not supported yet")

    class InputCatcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.n_data = 0

        def forward(self, inp, **kwargs):
            inps[self.n_data] = inp[0]
            if self.n_data == 0:
                for key, value in kwargs.items():
                    block_kwargs[key] = value
            self.n_data += 1
            raise ValueError
    
    transformer_blocks[0] = transformer_blocks[0].to(dev)
    transformer_blocks[0] = InputCatcher(transformer_blocks[0])
    for batch in calib_data:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    transformer_blocks[0] = transformer_blocks[0].module.to("cpu")
    if model_type == "llama":
        model.model.embed_tokens = model.model.embed_tokens.to("cpu")
        model.model.norm = model.model.norm.to("cpu")
    elif model_type == "opt":
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to("cpu") 
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to("cpu")
        if model.model.decoder.project_in is not None:
            model.model.decoder.project_in = model.model.decoder.project_in.to("cpu")
        if model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.to("cpu")
    elif model_type == "bloom":
        model.transformer.word_embeddings = model.transformer.word_embeddings.to("cpu")
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to("cpu")
        block_kwargs['alibi'] = block_kwargs['alibi'].to(dtype=torch.float32)
    else:
        raise NotImplementedError(f"{model_type} models are not supported yet")

    torch.cuda.empty_cache()

    return inps, block_kwargs


# for llms exploiting rotary embeddings (e.g., llama)
def get_rot_emb(config, position_ids):
    from modeling_llama_custom import LlamaRotaryEmbedding

    head_dim = config.hidden_size // config.num_attention_heads
    half_head_dim = head_dim // 2
    seqlen = position_ids.shape[-1]

    rotary_emb = LlamaRotaryEmbedding(
        dim=head_dim,
        max_position_embeddings=seqlen,
        base=config.rope_theta
    )
    cos, sin = rotary_emb(torch.rand([1], dtype=torch.float32), position_ids.cpu())
    cos, sin = cos.squeeze(), sin.squeeze()

    rot_emb = torch.zeros(*(seqlen, head_dim, head_dim), dtype=cos.dtype, device=cos.device)
    rot_emb[:, :half_head_dim, :half_head_dim] = torch.diag_embed(cos[:, :half_head_dim])
    rot_emb[:, :half_head_dim, half_head_dim:] = -torch.diag_embed(sin[:, :half_head_dim])
    rot_emb[:, half_head_dim:, :half_head_dim] = torch.diag_embed(sin[:, :half_head_dim])
    rot_emb[:, half_head_dim:, half_head_dim:] = torch.diag_embed(cos[:, :half_head_dim])

    rot_emb = rot_emb.unsqueeze(dim=1)
    return rot_emb


@torch.no_grad()
def compute_Hessian(transformer_block, wrappers, inps, fp_inps, block_kwargs, n_heads, n_kv_heads, rot_emb, block_v=True, learn_rounding=True, comp_method='GPTAQ'):
    if comp_method=='GPTAQ':
        # # floating a for block_v
        # if block_v:
        #     attn_probs_fp = []
        #     block_kwargs_temp = block_kwargs.copy()
        #     block_kwargs_temp['output_attentions'] = True  # need to get attention weights
        #     for j in range(len(inps)):
        #         attn_probs_fp.append(transformer_block(fp_inps[j].unsqueeze(0), **block_kwargs_temp)[-1].to("cpu"))
        
        # Catch FP inputs for dXXT
        handles = []
        for name, wrapper in wrappers.items():
            handles.append(transformer_block.get_submodule(name).register_forward_hook(wrapper.cache_fp_block_input))
        for j in range(len(inps)):
            fp_inps[j] = transformer_block(fp_inps[j].unsqueeze(0), **block_kwargs)[0]
        for h in handles:
            h.remove()

    handles = []
    for name, wrapper in wrappers.items():
        if name not in [QKV_NAMES["query"], QKV_NAMES["key"]]:  # Q, K, V share inputs.
            handles.append(transformer_block.get_submodule(name).register_forward_hook(wrapper.compute_cov_in_batch))
    
    if block_v:
        hook_inps = []
        def save_inps(module, inps, outs: Any):        
            nonlocal hook_inps
            hook_inps.append(inps[0])

        block_kwargs = block_kwargs.copy()
        block_kwargs['output_attentions'] = True  # need to get attention weights
        H_value = 0
        n_data = 0
        if comp_method == 'GPTAQ':
            dXXT_value = 0  # Initialize dXXT accumulator for Value layer
            hook_fp_inps = list(wrappers[QKV_NAMES["value"]].fp_inps) 

        handles.append(transformer_block.get_submodule(QKV_NAMES["value"]).register_forward_hook(save_inps))
    
    if learn_rounding:
        if rot_emb is None:
            handles.append(transformer_block.get_submodule(QKV_NAMES["query"]).register_forward_hook(functools.partial(wrappers[QKV_NAMES["query"]].compute_cov_out_batch, n_heads=n_heads)))
            handles.append(transformer_block.get_submodule(QKV_NAMES["key"]).register_forward_hook(functools.partial(wrappers[QKV_NAMES["key"]].compute_cov_out_batch, n_heads=n_heads)))
        else:
            hook_outs = {}
            def save_outs(module, inps, outs: Any, name: str):
                nonlocal hook_outs
                hook_outs[name] = outs

            cov_out = {"query": 0, "key": 0}
            n_data_out = 0

            handles.append(transformer_block.self_attn.rot_out_Q.register_forward_hook(functools.partial(save_outs, name="query")))
            handles.append(transformer_block.self_attn.rot_out_K.register_forward_hook(functools.partial(save_outs, name="key")))
    

    for j in range(len(inps)):
        if block_v:
            attn_probs = transformer_block(inps[j].unsqueeze(0), **block_kwargs)[-1]  # shape = [B, H, L, L]
            inps_value = hook_inps.pop().unsqueeze(dim=1)  # shape = [B, 1, L, d]
            inps_value = attn_probs @ inps_value  # shape = [B, H, L, d]
            inps_value = inps_value.transpose(0, 1).view((n_heads, -1, inps_value.shape[-1])).transpose(-1, -2).contiguous()  # shape = [H, d, BL]

            if comp_method == 'GPTAQ':
                fp_inps_value = hook_fp_inps.pop(0).to(inps_value.device).float() # shape = [d, L]
                fp_inps_value = fp_inps_value.t().unsqueeze(0).unsqueeze(1) # [d, L] -> [L, d] -> [1, L, d] -> [1, 1, L, d]
                fp_inps_value = attn_probs @ fp_inps_value # shape = [B, H, L, d]
                fp_inps_value = fp_inps_value.transpose(0, 1).view((n_heads, -1, fp_inps_value.shape[-1])).transpose(-1, -2).contiguous()  # shape = [H, d, BL]
                n_current = fp_inps_value.shape[-1]
                H_value *= n_data / (n_data + n_current)
                dXXT_value *= n_data / (n_data + n_current)
                n_data += n_current
                inps_value = math.sqrt(2 / n_data) * inps_value.float()
                fp_inps_value = math.sqrt(2 / n_data) * fp_inps_value.float()
                H_value += inps_value @ inps_value.transpose(-1, -2)
                dXXT_value += (fp_inps_value - inps_value) @ inps_value.transpose(-1, -2)

                del attn_probs, inps_value, fp_inps_value
                torch.cuda.empty_cache()
            else:
                n_current = inps_value.shape[-1]
                H_value *= n_data / (n_data + n_current)
                n_data += n_current
                inps_value = math.sqrt(2 / n_data) * inps_value.float()
                H_value += inps_value @ inps_value.transpose(-1, -2)
        else:
            transformer_block(inps[j].unsqueeze(0), **block_kwargs)

        if learn_rounding and rot_emb is not None:
            for name in hook_outs:
                outs = hook_outs[name]
                head_dim = outs.shape[-1]
                outs = outs.transpose(0, 1).view(n_heads if name=="query" else n_kv_heads, -1, head_dim).transpose(-1, -2).contiguous()  # [H, d_h, BL]

                n_current = outs.shape[-1]
                cov_out[name] *= n_data_out / (n_data_out + n_current)
                outs = math.sqrt(2 / (n_data_out + n_current)) * outs.float()
                cov_out[name] += outs @ outs.transpose(-1, -2)
            n_data_out += n_current

            hook_outs = {}
            torch.cuda.empty_cache()
    
    for h in handles:
        h.remove()

    wrappers[QKV_NAMES["query"]].H = wrappers[QKV_NAMES["value"]].H
    wrappers[QKV_NAMES["key"]].H = wrappers[QKV_NAMES["value"]].H
    if comp_method == 'GPTAQ':
        wrappers[QKV_NAMES["query"]].dXXT = wrappers[QKV_NAMES["value"]].dXXT
        wrappers[QKV_NAMES["key"]].dXXT = wrappers[QKV_NAMES["value"]].dXXT

    if block_v:
        if n_kv_heads != n_heads:
            H_value = H_value.reshape(n_kv_heads, n_heads // n_kv_heads, H_value.shape[-1], -1).mean(dim=1)
            if comp_method == 'GPTAQ':
                wrappers[QKV_NAMES["value"]].dXXT_per_qhead = dXXT_value
                dXXT_value = dXXT_value.reshape(n_kv_heads, n_heads // n_kv_heads, dXXT_value.shape[-1], -1).mean(dim=1)

        if comp_method == 'GPTAQ':
            wrappers[QKV_NAMES["value"]].dXXT = dXXT_value
            del dXXT_value
            torch.cuda.empty_cache()

        wrappers[QKV_NAMES["value"]].H = H_value
        del H_value
        torch.cuda.empty_cache()

    if learn_rounding:
        if n_kv_heads != n_heads:
            cov_out["query"] = cov_out["query"].reshape(n_kv_heads, n_heads // n_kv_heads, head_dim, head_dim).mean(dim=1)
            cov_out["key"] = cov_out["key"][:, None, :, :].expand(n_kv_heads, n_heads // n_kv_heads, head_dim, head_dim).reshape(n_heads, head_dim, head_dim)
            
        if rot_emb is None:
            wrappers[QKV_NAMES["query"]].cov_G = wrappers[QKV_NAMES["key"]].cov_out
            wrappers[QKV_NAMES["key"]].cov_G = wrappers[QKV_NAMES["query"]].cov_out

            delattr(wrappers[QKV_NAMES["query"]], "cov_out")
            delattr(wrappers[QKV_NAMES["query"]], "n_data_out")        
            delattr(wrappers[QKV_NAMES["key"]], "cov_out")
            delattr(wrappers[QKV_NAMES["key"]], "n_data_out")
        else:
            rot_emb = rot_emb.cuda()
            wrappers[QKV_NAMES["query"]].cov_G = (rot_emb.transpose(-1, -2) @ cov_out["key"] @ rot_emb).mean(0)
            wrappers[QKV_NAMES["key"]].cov_G = (rot_emb.transpose(-1, -2) @ cov_out["query"] @ rot_emb).mean(0)

    torch.cuda.empty_cache()

@torch.no_grad()
def compute_Hessian_llama(transformer_block, wrappers, inps, block_kwargs, n_heads, n_kv_heads, learn_rounding, block_v, rot_emb):
    handles = []
    for name, wrapper in wrappers.items():
        if name not in [QKV_NAMES["query"], QKV_NAMES["key"]]:  # Q, K, V share inputs.
            handles.append(transformer_block.get_submodule(name).register_forward_hook(wrapper.compute_cov_in_batch))
    
    if block_v:
        hook_inps = []
        def save_inps(module, inps, outs: Any):        
            nonlocal hook_inps
            hook_inps.append(inps[0])

        block_kwargs = block_kwargs.copy()
        block_kwargs['output_attentions'] = True  # need to get attention weights
        H_value = 0
        n_data = 0

        handles.append(transformer_block.get_submodule(QKV_NAMES["value"]).register_forward_hook(save_inps))

    if learn_rounding:
        hook_outs = {}
        def save_outs(module, inps, outs: Any, name: str):
            nonlocal hook_outs
            hook_outs[name] = outs

        cov_out = {"query": 0, "key": 0}
        n_data_out = 0

        handles.append(transformer_block.self_attn.rot_out_Q.register_forward_hook(functools.partial(save_outs, name="query")))
        handles.append(transformer_block.self_attn.rot_out_K.register_forward_hook(functools.partial(save_outs, name="key")))
    
    
    for j in range(len(inps)):
        if block_v:
            attn_probs = transformer_block(inps[j].unsqueeze(0), **block_kwargs)[-1]  # shape = [B, H, L, L]
            inps_value = hook_inps.pop().unsqueeze(dim=1)  # shape = [B, 1, L, d]
            inps_value = attn_probs @ inps_value  # shape = [B, H, L, d]
            inps_value = inps_value.transpose(0, 1).view((n_heads, -1, inps_value.shape[-1])).transpose(-1, -2).contiguous()  # shape = [H, d, BL]

            n_current = inps_value.shape[-1]
            H_value *= n_data / (n_data + n_current)
            n_data += n_current
            inps_value = math.sqrt(2 / n_data) * inps_value.float()
            H_value += inps_value @ inps_value.transpose(-1, -2)
        else:
            transformer_block(inps[j].unsqueeze(0), **block_kwargs)

        if learn_rounding:
            for name in hook_outs:
                outs = hook_outs[name]
                head_dim = outs.shape[-1]
                outs = outs.transpose(0, 1).view(n_heads if name=="query" else n_kv_heads, -1, head_dim).transpose(-1, -2).contiguous()  # [H, d_h, BL]

                n_current = outs.shape[-1]
                cov_out[name] *= n_data_out / (n_data_out + n_current)
                outs = math.sqrt(2 / (n_data_out + n_current)) * outs.float()
                cov_out[name] += outs @ outs.transpose(-1, -2)
            n_data_out += n_current
                
    for h in handles:
        h.remove()


    wrappers[QKV_NAMES["query"]].H = wrappers[QKV_NAMES["value"]].H
    wrappers[QKV_NAMES["key"]].H = wrappers[QKV_NAMES["value"]].H

    if block_v:
        if n_kv_heads != n_heads:
            H_value = H_value.reshape(n_kv_heads, n_heads // n_kv_heads, H_value.shape[-1], -1).mean(dim=1)
        wrappers[QKV_NAMES["value"]].H = H_value

    if learn_rounding:
        if n_kv_heads != n_heads:
            cov_out["query"] = cov_out["query"].reshape(n_kv_heads, n_heads // n_kv_heads, head_dim, head_dim).mean(dim=1)
            cov_out["key"] = cov_out["key"][:, None, :, :].expand(n_kv_heads, n_heads // n_kv_heads, head_dim, head_dim).reshape(n_heads, head_dim, head_dim)
        rot_emb = rot_emb.cuda()
        wrappers[QKV_NAMES["query"]].cov_G = (rot_emb.transpose(-1, -2) @ cov_out["key"] @ rot_emb).mean(0)
        wrappers[QKV_NAMES["key"]].cov_G = (rot_emb.transpose(-1, -2) @ cov_out["query"] @ rot_emb).mean(0)
    
    torch.cuda.empty_cache()