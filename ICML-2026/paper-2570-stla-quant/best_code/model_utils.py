import torch

def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip

def get_opt(model_path):
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    
    return model

def get_llama(model_path):
    from modeling_llama_custom import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(model_path, attn_implementation="eager", torch_dtype='auto')
    
    return model

def get_bloom(model_path):
    from modeling_bloom_custom import BloomForCausalLM

    model = BloomForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    model = convert_bloom(model)
    
    return model

def convert_bloom(model):
    from modeling_bloom_custom import BloomAttention, BloomBlock
    
    for module in model.modules():
        if isinstance(module, BloomAttention):
            module.custom_split_qkv()
            module.custom_renaming()

    for module in model.modules():
        if isinstance(module, BloomBlock):
            module.custom_renaming()

    return model
    
def get_model(model_path):
    if 'llama' in model_path:
        return get_llama(model_path)
    elif 'opt' in model_path:
        return get_opt(model_path)
    elif 'bloom' in model_path:
        return get_bloom(model_path)
    else:
        raise NotImplemented(f"Not support {model_path.split('/')[-1]} model.")
    

def get_transformer_blocks(llm, return_model_type=True):
    if hasattr(llm, 'transformer'):
        model_type = 'bloom'
        transformer_blocks = llm.transformer.h

    elif hasattr(llm.model, "decoder"):
        model_type = 'opt'
        transformer_blocks = llm.model.decoder.layers

    else:
        model_type = "llama"
        transformer_blocks = llm.model.layers

    if return_model_type:
        return transformer_blocks, model_type
    else:
        return transformer_blocks
    

def set_zfold_layers(model_type, llm_config):
    if model_type == "llama":
        zfold_list = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.down_proj']
        zeta_share_lists = {
            "QKV": ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
        }
    
        if llm_config.num_key_value_heads == llm_config.num_attention_heads:
            zfold_list.append("self_attn.o_proj")

    elif model_type == "bloom":
        zfold_list = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.dense_h_to_4h']   
        zeta_share_lists = {
            "QKV": ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
        }

    elif model_type == "opt":
        # TODO: OPT-350M is not supported
        # if llm.model.decoder.project_in is not None:  # Post LayerNorm architecture is used in OPT-350M
        #     zfold_list = ['self_attn.out_proj', 'fc2']
        #     zeta_share_lists = {}
        # else:
        zfold_list = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj', 'fc1', 'fc2']
        zeta_share_lists = {
            "QKV": ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
        }

    return zfold_list, zeta_share_lists