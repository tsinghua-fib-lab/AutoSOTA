from pathlib import Path

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

import modules.shared as shared
from modules.model import get_max_memory_dict

def load_quantized(model_name):
    path_to_model = Path(f'{shared.model_dir}/{model_name}')
    pt_path = None

    # Find the model checkpoint
    
    for ext in ['.safetensors', '.pt', '.bin']:
        found = list(path_to_model.glob(f"*{ext}"))
        if len(found) > 0:
            if len(found) > 1:
                print(f'More than one {ext} model has been found. The last one will be selected. It could be wrong.')

            pt_path = found[-1]
            break

    if pt_path is None:
        print("The model could not be loaded because its checkpoint file in .bin/.pt/.safetensors format could not be located.")
        return

    use_safetensors = pt_path.suffix == '.safetensors'
    if not (path_to_model / "quantize_config.json").exists():
        quantize_config = BaseQuantizeConfig(
            bits=bits if (bits := shared.wbits) > 0 else 4,
            group_size=gs if (gs := shared.groupsize) > 0 else -1,
            desc_act = shared.act_order#True
        )
    else:
        quantize_config = None

    # Define the params for AutoGPTQForCausalLM.from_quantized
    params = {
        'model_basename': pt_path.stem,
        'device': "cuda:0",
        'use_triton': False,
        'inject_fused_attention': False,
        'inject_fused_mlp': False,
        'use_safetensors': use_safetensors,
        'trust_remote_code': True,
        'max_memory': get_max_memory_dict(),
        'quantize_config': quantize_config
        #'disable_exllama': True
        #'use_marlin': False
    }

    print(f"The AutoGPTQ params are: {params}")
    model = AutoGPTQForCausalLM.from_quantized(path_to_model, **params)

    # These lines fix the multimodal extension when used with AutoGPTQ
    #if hasattr(model, 'model'):
    #    if not hasattr(model, 'dtype'):
    #        if hasattr(model.model, 'dtype'):
    #            model.dtype = model.model.dtype

    #    if hasattr(model.model, 'model') and hasattr(model.model.model, 'embed_tokens'):
    #        if not hasattr(model, 'embed_tokens'):
    #            model.embed_tokens = model.model.model.embed_tokens

    #        if not hasattr(model.model, 'embed_tokens'):
    #            model.model.embed_tokens = model.model.model.embed_tokens

    return model
