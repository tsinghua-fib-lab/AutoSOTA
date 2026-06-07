import copy
import torch
from types import MethodType
import torch.nn as nn
import torch.nn.functional as F

from mmengine.config import Config

from projects.vlm.tokenmask.models import QWEN25VL_VQSAM2Model

def cast_module_dtype_(module: torch.nn.Module, dtype: torch.dtype):
    for name, p in module.named_parameters(recurse=True):
        if p.dtype.is_floating_point:
            p.data = p.data.to(dtype)
    for name, b in module.named_buffers(recurse=True):
        if b.dtype.is_floating_point:
            b.data = b.data.to(dtype)
    return module

if __name__ == "__main__":
    save_path = "./work_dirs/ablation_qwen25vl_3b_res_1024x2/qwen25vl_3b_ft"
    cfg = Config.fromfile('projects/vlm/tokenmask/configs/ablation/qwen25vl_3b_mt_1024x2_res.py')
   
    model = QWEN25VL_VQSAM2Model(
        qwen25vl_hf_model=cfg.model.qwen25vl_hf_model,
        tokenizer=cfg.model.tokenizer,
        preprocessor=cfg.model.preprocessor,
        llm_lora=cfg.model.llm_lora,
        pretrained_pth=cfg.model.pretrained_pth,
        unfreeze_vision_encoder=cfg.model.unfreeze_vision_encoder,
    )

    pth_path = "./work_dirs/ablation_qwen25vl_3b_res_1024x2/iter_52752.pth"
    state = torch.load(pth_path, map_location="cpu", weights_only=False)
    model_sd = (state.get("state_dict")
                or state.get("model")
                or state.get("module")
                or state)
    if any(k.startswith("module.") for k in model_sd.keys()):
        model_sd = {k.replace("module.", "", 1): v for k, v in model_sd.items()}

    target_dtype = torch.bfloat16
    cast_module_dtype_(model, target_dtype)
   
    model.load_state_dict(model_sd, strict=False)
    print(f'Load PTH model from {pth_path}')

    cast_module_dtype_(model, target_dtype)

    model._merge_lora()
    cast_module_dtype_(model, target_dtype)
    model.qwen25vl_model.tie_weights()
    cast_module_dtype_(model.qwen25vl_model, target_dtype)

    model.qwen25vl_model.save_pretrained(save_path)
    model.tokenizer.save_pretrained(save_path)
    model.preprocessor.save_pretrained(save_path)
    print("Done!")

