import copy
import torch
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print("use npu success!")
except:
    print("npu not enabled!")
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModel
from mmengine.config import Config, ConfigDict
from xtuner.registry import BUILDER
from xtuner.model.utils import find_all_linear_names
from peft import get_peft_model, prepare_model_for_kbit_training
# from xtuner.model.utils import guess_load_checkpoint

import os.path as osp
from typing import List, Optional

def prepare_inputs_for_generation_cache(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen3VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs


def guess_load_checkpoint(pth_model):
    if osp.isfile(pth_model):
        state_dict = torch.load(pth_model, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    elif osp.isdir(pth_model):
        try:
            from xtuner.utils.zero_to_any_dtype import (
                get_state_dict_from_zero_checkpoint,
            )
        except ImportError:
            raise ImportError(
                "The provided PTH model appears to be a DeepSpeed checkpoint. "
                "However, DeepSpeed library is not detected in current "
                "environment. This suggests that DeepSpeed may not be "
                "installed or is incorrectly configured. Please verify your "
                "setup."
            )
        state_dict = get_state_dict_from_zero_checkpoint(
            osp.dirname(pth_model), osp.basename(pth_model)
        )
    else:
        raise FileNotFoundError(f"Cannot find {pth_model}")
    return state_dict

class QWEN3VL_VQSAM2Model(BaseModel):
    def __init__(
        self,
        qwen3vl_hf_model,
        tokenizer=None,
        preprocessor=None,
        llm_lora=None,
        pretrained_pth=None,
        unfreeze_vision_encoder=False,
    ):
        super(QWEN3VL_VQSAM2Model, self).__init__()
        
        #==================QWEN25VL===================
        self.qwen3vl_model = BUILDER.build(qwen3vl_hf_model)
        # if self.training:
        #     self.qwen3vl_model.model.language_model.forward = MethodType(Qwen2_5_VLTextModel_forward, self.qwen25vl_model.model.language_model)
        self.tokenizer = BUILDER.build(tokenizer)
        self.preprocessor = BUILDER.build(preprocessor)
        self.mt_start_token_id = self.tokenizer.convert_tokens_to_ids('<|mt_start|>')
        self.mt_end_token_id = self.tokenizer.convert_tokens_to_ids('<|mt_end|>')

        #==================freeze parameters================
        if llm_lora is not None:
            self.qwen3vl_model.model.requires_grad_(False)
        if unfreeze_vision_encoder:
            self.qwen3vl_model.model.visual.requires_grad_(True)
        else:
            self.qwen3vl_model.model.visual.requires_grad_(False)

        if hasattr(self.qwen3vl_model, "enable_input_require_grads"):
            self.qwen3vl_model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.qwen3vl_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        self.gradient_checkpointing_enable()

        #================support lora====================
        if llm_lora is not None:
            self.qwen3vl_model.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_cache, self.qwen3vl_model.language_model)
            self._prepare_llm_for_lora(llm_lora)
            self.qwen3vl_model.get_input_embeddings().requires_grad_(True)
            self.qwen3vl_model.get_output_embeddings().weight.requires_grad_(True)
            self.qwen3vl_model.tie_weights()

        #==================load weights===============        
        if pretrained_pth is not None:
            state = torch.load(pretrained_pth, map_location="cpu", weights_only=False)
            model_sd = (state.get("state_dict")
                        or state.get("model")
                        or state.get("module")
                        or state)
            if any(k.startswith("module.") for k in model_sd.keys()):
                model_sd = {k.replace("module.", "", 1): v for k, v in model_sd.items()}
            self.load_state_dict(model_sd, strict=False)
            print(f"Loaded pretrained weights from {pretrained_pth}")

        self.unfreeze_vision_encoder = unfreeze_vision_encoder


    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.qwen3vl_model.language_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.qwen3vl_model.language_model.gradient_checkpointing_disable()

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
            lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.qwen3vl_model.model.language_model = prepare_model_for_kbit_training(self.qwen3vl_model.model.language_model, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.qwen3vl_model.model.language_model)
            lora_config.target_modules = modules
        
        self.qwen3vl_model.model.language_model = get_peft_model(self.qwen3vl_model.model.language_model, lora_config)
    
    def _merge_lora(self):
        try:
            self.qwen3vl_model.model.language_model = self.qwen3vl_model.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        return
    
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        from collections import OrderedDict

        to_return = OrderedDict()

        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'language_model.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'llm_to_sam.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'lm_head.' in k})
        
        if self.unfreeze_vision_encoder:
            to_return.update(
                {k: v
                for k, v in state_dict.items() if 'qwen25vl_model.model.visual.' in k})
        
        # for k, v in to_return.items():
        #     print(k)
        # exit(0)

        return to_return
    
        
    def init_weights(self):
        pass

    def forward(self, data, data_samples=None, mode='loss'):
        # for n, p in self.named_parameters():
        #     if p.requires_grad:
        #         print(n)
        # exit(0)

        sam2_pixel_values = data.pop('sam2_pixel_values', None) #[2, 3, 1024, 1024]
        gt_masks = data.pop('masks', None) #list each has shape like this [1, 480, 640]
        codebook_embeds = data.pop('codebook_embeds', None)

        qwen3vl_output = self.qwen3vl_model(**data, use_cache=False, output_hidden_states=True)

        loss_dict = {
            'llm_loss': qwen3vl_output.loss,
        }

        return loss_dict
                
