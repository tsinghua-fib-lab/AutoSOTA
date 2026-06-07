import os
import copy
from collections import OrderedDict
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine import print_log
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig

import torch.utils.checkpoint
from xtuner.registry import BUILDER
from xtuner.model.modules import dispatch_modules
from xtuner.model.utils import (traverse_dict, make_inputs_require_grad, find_all_linear_names,
                    guess_load_checkpoint, get_peft_model_state_dict)

from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText


class PerceptionLM_TokenMask(BaseModel):
    def __init__(self, 
                 mllm,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 freeze_connector=False,
                 unfreeze_vocab=False,
                 unfreeze_lm_head=False,
                 llm_lora=False,
                 visual_encoder_lora=False,
                 pretrained_pth=None,
                 use_activation_checkpointing=True,
                 ):
        super().__init__()
        
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_connector = freeze_connector
        self.unfreeze_vocab = unfreeze_vocab
        self.unfreeze_lm_head = unfreeze_lm_head
        self.use_llm_lora = llm_lora
        self.use_visual_encoder_lora = visual_encoder_lora
        self.use_activation_checkpointing=use_activation_checkpointing

        config = AutoConfig.from_pretrained(mllm["pretrained_model_name_or_path"], trust_remote_code=True)

        self.config = config

        traverse_dict(mllm)

        self.model = AutoModelForImageTextToText.from_pretrained(mllm["pretrained_model_name_or_path"], trust_remote_code=True)
        self.model.model.config.use_cache = False

        dispatch_modules(self.model.model)

        self.processor = AutoProcessor.from_pretrained(mllm["pretrained_model_name_or_path"], trust_remote_code=True)

        if self.freeze_llm:
            self.model.model.language_model.requires_grad_(False)

        if self.freeze_visual_encoder:
            self.model.model.vision_tower.requires_grad_(False)
        
        if self.freeze_connector:
            self.model.model.multi_modal_projector.requires_grad_(False)

        if use_activation_checkpointing:
            # it is necessary when using gradient checkpointing
            if hasattr(self.model.model, 'enable_input_require_grads'):
                self.model.model.enable_input_require_grads()
            else:
                self.model.model.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)

        self.gradient_checkpointing_enable()
        
        if self.use_llm_lora:
            lora_config = LoraConfig(
                r=64,
                lora_alpha=128,
                lora_dropout=0.05,
                target_modules=[
                    "down_proj",
                    "o_proj",
                    "k_proj",
                    "q_proj",
                    "gate_proj",
                    "up_proj",
                    "v_proj",
                ],
                use_dora=True,
                init_lora_weights="gaussian",
            )
            lora_config.inference_mode = False
            self.model.add_adapter(lora_config)
            self.model.enable_adapters()
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, lora_config)

        # put this after llm_lora
        if self.unfreeze_vocab:
            self.model.get_input_embeddings().requires_grad_(True)
        if self.unfreeze_lm_head:
            self.model.get_output_embeddings().requires_grad_(True)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            mllm_state_dict = {}
            for k, v in pretrained_state_dict.items():
                if k.startswith('model.'):
                    mllm_state_dict[k[len('model.'):]] = v
            if len(mllm_state_dict) != 0:
                self.model.model.load_state_dict(mllm_state_dict, strict=False)
            
            print(f"Load pretrained weight from {pretrained_pth}")
        
        self._count = 0
        print_log(self, logger="current")
        print_log('Perception_LM construction is complete', logger='current')
  
    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.model.model = prepare_model_for_kbit_training(self.model.model, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.model)
            lora_config.target_modules = modules

        self.model.model = get_peft_model(self.model.model, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.model.model.gradient_checkpointing_enable()


    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.model.model.gradient_checkpointing_disable()
  

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()

        to_return.update(
            {k: v for k, v in state_dict.items() if 'tok_embeddings' in k or 'embed' in k or 'embed_tokens' in k}
        )
        # logit head
        to_return.update(
            {k: v for k, v in state_dict.items() if 'output.' in k and 'llm' in k and 'lora' not in k}
        )
        to_return.update(
            {k: v for k, v in state_dict.items() if
             'lm_head' in k and 'lora' not in k}
        )
        to_return.update(
            {k: v for k, v in state_dict.items() if
             'output' in k and 'lora' not in k}
        )

        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model.visual, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'model.visual.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.model, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'language_model' in k
            })

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

        # input_ids torch.Size([4, 3642])
        # attention_mask torch.Size([4, 3642])
        # position_ids torch.Size([4, 3642])
        # labels torch.Size([4, 3642])
        # pixel_values torch.Size([52, 3, 448, 448])

        plm_output = self.model(**data, use_cache=False)
        return {'llm_loss': plm_output.loss}