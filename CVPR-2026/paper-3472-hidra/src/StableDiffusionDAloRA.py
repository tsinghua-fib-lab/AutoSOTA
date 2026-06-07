import os
import re
import requests
import sys
import copy
import numpy as np
import torchvision
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import init as init

p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_lora_fwd, LayerNorm
# from basicsr.archs.arch_util import default_init_weights

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def get_layer_number(module_name):
    base_layers = {
        'down_blocks': 0,
        'mid_block': 4,
        'up_blocks': 5
    }

    if module_name == 'conv_out':
        return 9

    base_layer = None
    for key in base_layers:
        if key in module_name:
            base_layer = base_layers[key]
            break

    if base_layer is None:
        return None

    additional_layers = int(
        re.findall(r'\.(\d+)', module_name)[0])  # sum(int(num) for num in re.findall(r'\d+', module_name))
    final_layer = base_layer + additional_layers
    return final_layer


class StableDiffusionDAloRA(torch.nn.Module):
    def __init__(self, sd_path=None, pretrained_path=None, lora_rank_unet=32, lora_rank_vae=16, block_embedding_dim=64, num_embeddings=256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(sd_path)

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]

        self.vae_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings, 256),
            nn.ReLU(True),
        )

        self.unet_de_mlp = nn.Sequential(
            nn.Linear(num_embeddings, 256),
            nn.ReLU(True),
        )

        self.vae_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.unet_block_mlp = nn.Sequential(
            nn.Linear(block_embedding_dim, 64),
            nn.ReLU(True),
        )

        self.vae_fuse_mlp = nn.Linear(256 + 64, lora_rank_vae ** 2)
        self.unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet ** 2)

        default_init_weights([self.vae_de_mlp, self.unet_de_mlp, self.vae_block_mlp, self.unet_block_mlp,
                              self.vae_fuse_mlp, self.unet_fuse_mlp], 1e-5)

        # vae
        self.vae_block_embeddings = nn.Embedding(6, block_embedding_dim)
        self.unet_block_embeddings = nn.Embedding(10, block_embedding_dim)

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian",
                                         target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)

            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                                          target_modules=sd["unet_lora_target_modules"])
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

            _vae_de_mlp = self.vae_de_mlp.state_dict()
            for k in sd["state_dict_vae_de_mlp"]:
                _vae_de_mlp[k] = sd["state_dict_vae_de_mlp"][k]
            self.vae_de_mlp.load_state_dict(_vae_de_mlp)

            _unet_de_mlp = self.unet_de_mlp.state_dict()
            for k in sd["state_dict_unet_de_mlp"]:
                _unet_de_mlp[k] = sd["state_dict_unet_de_mlp"][k]
            self.unet_de_mlp.load_state_dict(_unet_de_mlp)

            _vae_block_mlp = self.vae_block_mlp.state_dict()
            for k in sd["state_dict_vae_block_mlp"]:
                _vae_block_mlp[k] = sd["state_dict_vae_block_mlp"][k]
            self.vae_block_mlp.load_state_dict(_vae_block_mlp)

            _unet_block_mlp = self.unet_block_mlp.state_dict()
            for k in sd["state_dict_unet_block_mlp"]:
                _unet_block_mlp[k] = sd["state_dict_unet_block_mlp"][k]
            self.unet_block_mlp.load_state_dict(_unet_block_mlp)

            _vae_fuse_mlp = self.vae_fuse_mlp.state_dict()
            for k in sd["state_dict_vae_fuse_mlp"]:
                _vae_fuse_mlp[k] = sd["state_dict_vae_fuse_mlp"][k]
            self.vae_fuse_mlp.load_state_dict(_vae_fuse_mlp)

            _unet_fuse_mlp = self.unet_fuse_mlp.state_dict()
            for k in sd["state_dict_unet_fuse_mlp"]:
                _unet_fuse_mlp[k] = sd["state_dict_unet_fuse_mlp"][k]
            self.unet_fuse_mlp.load_state_dict(_unet_fuse_mlp)

            embeddings_state_dict = sd["state_embeddings"]
            self.vae_block_embeddings.load_state_dict(embeddings_state_dict['state_dict_vae_block'])
            self.unet_block_embeddings.load_state_dict(embeddings_state_dict['state_dict_unet_block'])
        else:
            print("Initializing model with random weights")
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                                         target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                                          target_modules=target_modules_unet
                                          )
            unet.add_adapter(unet_lora_config)

        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

        self.vae_lora_layers = []
        for name, module in vae.named_modules():
            if 'base_layer' in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])

        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])

        for name, module in unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_layer_dict = {name: get_layer_number(name) for name in self.unet_lora_layers}

        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.vae_de_mlp.eval()
        self.unet_de_mlp.eval()
        self.vae_block_mlp.eval()
        self.unet_block_mlp.eval()
        self.vae_fuse_mlp.eval()
        self.unet_fuse_mlp.eval()

        self.vae_block_embeddings.requires_grad_(False)
        self.unet_block_embeddings.requires_grad_(False)

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.vae_de_mlp.train()
        self.unet_de_mlp.train()
        self.vae_block_mlp.train()
        self.unet_block_mlp.train()
        self.vae_fuse_mlp.train()
        self.unet_fuse_mlp.train()

        self.vae_block_embeddings.requires_grad_(True)
        self.unet_block_embeddings.requires_grad_(True)

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

        self.unet.conv_in.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def forward(self, c_t, img_prompt, txt_prompt):

        caption_tokens = self.tokenizer(txt_prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
        caption_enc = self.text_encoder(caption_tokens)[0]

        # degradation mlp forward
        vae_de_c_embed = self.vae_de_mlp(img_prompt)
        unet_de_c_embed = self.unet_de_mlp(img_prompt)

        # block embedding mlp forward
        vae_block_c_embeds = self.vae_block_mlp(self.vae_block_embeddings.weight)
        unet_block_c_embeds = self.unet_block_mlp(self.unet_block_embeddings.weight)
        vae_embeds = self.vae_fuse_mlp(torch.cat([vae_de_c_embed.unsqueeze(1).repeat(1, vae_block_c_embeds.shape[0], 1), \
                                                  vae_block_c_embeds.unsqueeze(0).repeat(vae_de_c_embed.shape[0], 1,
                                                                                         1)], -1))
        unet_embeds = self.unet_fuse_mlp(
            torch.cat([unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1), \
                       unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0], 1, 1)], -1))

        for layer_name, module in self.vae.named_modules():
            if layer_name in self.vae_lora_layers:
                split_name = layer_name.split(".")
                if split_name[1] == 'down_blocks':
                    block_id = int(split_name[2])
                    vae_embed = vae_embeds[:, block_id]
                elif split_name[1] == 'mid_block':
                    vae_embed = vae_embeds[:, -2]
                else:
                    vae_embed = vae_embeds[:, -1]
                module.de_mod = vae_embed.reshape(-1, self.lora_rank_vae, self.lora_rank_vae)

        for layer_name, module in self.unet.named_modules():
            if layer_name in self.unet_lora_layers:
                split_name = layer_name.split(".")

                if split_name[0] == 'down_blocks':
                    block_id = int(split_name[1])
                    unet_embed = unet_embeds[:, block_id]
                elif split_name[0] == 'mid_block':
                    unet_embed = unet_embeds[:, 4]
                elif split_name[0] == 'up_blocks':
                    block_id = int(split_name[1]) + 5
                    unet_embed = unet_embeds[:, block_id]
                else:
                    unet_embed = unet_embeds[:, -1]
                module.de_mod = unet_embed.reshape(-1, self.lora_rank_unet, self.lora_rank_unet)

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc, ).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip_conv" in k}
        sd["state_dict_vae_de_mlp"] = {k: v for k, v in self.vae_de_mlp.state_dict().items()}
        sd["state_dict_unet_de_mlp"] = {k: v for k, v in self.unet_de_mlp.state_dict().items()}
        sd["state_dict_vae_block_mlp"] = {k: v for k, v in self.vae_block_mlp.state_dict().items()}
        sd["state_dict_unet_block_mlp"] = {k: v for k, v in self.unet_block_mlp.state_dict().items()}
        sd["state_dict_vae_fuse_mlp"] = {k: v for k, v in self.vae_fuse_mlp.state_dict().items()}
        sd["state_dict_unet_fuse_mlp"] = {k: v for k, v in self.unet_fuse_mlp.state_dict().items()}

        sd["state_embeddings"] = {
            "state_dict_vae_block": self.vae_block_embeddings.state_dict(),
            "state_dict_unet_block": self.unet_block_embeddings.state_dict(),
        }

        torch.save(sd, outf)
