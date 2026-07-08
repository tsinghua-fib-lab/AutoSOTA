import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import timm
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model
import torch.nn.functional as F
import numpy as np
import logging
import os
from collections import OrderedDict
import torch
import copy
import random



class Adapter_lora(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.random_orth = True

        self.n_embd = config.d_model if d_model is None else d_model
        # Use ffn_num as attn_bn (they are the same in CL-LoRA)
        self.down_size = getattr(config, 'attn_bn', config.ffn_num) if bottleneck is None else bottleneck

        self.lora_A = nn.Linear(self.down_size, self.n_embd, bias=False)
        self.lora_B = nn.Linear(self.n_embd, self.down_size, bias=False)

        if self.random_orth:
            random_matrix = torch.rand(self.n_embd, self.down_size)
            q, r = torch.linalg.qr(random_matrix)
            with torch.no_grad():
                self.lora_B.weight.copy_(q.T)
            scaling_factor = 1.  # You can adjust this value if needed
            self.lora_B.weight.data *= scaling_factor
        else:
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))

        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.zeros_(self.lora_A.weight)
        else:
            raise NotImplementedError

    def forward(self, x):
        inter_x = self.lora_B(x)
        out = self.lora_A(inter_x)
        return out


class Attention_lora(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., msa = [0,0,0]):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.ffn_option = 'parallel'
        self.msa = msa


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


    def forward(self, x, adapt=None, prompt = None, rank_prompt = None, block_weight = None):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if adapt is not None:
            if block_weight is not None:
                block_weight = block_weight
            else:
                block_weight = torch.ones(3).to(x.device)
            if self.msa[0] == 1:
                adapt_x = adapt[0](x)
                q += block_weight[0] * adapt_x
            if self.msa[1] == 1:
                adapt_x = adapt[1](x)
                k += block_weight[1] * adapt_x
            if self.msa[2] == 1:
                adapt_x = adapt[2](x)
                v += block_weight[2] * adapt_x


        k = self._shape(k, -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(v, -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)


        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x




class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.msa_adapt = True
        self.norm1 = norm_layer(dim)
        self.attn = Attention_lora(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, msa = config.msa)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)



    # prompt and rank_prmopt can be considerred as potential future improvements by levergaing additional prompt information, but is not implemented in this work
    def forward(self, x, adapt=None, prompt=None, rank_prompt=None, block_weight=None):
        if self.msa_adapt:
            x = x + self.drop_path(
                self.attn(self.norm1(x), adapt, prompt, rank_prompt, block_weight))
            residual = x
            x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
            x = self.drop_path(self.mlp_drop(self.fc2(x)))
            x = residual + x
        return x



class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
        super().__init__()

        self.tuning_config = tuning_config
        if self.tuning_config.ffn_adapt:
            print("I'm using ViT with adapters.")
        else:
            print("I'm using ViT without adapters.")
            self.maskout_block = []
        self.adapt_msa = True
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.msa_adapt = self.tuning_config.msa_adapt
        self.use_distillation = self.tuning_config.use_distillation
        self.use_block_weight = self.tuning_config.use_block_weight

        if self.msa_adapt:
            self.msa = self.tuning_config.msa
        self.general_pos = self.tuning_config.general_pos
        self.specfic_pos = self.tuning_config.specfic_pos

        self.adapt_pos = self.general_pos+ self.specfic_pos
        self.adapt_pos = sorted(self.adapt_pos)


        if self.use_distillation:
            self.old_adapter_list = nn.ModuleList()

        if self.use_block_weight:
            self.block_weight_list = []
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
            nn.init.uniform_(self.block_weight, .5, 1.5)


        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        ######### MAE begins ############
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        ######## Adapter begins #########
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            # properly registered
            self.embeddings = nn.ParameterList(  # batch, num_prompt, embed_dim
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _
                 in range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        self.config = tuning_config
        self._device = tuning_config._device
        self.adapter_list = []
        self.adapter_pos_list = []
        self.cur_adapter = nn.ModuleList()
        if self.msa_adapt:
            self.get_new_adapter_initial_msa()

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist           

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        
        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad = True


    def get_new_adapter_initial_msa(self):
        config = self.config
        if config.ffn_adapt:
            for i in range(len(self.adapt_pos)):
                temp_adapter = nn.ModuleList()
                for j in self.msa:
                    if j ==1:
                        adapter = Adapter_lora(self.config, dropout=0.0, bottleneck=config.ffn_num,
                                                init_option=config.ffn_adapter_init_option,
                                                adapter_scalar=config.ffn_adapter_scalar,
                                                adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                                ).to(self._device)
                    else:
                        adapter = nn.Identity()
                    temp_adapter.append(adapter)

                self.cur_adapter.append(temp_adapter)
            self.cur_adapter.requires_grad_(True)

        else:
            print("====Not use adapter===")

    def get_new_adapter_msa(self):
        config = self.config

        if config.ffn_adapt:
            for i in range(len(self.specfic_pos)):
                pos = self.adapt_pos.index(self.specfic_pos[i])
                temp_adapter = nn.ModuleList()
                for j in self.msa:
                    if j == 1:
                        adapter = Adapter_lora(self.config, dropout=0.0, bottleneck=config.ffn_num,
                                               init_option=config.ffn_adapter_init_option,
                                               adapter_scalar=config.ffn_adapter_scalar,
                                               adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                               ).to(self._device)
                        adapter.requires_grad_(True)
                    else:
                        adapter = nn.Identity()
                    temp_adapter.append(adapter)
                self.cur_adapter[pos] = temp_adapter

            if len(self.specfic_pos) < 12:
                self.cur_adapter.requires_grad_(True)

                for i in self.adapt_pos:
                    if i in self.general_pos:
                        pos = self.adapt_pos.index(i)
                        for j in range(len(self.msa)):
                            if self.msa[j] == 1:
                                self.cur_adapter[pos][j].lora_B.requires_grad_(False)
        else:
            print("====Not use adapter===")


    def add_adapter_to_list(self):
        temp_adapter = []
        for i in range(len(self.specfic_pos)):
            temp_pos = self.adapt_pos.index(self.specfic_pos[i])
            temp_adapter.append(copy.deepcopy(self.cur_adapter[temp_pos].requires_grad_(False)))
        self.adapter_list.append(temp_adapter)

        if self.use_block_weight:
            self.block_weight_old = copy.deepcopy(self.block_weight)
            self.block_weight_list.append(self.block_weight_old.requires_grad_(False))
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
            nn.init.uniform_(self.block_weight, .5, 1.5)


        self.adapter_pos_list.append(self.adapt_pos)

        if self.use_distillation:
            self.old_adapter_list.append(copy.deepcopy(self.cur_adapter).requires_grad_(False))
        if self.msa_adapt:
            self.get_new_adapter_msa()

    def forward_train(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            rank_prompt = None
            prompt = None

            if self.config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)

            if self.config.ffn_adapt:
                if idx in self.adapt_pos:
                    pos = self.adapt_pos.index(idx)
                    block_weight = None
                    if self.use_block_weight and idx in self.specfic_pos:
                        pos_spec = self.specfic_pos.index(idx)
                        x = blk(x, self.cur_adapter[pos], prompt, rank_prompt,
                                block_weight=self.block_weight[:, pos_spec])
                    else:
                        x = blk(x, self.cur_adapter[pos], prompt, rank_prompt, block_weight=None)
                else:
                    x = blk(x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
            else:
                x = blk(x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
            if self.config.vpt_on:
                x = x[:, self.config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_test(self, x, use_init_ptm=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)

        features = []

        if use_init_ptm:
            x = copy.deepcopy(x_init)
            x = self.blocks(x)
            x = self.norm(x)
            features.append(x)
        if self.config.ffn_adapt:
            for i in range(len(self.adapter_list)):
                x = copy.deepcopy(x_init)
                for j in range(len(self.blocks)):

                    rank_prompt = None
                    prompt = None

                    if j in self.adapt_pos:
                        if j in self.general_pos:
                            pos = self.adapt_pos.index(j)
                            adapt = self.cur_adapter[pos]
                        else:
                            pos = self.specfic_pos.index(j)
                            adapt = self.adapter_list[i][pos]

                        if self.use_block_weight and j in self.specfic_pos:
                            pos_spec = self.specfic_pos.index(j)
                            block_weight = self.block_weight_list[i][:, pos_spec]
                        else:
                            block_weight = None
                        x = self.blocks[j](x, adapt, prompt, rank_prompt, block_weight)

                    else:
                        x = self.blocks[j](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)

                x = self.norm(x)
                features.append(x)

            x = copy.deepcopy(x_init)
            for i in range(len(self.blocks)):

                rank_prompt = None
                prompt = None

                if i in self.adapt_pos:
                    pos = self.adapt_pos.index(i)
                    adapt = self.cur_adapter[pos]
                    if self.use_block_weight and i in self.specfic_pos:
                        pos_spec = self.specfic_pos.index(i)
                        block_weight = self.block_weight[:, pos_spec]
                    else:
                        block_weight = None
                    x = self.blocks[i](x, adapt, prompt, rank_prompt, block_weight)
                else:
                    x = self.blocks[i](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
            x = self.norm(x)
            features.append(x)

        return features

    def forward(self, x, test=False, use_init_ptm=False):
        if not test:
            output = self.forward_train(x)
            return output

        else:
            features = self.forward_test(x, use_init_ptm)
            output = torch.Tensor().to(features[0].device)
            for x in features:
                cls = x[:, 0, :]
                output = torch.cat((
                    output,
                    cls
                ), dim=1)
            return output

    def forward_proto(self, x, adapt_index):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)

        # the init_PTM's feature
        if adapt_index == -1:
            x = copy.deepcopy(x_init)
            x = self.blocks(x)
            x = self.norm(x)
            output = x[:, 0, :]
            return output

        i = adapt_index
        x = copy.deepcopy(x_init)
        if self.config.ffn_adapt:
            if i < len(self.adapter_list):
                for j in range(len(self.blocks)):

                    rank_prompt = None
                    prompt = None

                    if j in self.adapt_pos:
                        if j in self.general_pos:
                            pos = self.adapt_pos.index(j)
                            adapt = self.cur_adapter[pos]
                        else:
                            pos = self.specfic_pos.index(j)
                            adapt = self.adapter_list[i][pos]
                        if self.use_block_weight and j in self.specfic_pos:
                            pos_spec = self.specfic_pos.index(j)
                            block_weight = self.block_weight_list[i][:, pos_spec]
                        else:
                            block_weight = None
                        x = self.blocks[j](x, adapt, prompt, rank_prompt, block_weight)

                    else:
                        x = self.blocks[j](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
            else:
                for j in range(len(self.blocks)):
                    rank_prompt = None
                    prompt = None

                    if j in self.adapt_pos:
                        pos = self.adapt_pos.index(j)
                        adapt = self.cur_adapter[pos]
                        if self.use_block_weight and j in self.specfic_pos:
                            pos_spec = self.specfic_pos.index(j)
                            block_weight = self.block_weight[:, pos_spec]
                        else:
                            block_weight = None

                        x = self.blocks[j](x, adapt, prompt, rank_prompt, block_weight)
                    else:
                        x = self.blocks[j](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
        else:
            for j in range(len(self.blocks)):
                rank_prompt = None
                prompt = None

                x = self.blocks[j](x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)

        x = self.norm(x)
        output = x[:, 0, :]

        return output

    def forward_general_cls(self, x, t_idx):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x_teacher = copy.deepcopy(x)

        for j in self.general_pos:
            pos = self.adapt_pos.index(j)
            adapt = self.cur_adapter[pos]
            x = self.blocks[j](x, adapt)

        x = self.norm(x)
        output_new = x[:, 0, :]



        for j in self.general_pos:
            pos = self.adapt_pos.index(j)
            adapt = self.old_adapter_list[t_idx-1][pos]
            x_teacher = self.blocks[j](x_teacher, adapt)
        x_teacher = self.norm(x_teacher)
        output_teacher= x_teacher[:, 0, :]

        return output_new, output_teacher




def vit_base_patch16_224_cllora(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    checkpoint_model=timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False

    if not model.msa_adapt:
        for adapter_temp in model.cur_adapter:
            #for adapter in adapter_temp:
            for param in adapter_temp.lora_B.parameters():
                param.requires_grad = False
    else:
        for i in model.adapt_pos:
            #if i in model.general_pos:
            if i in model.general_pos:
                pos = model.adapt_pos.index(i)
                for j in range(len(model.msa)):
                    if model.msa[j] == 1:
                    #for adapter in adapter_temp:
                        for param in model.cur_adapter[pos][j].lora_B.parameters():
                            param.requires_grad = False
    #
    return model

def vit_base_patch16_224_in21k_cllora(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    checkpoint_model=timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False


    if not model.msa_adapt:
        for adapter_temp in model.cur_adapter:
            #for adapter in adapter_temp:
            for param in adapter_temp.lora_B.parameters():
                param.requires_grad = False
    else:
        for i in model.adapt_pos:
            #if i in model.general_pos:
            if i in model.general_pos:
                pos = model.adapt_pos.index(i)
                for j in range(len(model.msa)):
                    if model.msa[j] == 1:
                    #for adapter in adapter_temp:
                        for param in model.cur_adapter[pos][j].lora_B.parameters():
                            param.requires_grad = False

    return model

