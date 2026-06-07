import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, Mlp
from timm.models.helpers import checkpoint_seq
import math
from functools import reduce
from operator import mul
import numpy as np

class PromptViT(nn.Module):
    '''
    Vision Transformer with added prompts at the input layer
    '''
    def __init__(self,
                vit:VisionTransformer,
                num_prompts = 1,
                prompt_init = None):
        super().__init__()
        self.vit = vit
        self.num_prompts = num_prompts
        self.prompt_dim = vit.embed_dim

        if num_prompts > 0:
            self.prompts = nn.Parameter(torch.zeros(1, num_prompts, self.prompt_dim))
            if prompt_init is not None:
                # IDEA-013: Use provided initialization (e.g., mean patch embedding)
                with torch.no_grad():
                    self.prompts.data.copy_(prompt_init)
            else:
                # initialization adopted from vpt, https://arxiv.org/abs/2203.12119
                val = math.sqrt(6. / float(3 * reduce(mul, vit.patch_embed.patch_size, 1) + self.prompt_dim)) # noqa
                nn.init.uniform_(self.prompts.data, -val, val) # xavier_uniform initialization
        self._orig_prompt_init = self.prompts.data.clone() if num_prompts > 0 else None
    
    def reset(self):
        if self._orig_prompt_init is not None:
            with torch.no_grad():
                self.prompts.data.copy_(self._orig_prompt_init)
        else:
            val = math.sqrt(6. / float(3 * reduce(mul, self.vit.patch_embed.patch_size, 1) + self.prompt_dim)) # noqa
            nn.init.uniform_(self.prompts.data, -val, val) # xavier_uniform initialization

    # 修改此方法以接受外部 prompts_tensor
    def prompt_injection(self, x, prompts_tensor=None):
        if self.num_prompts > 0:
            # 如果提供了 prompts_tensor，则使用它，否则使用 self.prompts
            if prompts_tensor is None:
                actual_prompts = self.prompts.expand(x.shape[0], -1, -1)
            else:
                # 确保传入的 prompts_tensor 的批次大小与输入 x 匹配
                assert prompts_tensor.shape[0] == x.shape[0], "Batch size of prompts_tensor must match x"
                actual_prompts = prompts_tensor

            x = torch.cat((
                x[:,:1,:], # CLS token
                actual_prompts,
                x[:,1:,:] # Patch tokens
            ), dim=1)
        return x
    
    def _collect_layers_features(self, x):
        # collecting features for each layer
        cls_features_per_layer = [] # 存储每层的CLS特征列表
        for i in range(len(self.vit.blocks)):
            x = self.vit.blocks[i](x)
            if i < len(self.vit.blocks) - 1:
                cls_features_per_layer.append(self.vit.blocks[i+1].norm1(x[:, 0]))
            else:
                cls_features_per_layer.append(self.vit.norm(x[:, 0]))
        
        return cls_features_per_layer # 现在返回一个列表，每个元素是一个CLS特征张量
    
    def forward_features(self, x):
        '''
        Forwarding a batch of samples with prompts' embeddings inserted
        We added only the highlighted line of code based on `timm` library
        '''
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        # inject prompts
        x = self.prompt_injection(x) # 这里仍然使用默认的 self.prompts
        # !!end
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.vit.forward_head(x)
        return x
    
    def layers_cls_features(self, x):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.norm_pre(x)
        return self._collect_layers_features(x)
    
    # 修改此方法以接受外部 prompts_tensor
    def layers_cls_features_with_prompts(self, x, prompts_tensor=None):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        # inject prompts
        x = self.prompt_injection(x, prompts_tensor) # 将 prompts_tensor 传递给 prompt_injection
        # !!end
        x = self.vit.norm_pre(x)
        return self._collect_layers_features(x)
