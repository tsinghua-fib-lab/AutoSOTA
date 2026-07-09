import os
import torch
import torch.nn as nn

from . import clip
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()



def load_clip_to_cpu():
    backbone_name = 'ViT-B/16'#'ViT-B/16'

    # First check if we have a pre-converted model in the cache
    local_path = os.path.join(os.path.expanduser("~/.cache/clip"), "ViT-B-16.pt")
    if os.path.exists(local_path):
        jit_model = torch.jit.load(local_path, map_location="cpu").eval()
        model = clip.build_model(jit_model.state_dict())
        return model

    url = clip._MODELS[backbone_name]
    # 模型参数下载，并返回模型文件存储位置
    model_path = clip._download(url)

    try:
        # loading JIT archive，JIT存储格式的模型加载到CPU，model为CLIP全部模型参数
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding # torch.Size([77, 512])
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND torch.Size([77, 2, 512])
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD ([2, 77, 512])
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

# 针对prompt的特有学习器
# clip_model不更新参数，都不是self.clip_model，只有self.ctx是待优化的
class PromptLearner(nn.Module):
    def __init__(self, n_ctx, classnames, clip_model,class_token_position='end'):
        super().__init__()
        n_cls = len(classnames) # 类别数量2
        
        
        dtype = clip_model.dtype # 规定后续的向量设计按照固定的数据类型
        
        prompt_prefix = " ".join(["X"] * n_ctx)#'X X X X X X X X X X X X X X X X' prompt的前缀占位符

        device = clip_model.token_embedding.weight.device # 类型为torch.device，str值为CPU
       
        classnames = [name.replace("_", " ") for name in classnames]#['real', 'fake'] 去除类名称中的下划线，为了加入prompt模板？
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] #[1, 1] 
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # torch.Size([2, 77, 512]) clip模型在此处利用预训练模型编码 1位的prefix，16位的prompts，60的后缀(包含了标签)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS#torch.Size([2, 1, 512])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS 把非n_ctx要求（前面都是X）的向量长度即为后缀 #torch.Size([2, 60, 512])

        self.n_cls = n_cls # 类别数 2
        self.n_ctx = n_ctx # 16 prompt模板前缀的长度
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = class_token_position #'end' 表示类别的单词的位置，在句子的尾部

    def forward(self,ctx):
        
        if ctx.dim() == 2: # 维度是2
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) 

        prefix = self.token_prefix # torch.Size([2, 1, 512])
        suffix = self.token_suffix # torch.Size([2, 60, 512])

        if self.class_token_position == "end":
            prompts = torch.cat(
                [   prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [   prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts 

