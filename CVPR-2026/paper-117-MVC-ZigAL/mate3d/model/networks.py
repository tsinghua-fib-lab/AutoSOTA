import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip.clip import tokenize
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

temperature = 0.001
clip_vis = 'ViT-B/16'


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x_all = x @ self.text_projection
        x = x[torch.arange(x.shape[0]).to(x.device), tokenized_prompts.argmax(dim=-1).to(
            x.device)] @ self.text_projection.to(x.device)

        return x, x_all


class PromptLearner(nn.Module):
    def __init__(self, device, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.n_ctx  # number of learnable token
        ctx_init = args.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.device = device

        if ctx_init:

            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:

            if args.csc:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(self.device)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args.class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
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
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
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
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts


class HyperNet(nn.Module):
    def __init__(self):
        super(HyperNet, self).__init__()
        self.fc1w_conv = nn.Conv2d(in_channels=112, out_channels=512, kernel_size=3, padding=1)
        self.fc1b_fc = nn.Linear(112, 112)
        self.fc2w_conv = nn.Conv2d(in_channels=112, out_channels=128, kernel_size=3, padding=1)
        self.fc2b_fc = nn.Linear(112, 56)
        self.fc3w_conv = nn.Conv2d(in_channels=112, out_channels=32, kernel_size=3, padding=1)
        self.fc3b_fc = nn.Linear(112, 28)
        self.fc4w_fc = nn.Linear(112, 28)
        self.fc4b_fc = nn.Linear(112, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        fc1w = self.fc1w_conv(x).view(-1, 112, 224, 1, 1)
        fc2w = self.fc2w_conv(x).view(-1, 56, 112, 1, 1)
        fc3w = self.fc3w_conv(x).view(-1, 28, 56, 1, 1)
        fc4w = self.fc4w_fc(self.pool(x).squeeze()).view(-1, 1, 28, 1, 1)

        fc1b = self.fc1b_fc(self.pool(x).squeeze()).view(-1, 112)
        fc2b = self.fc2b_fc(self.pool(x).squeeze()).view(-1, 56)
        fc3b = self.fc3b_fc(self.pool(x).squeeze()).view(-1, 28)
        fc4b = self.fc4b_fc(self.pool(x).squeeze()).view(-1, 1)
        out = {}
        out['fc1w'] = fc1w
        out['fc2w'] = fc2w
        out['fc3w'] = fc3w
        out['fc4w'] = fc4w
        out['fc1b'] = fc1b
        out['fc2b'] = fc2b
        out['fc3b'] = fc3b
        out['fc4b'] = fc4b
        return out


class TargetNet(nn.Module):
    def __init__(self, paras):
        super(TargetNet, self).__init__()
        self.fc1 = nn.Sequential(
            TargetFC(paras['fc1w'], paras['fc1b']),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            TargetFC(paras['fc2w'], paras['fc2b']),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            TargetFC(paras['fc3w'], paras['fc3b']),
            nn.ReLU(),
        )
        self.fc4 = nn.Sequential(
            TargetFC(paras['fc4w'], paras['fc4b']),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class TargetFC(nn.Module):
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):
        input_re = input_
        weight_re = self.weight.squeeze(0)
        bias_re = self.bias.squeeze(0)
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re)
        return out
