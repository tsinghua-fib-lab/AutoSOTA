# ---------------------------------------------------------------------------------
# Portions of this file are derived from UniTS
# - Source: https://github.com/mims-harvard/UniTS
# - Paper: UNITS: A Unified Multi-Task Time Series Model
# - License: MIT License
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import math

import torch
import torch.nn.functional as F
from timm.layers import DropPath, Mlp
from timm.layers.helpers import to_2tuple
from torch import nn

from probts.model.forecaster import Forecaster


def calculate_unfold_output_length(input_length, size, step):
    # Calculate the number of windows
    num_windows = (input_length - size) // step + 1
    return num_windows


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            var_num=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if var_num is not None:
            self.template = nn.Parameter(
                torch.zeros(var_num, dim), requires_grad=True)
            torch.nn.init.normal_(self.template, std=.02)
        self.var_num = var_num

    def forward(self, x, query=None):
        B, N, C = x.shape
        if query is not None:
            q = self.q(query).reshape(
                B, query.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            var_num = query.shape[1]
        else:
            q = self.q(self.template).reshape(1, self.var_num,
                                              self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            q = q.repeat(B, 1, 1, 1)
            var_num = self.var_num
        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, var_num, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DynamicLinear(nn.Module):
    """
    A dynamic linear layer that can interpolate the weight size to support any given input and output feature dimension.
    """

    def __init__(self, in_features=None, out_features=None, fixed_in=0, bias=True):
        super(DynamicLinear, self).__init__()
        assert fixed_in < in_features, "fixed_in < in_features is required !!!"
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.fixed_in = fixed_in

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, out_features):
        """
        Forward pass for the dynamic linear layer.
        """
        fixed_weights = self.weights[:, :self.fixed_in]
        dynamic_weights = self.weights[:, self.fixed_in:]
        this_bias = self.bias
        in_features = x.shape[-1]

        if in_features != self.weights.size(1) or out_features != self.weights.size(0):
            dynamic_weights = F.interpolate(dynamic_weights.unsqueeze(0).unsqueeze(0), size=(
                out_features, in_features-self.fixed_in), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            if self.fixed_in != 0:
                fixed_weights = F.interpolate(fixed_weights.unsqueeze(0).unsqueeze(0), size=(
                    out_features, self.fixed_in), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        if out_features != self.weights.size(0):
            this_bias = F.interpolate(this_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), size=(
                1, out_features), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).squeeze(0)
        return F.linear(x, torch.cat((fixed_weights, dynamic_weights), dim=1), this_bias)


class DynamicLinearMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            prefix_token_length=None,
            group=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Conv1d(in_features, hidden_features,
                             3, groups=group, bias=bias[0], padding=1)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        self.norm = norm_layer(
            hidden_features) if norm_layer is not None else nn.Identity()
        self.seq_fc = DynamicLinear(
            hidden_features//4, hidden_features//4, bias=bias[1], fixed_in=prefix_token_length)
        self.prompt_fc = DynamicLinear(
            hidden_features//4, prefix_token_length, bias=bias[1], fixed_in=prefix_token_length)

        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.hidden_features = hidden_features
        self.prefix_token_length = prefix_token_length

    def dynamic_linear(self, x, prefix_seq_len):
        x_func = x[:, :, prefix_seq_len:]
        x_seq = x[:, :, :prefix_seq_len]
        x_seq_out = self.seq_fc(
            x_seq, x_seq.shape[-1]-self.prefix_token_length)
        x_prompt = self.prompt_fc(x_seq, self.prefix_token_length)
        x = torch.cat((x_prompt, x_seq_out, x_func), dim=-1)
        return x

    def split_dynamic_linear(self, x, prefix_seq_len):
        x1, x2 = x.chunk(2, dim=-2)
        x1 = self.dynamic_linear(x1, prefix_seq_len)
        return torch.cat((x1, x2), dim=-2)

    def forward(self, x, prefix_seq_len, dim=2):
        n, var, l, c = x.shape
        x = x.view(-1, l, c)
        x = x.transpose(-1, -2)
        x = self.fc1(x)
        x = self.split_dynamic_linear(x, prefix_seq_len)
        x = self.act(x)
        x = self.drop1(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.fc2(x).view(n, var, l, c)
        x = self.drop2(x)
        return x


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = nn.Parameter(torch.zeros(
            1, 1, max_len, d_model), requires_grad=True)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)
        self.pe.data.copy_(pe.float())
        del pe

    def forward(self, x, offset=0):
        return self.pe[:, :, offset:offset+x.size(2)]


class SeqAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q, k, v,  # attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VarAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, P, C = x.shape

        qkv = self.qkv(x).reshape(B, N, P, 3, self.num_heads,
                                  self.head_dim).permute(3, 0, 2, 4, 1, 5)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.mean(dim=1, keepdim=False)
        k = k.mean(dim=1, keepdim=False)
        v = v.permute(0, 2, 3, 4, 1).reshape(B, self.num_heads, N, -1)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.view(B, self.num_heads, N, -1, P).permute(0,
                                                        2, 4, 1, 3).reshape(B, N, P, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GateLayer(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        gate_value = self.gate(x)
        return gate_value.sigmoid() * x


class SeqAttBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_seq = SeqAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.ls1 = GateLayer(dim, init_values=init_values)
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask):
        x_input = x
        x = self.norm1(x)
        n_vars, n_seqs = x.shape[1], x.shape[2]
        x = torch.reshape(
            x, (-1, x.shape[-2], x.shape[-1]))
        x = self.attn_seq(x, attn_mask)
        x = torch.reshape(
            x, (-1, n_vars, n_seqs, x.shape[-1]))
        x = x_input + self.drop_path1(self.ls1(x))
        return x


class VarAttBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_var = VarAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = GateLayer(dim, init_values=init_values)
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn_var(self.norm1(x))))
        return x


class MLPBlock(nn.Module):

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            proj_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=None,
            prefix_token_length=0,
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        if mlp_layer is DynamicLinearMlp:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
                prefix_token_length=prefix_token_length,
            )
        else:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
            )
        self.ls2 = GateLayer(dim, init_values=init_values)
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, prefix_seq_len=None):
        if prefix_seq_len is not None:
            x = x + \
                self.drop_path2(
                    self.ls2(self.mlp(self.norm2(x), prefix_seq_len=prefix_seq_len)))
        else:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class BasicBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=8.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            prefix_token_length=0,
    ):
        super().__init__()
        self.seq_att_block = SeqAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
                                         attn_drop=attn_drop, init_values=init_values, proj_drop=proj_drop,
                                         drop_path=drop_path, norm_layer=norm_layer)

        self.var_att_block = VarAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
                                         attn_drop=attn_drop, init_values=init_values, proj_drop=proj_drop,
                                         drop_path=drop_path, norm_layer=norm_layer)

        self.dynamic_mlp = MLPBlock(dim=dim, mlp_ratio=mlp_ratio, mlp_layer=DynamicLinearMlp,
                                    proj_drop=proj_drop, init_values=init_values, drop_path=drop_path,
                                    act_layer=act_layer, norm_layer=norm_layer,
                                    prefix_token_length=prefix_token_length)

    def forward(self, x, prefix_seq_len, attn_mask):
        x = self.seq_att_block(x, attn_mask)
        x = self.var_att_block(x)
        x = self.dynamic_mlp(x, prefix_seq_len=prefix_seq_len)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        assert self.patch_len == self.stride, "non-overlap"
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


class CLSHead(nn.Module):
    def __init__(self, d_model, head_dropout=0):
        super().__init__()
        d_mid = d_model
        self.proj_in = nn.Linear(d_model, d_mid)
        self.cross_att = CrossAttention(d_mid)

        self.mlp = MLPBlock(dim=d_mid, mlp_ratio=8, mlp_layer=Mlp,
                            proj_drop=head_dropout, init_values=None, drop_path=0.0,
                            act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                            prefix_token_length=None)

    def forward(self, x, category_token=None, return_feature=False):
        x = self.proj_in(x)
        B, V, L, C = x.shape
        x = x.view(-1, L, C)
        cls_token = x[:, -1:]
        cls_token = self.cross_att(x, query=cls_token)
        cls_token = cls_token.reshape(B, V, -1, C)

        cls_token = self.mlp(cls_token)
        if return_feature:
            return cls_token
        m = category_token.shape[2]
        cls_token = cls_token.expand(B, V, m, C)
        distance = torch.einsum('nvkc,nvmc->nvm', cls_token, category_token)

        distance = distance.mean(dim=1)
        return distance


class ForecastHead(nn.Module):
    def __init__(self, d_model, patch_len, stride, pad, head_dropout=0, prefix_token_length=None):
        super().__init__()
        d_mid = d_model
        self.proj_in = nn.Linear(d_model, d_mid)
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=int(d_model * 4),
            act_layer=nn.GELU,
            drop=head_dropout,
        )
        self.proj_out = nn.Linear(d_model, patch_len)
        self.pad = pad
        self.patch_len = patch_len
        self.stride = stride
        self.pos_proj = DynamicLinear(
            in_features=128, out_features=128, fixed_in=prefix_token_length)

    def forward(self, x_full, pred_len, token_len):
        x_full = self.proj_in(x_full)
        x_pred = x_full[:, :, -token_len:]
        x = x_full.transpose(-1, -2)
        x = self.pos_proj(x, token_len)
        x = x.transpose(-1, -2)
        x = x + x_pred
        x = self.mlp(x)
        x = self.proj_out(x)

        bs, n_vars = x.shape[0], x.shape[1]
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.fold(x, output_size=(
            pred_len, 1), kernel_size=(self.patch_len, 1), stride=(self.stride, 1))
        x = x.squeeze(dim=-1)
        x = x.reshape(bs, n_vars, -1)
        x = x.permute(0, 2, 1)
        return x


class Model(nn.Module):
    """
    UniTS: Building a Unified Time Series Model
    """

    def __init__(self, args, configs_list, pretrain=False):
        super().__init__()

        # (zhenwei) we do not pretrain the model in this stage
        # if pretrain:
        #     self.right_prob = args.right_prob
        #     self.min_mask_ratio = args.min_mask_ratio
        #     self.max_mask_ratio = args.max_mask_ratio

        # Tokens settings
        self.num_task = len(configs_list)
        self.prompt_tokens = nn.ParameterDict({})
        self.mask_tokens = nn.ParameterDict({})
        self.cls_tokens = nn.ParameterDict({})
        self.category_tokens = nn.ParameterDict({})

        for i in range(self.num_task):
            dataset_name = configs_list[i][1]['dataset']
            task_data_name = configs_list[i][0]
            if dataset_name not in self.prompt_tokens:
                self.prompt_tokens[dataset_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], args.prompt_num, args.d_model)
                torch.nn.init.normal_(
                    self.prompt_tokens[dataset_name], std=.02)
                self.mask_tokens[dataset_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], 1, args.d_model)

            if configs_list[i][1]['task_name'] == 'classification':
                self.category_tokens[task_data_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], configs_list[i][1]['num_class'], args.d_model)
                torch.nn.init.normal_(
                    self.category_tokens[task_data_name], std=.02)
                self.cls_tokens[task_data_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], 1, args.d_model)
                torch.nn.init.normal_(self.cls_tokens[task_data_name], std=.02)
            if pretrain:
                self.cls_tokens[task_data_name] = torch.zeros(
                    1, configs_list[i][1]['enc_in'], 1, args.d_model)
                torch.nn.init.normal_(self.cls_tokens[task_data_name], std=.02)

        self.cls_nums = {}
        for i in range(self.num_task):
            task_data_name = configs_list[i][0]
            if configs_list[i][1]['task_name'] == 'classification':
                self.cls_nums[task_data_name] = configs_list[i][1]['num_class']
            elif configs_list[i][1]['task_name'] == 'long_term_forecast':
                remainder = configs_list[i][1]['seq_len'] % args.patch_len
                if remainder == 0:
                    padding = 0
                else:
                    padding = args.patch_len - remainder
                input_token_len = calculate_unfold_output_length(
                    configs_list[i][1]['seq_len']+padding, args.stride, args.patch_len)
                input_pad = args.stride * \
                    (input_token_len - 1) + args.patch_len - \
                    configs_list[i][1]['seq_len']
                pred_token_len = calculate_unfold_output_length(
                    configs_list[i][1]['pred_len']-input_pad, args.stride, args.patch_len)
                real_len = configs_list[i][1]['seq_len'] + \
                    configs_list[i][1]['pred_len']
                self.cls_nums[task_data_name] = [pred_token_len,
                                                 configs_list[i][1]['pred_len'], real_len]

        self.configs_list = configs_list

        ### model settings ###
        self.prompt_num = args.prompt_num
        self.stride = args.stride
        self.pad = args.stride
        self.patch_len = args.patch_len

        # input processing
        self.patch_embeddings = PatchEmbedding(
            args.d_model, args.patch_len, args.stride, args.stride, args.dropout)
        self.position_embedding = LearnablePositionalEmbedding(args.d_model)
        self.prompt2forecat = DynamicLinear(128, 128, fixed_in=args.prompt_num)

        # basic blocks
        self.block_num = args.e_layers
        self.blocks = nn.ModuleList(
            [BasicBlock(dim=args.d_model, num_heads=args.n_heads, qkv_bias=False, qk_norm=False,
                        mlp_ratio=8., proj_drop=args.dropout, attn_drop=0., drop_path=0.,
                        init_values=None, prefix_token_length=args.prompt_num) for l in range(args.e_layers)]
        )

        # output processing
        self.cls_head = CLSHead(args.d_model, head_dropout=args.dropout)
        self.forecast_head = ForecastHead(
            args.d_model, args.patch_len, args.stride, args.stride, prefix_token_length=args.prompt_num, head_dropout=args.dropout)
        if pretrain:
            self.pretrain_head = ForecastHead(
                args.d_model, args.patch_len, args.stride, args.stride, prefix_token_length=1, head_dropout=args.dropout)

    def tokenize(self, x, mask=None):
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        if mask is not None:
            x = x.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x * x, dim=1) /
                               torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(dim=1)
        else:
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        x = x.permute(0, 2, 1)
        remainder = x.shape[2] % self.patch_len
        if remainder != 0:
            padding = self.patch_len - remainder
            x = F.pad(x, (0, padding))
        else:
            padding = 0
        x, n_vars = self.patch_embeddings(x)
        return x, means, stdev, n_vars, padding

    def prepare_prompt(self, x, n_vars, prefix_prompt, task_prompt, task_prompt_num, task_name=None, mask=None):
        x = torch.reshape(
            x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        # append prompt tokens
        this_prompt = prefix_prompt.repeat(x.shape[0], 1, 1, 1)

        if task_name == 'forecast':
            this_mask_prompt = task_prompt.repeat(
                x.shape[0], 1, task_prompt_num, 1)
            init_full_input = torch.cat(
                (this_prompt, x, this_mask_prompt), dim=-2)
            init_mask_prompt = self.prompt2forecat(init_full_input.transpose(
                -1, -2), init_full_input.shape[2]-prefix_prompt.shape[2]).transpose(-1, -2)
            this_function_prompt = init_mask_prompt[:, :, -task_prompt_num:]
            x = torch.cat((this_prompt, x, this_function_prompt), dim=2)
            x[:, :, self.prompt_num:] = x[:, :, self.prompt_num:] + \
                self.position_embedding(x[:, :, self.prompt_num:])
        elif task_name == 'classification':
            this_function_prompt = task_prompt.repeat(x.shape[0], 1, 1, 1)
            x = x + self.position_embedding(x)
            x = torch.cat((this_prompt, x, this_function_prompt), dim=2)
        elif task_name == 'imputation':
            # fill the masked parts with mask tokens
            # for imputation, masked is 0, unmasked is 1, so here to reverse mask
            mask = 1-mask
            mask = mask.permute(0, 2, 1)
            mask = self.mark2token(mask)
            mask_repeat = mask.unsqueeze(dim=-1)

            mask_token = task_prompt
            mask_repeat = mask_repeat.repeat(1, 1, 1, x.shape[-1])
            x = x * (1-mask_repeat) + mask_token * mask_repeat

            init_full_input = torch.cat((this_prompt, x), dim=-2)
            init_mask_prompt = self.prompt2forecat(
                init_full_input.transpose(-1, -2), x.shape[2]).transpose(-1, -2)
            # keep the unmasked tokens and fill the masked ones with init_mask_prompt.
            x = x * (1-mask_repeat) + init_mask_prompt * mask_repeat
            x = x + self.position_embedding(x)
            x = torch.cat((this_prompt, x), dim=2)
        elif task_name == 'anomaly_detection':
            x = x + self.position_embedding(x)
            x = torch.cat((this_prompt, x), dim=2)

        return x

    def mark2token(self, x_mark):
        x_mark = x_mark.unfold(
            dimension=-1, size=self.patch_len, step=self.stride)
        x_mark = x_mark.mean(dim=-1)
        x_mark = (x_mark > 0).float()
        return x_mark

    def backbone(self, x, prefix_len, seq_len):
        attn_mask = None
        for block in self.blocks:
            x = block(x, prefix_seq_len=prefix_len +
                      seq_len, attn_mask=attn_mask)
        return x

    def forecast(self, x, x_mark, task_id):
        dataset_name = self.configs_list[task_id][1]['dataset']
        task_data_name = self.configs_list[task_id][0]
        prefix_prompt = self.prompt_tokens[dataset_name]
        task_prompt = self.mask_tokens[dataset_name]
        task_prompt_num = self.cls_nums[task_data_name][0]
        task_seq_num = self.cls_nums[task_data_name][1]
        real_seq_len = self.cls_nums[task_data_name][2]

        x, means, stdev, n_vars, _ = self.tokenize(x)

        x = self.prepare_prompt(
            x, n_vars, prefix_prompt, task_prompt, task_prompt_num, task_name='forecast')

        seq_token_len = x.shape[-2]-prefix_prompt.shape[2]
        x = self.backbone(x, prefix_prompt.shape[2], seq_token_len)

        x = self.forecast_head(
            x, real_seq_len, seq_token_len)
        x = x[:, -task_seq_num:]

        # De-Normalization from Non-stationary Transformer
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))

        return x

    def classification(self, x, x_mark, task_id):
        dataset_name = self.configs_list[task_id][1]['dataset']
        task_data_name = self.configs_list[task_id][0]
        prefix_prompt = self.prompt_tokens[dataset_name]
        task_prompt = self.cls_tokens[task_data_name]
        task_prompt_num = 1
        category_token = self.category_tokens[task_data_name]

        x, means, stdev, n_vars, _ = self.tokenize(x)

        seq_len = x.shape[-2]

        x = self.prepare_prompt(
            x, n_vars, prefix_prompt, task_prompt, task_prompt_num, task_name='classification')

        x = self.backbone(x, prefix_prompt.shape[2], seq_len)

        x = self.cls_head(x, category_token)

        return x

    def imputation(self, x, x_mark, mask, task_id):
        dataset_name = self.configs_list[task_id][1]['dataset']
        prefix_prompt = self.prompt_tokens[dataset_name]
        task_prompt = self.mask_tokens[dataset_name]

        seq_len = x.shape[1]
        x, means, stdev, n_vars, padding = self.tokenize(x, mask)

        x = self.prepare_prompt(
            x, n_vars, prefix_prompt, task_prompt, None, mask=mask, task_name='imputation')
        seq_token_len = x.shape[-2]-prefix_prompt.shape[2]
        x = self.backbone(x, prefix_prompt.shape[2], seq_token_len)

        x = self.forecast_head(
            x, seq_len+padding, seq_token_len)
        x = x[:, :seq_len]

        # De-Normalization from Non-stationary Transformer
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))

        return x

    def anomaly_detection(self, x, x_mark, task_id):
        dataset_name = self.configs_list[task_id][1]['dataset']
        prefix_prompt = self.prompt_tokens[dataset_name]

        seq_len = x.shape[1]
        x, means, stdev, n_vars, padding = self.tokenize(x)

        x = self.prepare_prompt(x, n_vars, prefix_prompt,
                                None, None, task_name='anomaly_detection')
        seq_token_len = x.shape[-2]-prefix_prompt.shape[2]
        x = self.backbone(x, prefix_prompt.shape[2], seq_token_len)

        x = self.forecast_head(
            x, seq_len+padding, seq_token_len)
        x = x[:, :seq_len]

        # De-Normalization from Non-stationary Transformer
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))

        return x

    def random_masking(self, x, min_mask_ratio, max_mask_ratio):
        """
        Perform per-sample random masking.
        """
        N, V, L, D = x.shape  # batch, var, length, dim

        # Calculate mask ratios and lengths to keep for each sample in the batch
        mask_ratios = torch.rand(N, device=x.device) * \
            (max_mask_ratio - min_mask_ratio) + min_mask_ratio
        len_keeps = (L * (1 - mask_ratios)).long()

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)

        # Create a range tensor and compare with len_keeps for mask generation
        range_tensor = torch.arange(L, device=x.device).expand(N, L)
        mask = (range_tensor >= len_keeps.unsqueeze(1))

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.float()

        return mask

    def right_masking(self, x, min_mask_ratio, max_mask_ratio):
        N, V, L, D = x.shape  # batch, var, length, dim

        # Randomly choose a mask ratio for each sample within the specified range
        mask_ratios = torch.rand(N, device=x.device) * \
            (max_mask_ratio - min_mask_ratio) + min_mask_ratio
        len_keeps = (L * (1 - mask_ratios)).long()

        # Binary mask creation without a for loop
        len_keeps_matrix = len_keeps.unsqueeze(1).expand(N, L)
        indices = torch.arange(L, device=x.device).expand_as(len_keeps_matrix)
        mask = indices >= len_keeps_matrix
        mask = mask.float()

        return mask

    def choose_masking(self, x, right_prob, min_mask_ratio, max_mask_ratio):
        # Generate a random number to decide which masking function to use
        if torch.rand(1).item() > right_prob:
            return self.random_masking(x, min_mask_ratio, max_mask_ratio)
        else:
            return self.right_masking(x, min_mask_ratio, max_mask_ratio)

    def get_mask_seq(self, mask, seq_len):
        mask_seq = mask.unsqueeze(dim=-1).repeat(1, 1, self.patch_len)
        mask_seq = mask_seq.permute(0, 2, 1)
        mask_seq = mask_seq.masked_fill(mask_seq == 0, -1e9)
        # Fold operation
        mask_seq = torch.nn.functional.fold(mask_seq, output_size=(
            seq_len, 1), kernel_size=(self.patch_len, 1), stride=(self.stride, 1))
        # Apply threshold to bring back to 0/1 values
        mask_seq = (mask_seq > 0).float()
        mask_seq = mask_seq.squeeze(dim=-1).squeeze(dim=1)
        return mask_seq

    def pretraining(self, x, x_mark, task_id, enable_mask=False):
        dataset_name = self.configs_list[task_id][1]['dataset']
        task_data_name = self.configs_list[task_id][0]
        prefix_prompt = self.prompt_tokens[dataset_name]
        mask_token = self.mask_tokens[dataset_name]
        cls_token = self.cls_tokens[task_data_name]

        seq_len = x.shape[1]
        x, means, stdev, n_vars, padding = self.tokenize(x)
        seq_token_len = x.shape[-2]

        # append prompt tokens
        x = torch.reshape(
            x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        # prepare prompts
        this_prompt = prefix_prompt.repeat(x.shape[0], 1, 1, 1)

        if enable_mask:
            mask = self.choose_masking(x, self.right_prob,
                                       self.min_mask_ratio, self.max_mask_ratio)
            mask_repeat = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            mask_repeat = mask_repeat.repeat(1, x.shape[1], 1, x.shape[-1])
            x = x * (1-mask_repeat) + mask_token * mask_repeat  # todo

            init_full_input = torch.cat((this_prompt, x), dim=-2)
            init_mask_prompt = self.prompt2forecat(
                init_full_input.transpose(-1, -2), x.shape[2]).transpose(-1, -2)
            # keep the unmasked tokens and fill the masked ones with init_mask_prompt.
            x = x * (1-mask_repeat) + init_mask_prompt * mask_repeat
            x = x + self.position_embedding(x)
            mask_seq = self.get_mask_seq(mask, seq_len+padding)
            mask_seq = mask_seq[:, :seq_len]
        this_function_prompt = cls_token.repeat(x.shape[0], 1, 1, 1)
        x = torch.cat((this_prompt, x, this_function_prompt), dim=2)

        x = self.backbone(x, prefix_prompt.shape[2], seq_token_len)

        if enable_mask:
            mask_dec_out = self.forecast_head(
                x[:, :, :-1], seq_len+padding, seq_token_len)
            mask_dec_out = mask_dec_out[:, :seq_len]
            # De-Normalization from Non-stationary Transformer
            mask_dec_out = mask_dec_out * \
                (stdev[:, 0, :].unsqueeze(1).repeat(
                    1, mask_dec_out.shape[1], 1))
            mask_dec_out = mask_dec_out + \
                (means[:, 0, :].unsqueeze(1).repeat(
                    1, mask_dec_out.shape[1], 1))
            cls_dec_out = self.cls_head(x, return_feature=True)
            # detach grad of the forecasting on tokens
            fused_dec_out = torch.cat(
                (cls_dec_out, x[:, :, self.prompt_num:-1].detach()), dim=2)
            cls_dec_out = self.pretrain_head(
                fused_dec_out, seq_len+padding, seq_token_len)
            cls_dec_out = cls_dec_out[:, :seq_len]
            cls_dec_out = cls_dec_out * \
                (stdev[:, 0, :].unsqueeze(1).repeat(
                    1, cls_dec_out.shape[1], 1))
            cls_dec_out = cls_dec_out + \
                (means[:, 0, :].unsqueeze(1).repeat(
                    1, cls_dec_out.shape[1], 1))

            return cls_dec_out, mask_dec_out, mask_seq
        else:
            return cls_dec_out

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None,
                mask=None, task_id=None, task_name=None, enable_mask=None):
        task_id = 0

        # if task_name == 'long_term_forecast' or task_name == 'short_term_forecast':
        dec_out = self.forecast(x_enc, x_mark_enc, task_id)
        return dec_out  # [B, L, D]
        # if task_name == 'imputation':
        #     dec_out = self.imputation(
        #         x_enc, x_mark_enc, mask, task_id)
        #     return dec_out  # [B, L, D]
        # if task_name == 'anomaly_detection':
        #     dec_out = self.anomaly_detection(x_enc, x_mark_enc, task_id)
        #     return dec_out  # [B, L, D]
        # if task_name == 'classification':
        #     dec_out = self.classification(x_enc, x_mark_enc, task_id)
        #     return dec_out  # [B, N]
        # if 'pretrain' in task_name:
        #     dec_out = self.pretraining(x_enc, x_mark_enc, task_id,
        #                                enable_mask=enable_mask)
        #     return dec_out
        # return None


class UniTS(Forecaster):
    def __init__(
        self,
        ckpt_path: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.no_training = True
        
        if (type(self.context_length).__name__=='list'):
            context_length = max(context_length)
            
        if (type(self.prediction_length).__name__=='list'):
            prediction_length = max(prediction_length)

        args, configs_list = self.generate_units_default_args(self.dataset)
        self.model = Model(args, configs_list, pretrain=False)
        
        pretrain_weight_path = ckpt_path

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(pretrain_weight_path, map_location=device)['student']
        ckpt = {}
        for k, v in state_dict.items():
            if not ('cls_prompts' in k):
                k = k.replace('module.', '') if 'module.' in k else k
                ckpt[k] = v
        
        msg = self.model.load_state_dict(ckpt, strict=False)
        if len(msg.missing_keys) > 0:
            print(f"""Warning: There are missing keys in the pretrained model: {msg.missing_keys}, 
                which may cause prediction results less accurate.""")


    def generate_units_default_args(self, dataset_name='ETTh1'):
        class Args:
            def __init__(self):
                self.d_model = 128
                self.n_heads = 8
                self.e_layers = 3
                self.prompt_num = 10
                self.dropout = 0.1
                self.patch_len = 16
                self.stride = 16
                self.batch_size = 32

        args = Args()

        # parse dataset names - ECL, ETTh1, Exchange, ILI, Traffic, Weather
        units_valid_dataset_map = {
            'ECL': ['ECL', 'electricity'],
            'ETTh1': ['ETT'],
            'Exchange': ['Exchange'],
            'ILI': ['ILI'],
            'Traffic': ['Traffic'],
            'Weather': ['Weather']
        }

        units_dataset_name = 'DEFAULT'
        for key, value_list in units_valid_dataset_map.items():
            if any(substring.lower() in dataset_name for substring in value_list):
                units_dataset_name = key
                break
        task_name = f"LTF_{units_dataset_name}_p{self.prediction_length}"

        task_data_config = {
            task_name: {
                "task_name": "long_term_forecast",
                "dataset": units_dataset_name,
                "data": units_dataset_name,
                "embed": "timeF",
                "features": "M",
                "seq_len": self.context_length,
                "label_len": 48,
                "pred_len": self.prediction_length,
                "enc_in": self.target_dim,
                "dec_in": self.target_dim,
                "c_out": self.target_dim
            }
        }
        task_data_config_list = []
        for task_name, task_config in task_data_config.items():
            task_config['max_batch'] = args.batch_size
            task_data_config_list.append([task_name, task_config])
        return args, task_data_config_list


    def forecast(self, batch_data, pred_len=None, dataset_name=None, *args, **kwargs):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = inputs[:, -self.context_length:]
        B, _, K = inputs.shape
        point_forecast = self.model.forward(inputs, None)
        return point_forecast.unsqueeze(1)
