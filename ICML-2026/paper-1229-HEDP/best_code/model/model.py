from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ModifiedResNet使用
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        #---------------------------------------------------#
        # 所有的卷积层的步长均为1,但是当步长大于1时,在第二次卷积之后
        # 将会有一个平均池化层
        #---------------------------------------------------#
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 当步长大于1时,将会通过一个平均池化层,
        # 否则将会直接对其跳过
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        # 执行该if语句时,"downsample layer"
        # 将会由二维平均池化,卷积以及BatchNorm2d组成
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        # 当downsample层不为空时,其将会对原始的输入张量执行三个序列操作
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# 对于AttentionPool2d这个类的定义
# 在ModifiedResNet50中被使用
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # nn.Parameter()的作用为作为nn.Module中的可训练参数使用
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # 通过全连接层来获取以下四个映射量
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # 首先进行张量shape的转变,由 batch_size,c,h,w -> (h*w),batch_size,c
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        # (h*w),batch_size,c -> (h*w+1),batch_size,c
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # tensor的shape以及type均不发生改变,所做的只是将位置信息嵌入至原先的tensor中
        # shape:(h*w+1),batch_size,c
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        # 将输入的张量pass through 多头注意力机制模块
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


# CLIP中所使用到的ModifiedResNet50的定义
class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    - 最后的平均池化层我们使用一个 QKV注意力池化层来进行替代
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    # ModifiedResNet50中的残差层的定义
    # 其中的blocks即为标准的残差结构--Bottleneck
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    # ModifiedResNet50的前向传播函数
    def forward(self, x):
        # As to the 3-"stem" convolution layers
        # 在这里我们将三个卷积层集成到一个函数中使用
        # 每一层均为 conv->bn->relu
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        # 这行code的作用在于对输入的张量进行一个type的转换
        x = x.type(self.conv1.weight.dtype)
        # 过三个卷积层
        x = stem(x)
        # 过ModifiedResNet50中的残差结构,共4层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 对于最后的平均池化层我们使用一个QKV注意力池化层来进行替代
        x = self.attnpool(x)

        return x


# transformer模块中 ResidualAttentionBlock 所使用到的LayerNorm层
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


# QuickGELU激活函数的定义
# transformer结构中ResidualAttentionBlock的MLP层中被使用
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# transformer中的核心模块,将会在transformer结构中被使用
# 1.多头注意力层
# 2.LayerNorm层
# 3.MLP层
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        # 多头注意力机制
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        # 在MLP层中首先是进行一次全连接,之后是过QuickGELU激活函数,最后是通过投影进行映射
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
    # 该函数的作用是对输入的张量使用多头注意力机制
    def attention(self, x: torch.Tensor, prefix_prompt=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # prompt生效
        if prefix_prompt is not None:
            #prefix_tuning + prompt_tuning
            px = torch.cat([x[:1,:,:], prefix_prompt, x[1:,:,:]], dim=0)
            return self.attn(x, px, px, need_weights=False, attn_mask=self.attn_mask)[0]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    # 在这个前向传播函数中,对于transformer模块进行了定义以及说明
    def forward(self, x: torch.Tensor,prefix_prompt=None,ln_pre=None):
        #prompt tuning 第一层
        # if prefix_prompt is not None:
        #     p = prefix_prompt.to(x.dtype)
        #     p=p.permute(1, 0, 2)
        #     #p=ln_pre(p)
        #     px = torch.cat([x[:1,:,:], p, x[1:,:,:]], dim=0)
        #     #px拼接了经过ln_pre的prompt
        #     #p=self.ln_1(p)
        #     #这里输入到attention的x经过了self.ln_1运算，一定要一致的话，prompt也要ln_1
        #     x = px + self.attention(self.ln_1(x),prefix_prompt=p)
        # K=V
        if prefix_prompt is not None:
            p = prefix_prompt.to(x.dtype).to(x.device)
            p = p.permute(1, 0, 2)
            p = ln_pre(p)
            # px = torch.cat([x[:1,:,:], p, x[1:,:,:]], dim=0)
            # px拼接了经过ln_pre的prompt
            p = self.ln_1(p)
            #这里输入到attention的x经过了self.ln_1运算，一定要一致的话，prompt也要ln_1
            x = x + self.attention(self.ln_1(x), prefix_prompt=p)
        else:
            x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x # torch.Size([197, 128, 768])


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor,prefix_prompt=None,ln_pre=None):
        #return self.resblocks(x)#如果要加prompt，再这里for循环遍历
        C, B, D = x.shape
        #prompt_loss = torch.zeros((1,), requires_grad=True,device=self.pos_embed.device)
        #selection=None
        for i, blk in enumerate(self.resblocks):#i是层序号，blk是层模型
            if prefix_prompt is not None:
                    #这里损失主要是相似度相关的计算
                p_list = prefix_prompt.forward(i,B)#传入本批次的查询向量q,层序号i，本批次输入x,训练状态，任务id,得到选出的prompt（[Pk, Pv]），loss,x不变
                
            else:
                p_list = None 
            #在register_blk里指定层下，保存该层的注意力矩阵的梯度；这里传入prompt
            x = blk(x, prefix_prompt=p_list,ln_pre=ln_pre)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))#把768转为512 [768,512]

    def forward(self, x: torch.Tensor, instance_tokens=None,prefix_prompt=None):
        # x.shape = [*, width, grid, grid]# 输入：[128.3.224.224] 输出：[128,768,14,14]
        # if len(x.shape)==2:
        #     x = x.reshape(x.shape[0], 768,14,14)
        # else:
        x = self.conv1(x)
        # x.shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 输入：[128,768,14,14] 输出：[128, 768, 196]
        # permute将tensor的维度换位 （0,2,1）表示第2个维度和第3个维度换位
        # x.shape = [*, grid ** 2, width]
        x = x.permute(0, 2, 1)  # 输入：[128, 768, 196] 输出：[128, 196, 768]
        # x.shape = [*, grid ** 2 + 1, width]
        # self.class_embedding.to(x.dtype) [768]
        # torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device) [128,1,768]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
                      dim=1) # 输入：[128, 196, 768] 输出：[128, 197, 768]
        if instance_tokens is not None:
            instance_tokens = instance_tokens.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)#  输入： [10,768] + [128, 1, 768] 
        # 输出：[128, 10, 768]
        x = x + self.positional_embedding.to(x.dtype) #  输入：[128, 197, 768] + [197,768] 输出：[128, 197, 768]
        # [X_cls, Prompts, X]
        if instance_tokens is not None:
            x = torch.cat([x[:,:1,:], instance_tokens, x[:,1:,:]], dim=1) # 输入：[128, 1, 768] [128,10,768] [128,196,768] 输出[128, 207, 768]

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x,prefix_prompt=prefix_prompt,ln_pre = self.ln_pre)#传入x或prompt还是768
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:#将768转为512
            x = x @ self.proj

        return x  # 输出[128,512] 


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,#512
                 # vision
                 image_resolution: int,#224
                 vision_layers: Union[Tuple[int, int, int, int], int],#12
                 vision_width: int, # 768
                 vision_patch_size: int,#16
                 # text
                 context_length: int,#77
                 vocab_size: int,#49408
                 transformer_width: int,#512
                 transformer_heads: int,#8
                 transformer_layers: int#12
                 ):
        super().__init__()

        self.context_length = context_length # 77

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            #图片 transformer 不更新 创建一个仅有结构参数的VisionTransformer实例
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,#768
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim#512
            )
        #文本 transformer 创建一个仅有结构参数的Transformer实例
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size # 49408
        self.token_embedding = nn.Embedding(vocab_size, transformer_width) #(49408,512) 把词库按one hot编码
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width)) #(77,512) 不更新 torch.empty()返回填充有未初始化数据的张量。 张量的形状由可变的参数大小定义。
        self.ln_final = LayerNorm(transformer_width)# LayerNorm 512
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))#(512,512)文本映射，不更新
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))#

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)#(49408,512)
        nn.init.normal_(self.positional_embedding, std=0.01)#(77,512)

        if isinstance(self.visual, ModifiedResNet):#false
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        #这里text是文本编码，经过token_embedding变成向量
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16, fp16是指采用2字节(16位)进行编码存储的一种数据类型"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0] #768
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])#12
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]#16
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)#14
        image_resolution = vision_patch_size * grid_size #224
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    # 从CLIP模型参数取出相关结构参数，并构建一个model,为何取结构参数
    embed_dim = state_dict["text_projection"].shape[1] #512
    context_length = state_dict["positional_embedding"].shape[0] #77
    vocab_size = state_dict["token_embedding.weight"].shape[0] #49408
    transformer_width = state_dict["ln_final.weight"].shape[0] #512
    transformer_heads = transformer_width // 64 #8
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))#12

    model = CLIP(
        embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )
    #input_resolution 224
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    # 注意del删除变量的引用,不是数据
    # if __name__=='__main__':
    #     a=1       # 对象 1 被 变量a引用，对象1的引用计数器为1
    #     b=a       # 对象1 被变量b引用，对象1的引用计数器加1
    #     c=a       #1对象1 被变量c引用，对象1的引用计数器加1
    #     del a     #删除变量a，解除a对1的引用
    #     del b     #删除变量b，解除b对1的引用
    #     print(c)  #最终变量c仍然引用1

    convert_weights(model)
    model.load_state_dict(state_dict) #　从state_dict把全部参数加载进来
    return model.eval()
