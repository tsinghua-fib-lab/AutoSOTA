import torch 
import torch.nn as nn
from einops import rearrange

from .xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from .xlstm.blocks.mlstm.block import mLSTMBlockConfig
from .xlstm.blocks.mlstm.layer import mLSTMLayerConfig
from .xlstm.blocks.slstm.block import sLSTMBlockConfig
from .xlstm.blocks.slstm.layer import sLSTMLayerConfig
from .xlstm.components.feedforward import FeedForwardConfig

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel = configs.channel
        self.embedding_dim = configs.embedding_dim
        self.patch_size = configs.patch_size  # patch size
        self.stride = configs.stride  # the stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1  # the number of patches

        # xLSTM configuration
        cfg = xLSTMBlockStackConfig(
            # mlstm_block=mLSTMBlockConfig(
            #     mlstm=mLSTMLayerConfig(conv1d_kernel_size=configs.conv1d_kernel_size, 
            #                            qkv_proj_blocksize=4, num_heads=configs.num_heads)
            # ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(num_heads=configs.num_heads, 
                                       conv1d_kernel_size=configs.conv1d_kernel_size,
                                       bias_init="powerlaw_blockdependent"),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=self.seq_len,
            num_blocks=configs.num_blocks,
            embedding_dim=configs.embedding_dim,
            dropout=configs.dropout,
            slstm_at="all"
        )

        self.embedding = nn.Linear(self.patch_size, self.embedding_dim)  
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.projection = nn.Linear(self.embedding_dim * self.patch_num, self.pred_len)  

    def forward(self, x):

        # x: [Batch, Input length, Channel]
        B, L, C = x.shape

        # Rearrange dimensions 
        x = rearrange(x, 'b l c -> b c l')

        # Create patches
        x = rearrange(x, 'b c l -> (b c) l')  # [(B * C), Num_patches, Patch_size]
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # [(B * C), Num_patches, Patch_size]

        x = self.embedding(x)  # [(B * C), Num_patches, Embedding_dim]

        x = self.xlstm_stack(x)  # [(B * C), Num_patches, Embedding_dim]

        x = x.flatten(1)  # [(B * C), Num_patches * Embedding_dim]

        x = self.projection(x)  # [(B * C), Pred_len]

        # Reshape back to b c l
        x = rearrange(x, '(b c) l -> b c l', b=B, c=C)  # [B, C, Pred_len]

        # Reshape from b c l to b l c
        x = rearrange(x, 'b c l -> b l c')  # [B, Pred_len, C]

        return x

