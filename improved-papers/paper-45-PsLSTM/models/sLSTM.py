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

        # sLSTM configuration
        cfg = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=configs.num_heads,
                    conv1d_kernel_size=configs.conv1d_kernel_size,
                    bias_init="powerlaw_blockdependent"
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=1.3,
                    act_fn="gelu"
                ),
            ),
            context_length=self.seq_len,
            num_blocks=configs.num_blocks,
            embedding_dim=configs.embedding_dim,
            slstm_at="all"
        )

       
        self.embedding = nn.Linear(self.channel, self.embedding_dim)
        
        self.xlstm_stack = xLSTMBlockStack(cfg)

        self.projection = nn.Linear(self.embedding_dim, self.pred_len * self.channel)

    def forward(self, x):

        # x: [Batch, Input length, Channel]
        B, L, C = x.shape

        x = self.embedding(x)  # [Batch, Input length, Embedding_dim]

        x = self.xlstm_stack(x)  # [Batch, Input length, Embedding_dim]

        x = x[:, -1, :]  # [Batch, Embedding_dim]

        x = self.projection(x)  # [Batch, Pred_len * Channel]

        x = x.view(-1, self.pred_len, self.channel)  # [Batch, Pred_len, Channel]

        return x
