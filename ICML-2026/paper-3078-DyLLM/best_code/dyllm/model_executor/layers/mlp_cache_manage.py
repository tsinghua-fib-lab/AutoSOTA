import torch
from torch import nn

from dyllm.utils.context import get_context
from dyllm.utils.metadata import get_metadata
from dyllm.engine.cache_manager import CacheManager


class MLPcache(nn.Module):

    def __init__(self, hidden_dim: int = 0):
        super().__init__()
        self.cache_manager = CacheManager(hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor):
        ctx = get_context()
        metadata = get_metadata()

        if ctx.is_full:
            self.cache_manager.reset_full(x, metadata.running_seqs_tensor, seq_ids_list=metadata.running_seqs)
            self.cache_manager.finish(metadata.finished_seqs)
            return self.cache_manager.get_seqs(metadata.running_seqs_tensor)
        else:
            if ctx.idx_salient_row_k is not None:
                self.cache_manager.scatter_update(
                    metadata.running_seqs_tensor,
                    ctx.idx_salient_row_k,
                    x,
                )
                self.cache_manager.finish(metadata.finished_seqs)
                return self.cache_manager.get_seqs_block(metadata.running_seqs_tensor)

            else:
                self.cache_manager.scatter_update(
                    metadata.running_seqs_tensor,
                    ctx.idx_salient_row,
                    x,
                )
                self.cache_manager.finish(metadata.finished_seqs)
                return self.cache_manager.get_seqs(metadata.running_seqs_tensor)
