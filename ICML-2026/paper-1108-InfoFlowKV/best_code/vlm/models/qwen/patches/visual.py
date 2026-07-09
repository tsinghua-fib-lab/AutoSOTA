"""Patch for Qwen3-VL visual model (model.visual)."""

import torch
import torch.nn.functional as F
from typing import List

from ...base import BasePatch


class VisualPatch(BasePatch):
    """
    Patch model.visual.forward to capture intermediate outputs.

    Usage:
        with VisualPatch(model.visual) as patch:
            output = model(...)
            print(patch.rotary_emb)
            print(patch.hidden_states_per_layer)
    """

    def __init__(self, visual_model):
        self.visual = visual_model
        self.original_forward = None

        # Captured outputs
        self.rotary_emb = None
        self.pos_embeds = None
        self.hidden_states_per_layer: List[torch.Tensor] = []

    def apply(self):
        self.original_forward = self.visual.forward
        self.visual.forward = self._patched_forward

    def remove(self):
        if self.original_forward is not None:
            self.visual.forward = self.original_forward
            self.original_forward = None

    def clear(self):
        self.rotary_emb = None
        self.pos_embeds = None
        self.hidden_states_per_layer.clear()

    def _patched_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        """Same as original forward, but captures intermediate outputs."""
        self.clear()

        # Patch embedding
        hidden_states = self.visual.patch_embed(hidden_states)

        # Position embedding
        pos_embeds = self.visual.fast_pos_embed_interpolate(grid_thw)
        self.pos_embeds = pos_embeds.clone().detach()
        hidden_states = hidden_states + pos_embeds

        # Rotary position embedding
        rotary_pos_emb = self.visual.rot_pos_emb(grid_thw)

        self.rotary_emb = rotary_pos_emb.clone().detach()

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Process blocks
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            self.hidden_states_per_layer.append(hidden_states.clone().detach())

            if layer_num in self.visual.deepstack_visual_indexes:
                idx = self.visual.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.visual.deepstack_merger_list[idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.visual.merger(hidden_states)

        return hidden_states, deepstack_feature_lists
