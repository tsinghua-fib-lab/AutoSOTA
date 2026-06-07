import abc
import math
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from einops import rearrange
from typing import Callable, Optional, Union
from diffusers.models.attention_processor import AttnProcessor, Attention

# copied from https://github.com/huggingface/diffusers/blob/v0.14.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_attend_and_excite.py#L71
# this is not very good for our purpose
# but i leave it here for reference
# you can compare it with the attention store we actually use
# class AttentionStore:
#     '''store attention maps for later use, only cross-attention considered'''
#     @staticmethod
#     def get_empty_store():
#         return {"down": [], "mid": [], "up": []}

#     def __call__(self, attn, is_cross: bool, place_in_unet: str):
#         # TODO: enable self-attention
#         if self.cur_att_layer >= 0 and is_cross:
#             if attn.shape[1] == self.attn_res**2:
#                 self.step_store[place_in_unet].append(attn)

#         self.cur_att_layer += 1
#         if self.cur_att_layer == self.num_att_layers:
#             self.cur_att_layer = 0
#             self.between_steps()

#     def between_steps(self):
#         self.attention_store = self.step_store
#         self.step_store = self.get_empty_store()

#     def get_average_attention(self):
#         average_attention = self.attention_store
#         return average_attention

#     def aggregate_attention(self, from_where: List[str]) -> torch.Tensor:
#         """Aggregates the attention across the different layers and heads at the specified resolution."""
#         out = []
#         attention_maps = self.get_average_attention()
#         for location in from_where:
#             for item in attention_maps[location]:
#                 cross_maps = item.reshape(-1, self.attn_res, self.attn_res, item.shape[-1])
#                 out.append(cross_maps)
#         out = torch.cat(out, dim=0)
#         out = out.sum(0) / out.shape[0]
#         return out

#     def reset(self):
#         self.cur_att_layer = 0
#         self.step_store = self.get_empty_store()
#         self.attention_store = {}

#     def __init__(self, attn_res=64):
#         """
#         Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
#         process
#         """
#         self.num_att_layers = -1
#         self.cur_att_layer = 0
#         self.step_store = self.get_empty_store()
#         self.attention_store = {}
#         self.curr_step_index = 0
#         self.attn_res = attn_res


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        attn = self.forward(attn, is_cross, place_in_unet)
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


# copied from https://github.com/wl-zhao/VPD/blob/main/vpd/models.py#L98
class AttentionStore(AttentionControl):
    '''store attention maps for later use, only cross-attention considered'''
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if not self.train_unet:
            attn = attn.detach()
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if (self.min_size) ** 2 <= attn.shape[1] and attn.shape[1] <= (self.max_size) ** 2:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item for item in self.step_store[key]] for key in self.step_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, min_size=32, max_size=64, train_unet=False):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.min_size = min_size
        self.max_size = max_size
        self.train_unet = train_unet

    def aggregate_attention(self, attn_selector):
        avg_attn = self.get_average_attention()  # all attentions
        attns = {key: {} for key in attn_selector}  # but we only want some of them
        # for each attention category
        for k in attn_selector:
            # for each attention map under this category
            for up_attn in avg_attn[k]:
                # they dont necessarily have the same shape
                # and attention maps of the same shape are gathered together
                size = int(math.sqrt(up_attn.shape[1]))
                reshaped = rearrange(up_attn, 'b (h w) c -> b c h w', h=size)
                if size in attns[k]:
                    attns[k][size].append(reshaped)
                else:
                    attns[k][size] = [reshaped]
            # in the end, we take the average over all maps of the same category & size
            for size, a in attns[k].items():
                attns[k][size] = torch.stack(a).mean(0)
        return attns


# mainly copied from https://github.com/huggingface/diffusers/blob/v0.14.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_attend_and_excite.py#L124
class AttnStoreProcessor:
    '''a custom attention processor, replacing the default one in the pipeline,
    so that we can store the inner attention maps during forward call'''
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    # but the call func is copied from attention_processor.py's AttnProcessor
    # and it's suggested you update it to your diffusers version accordingly
    # there are two modifications and they are specially marked with "MODIFY"
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # MODIFY #1
        is_cross = encoder_hidden_states is not None

        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # MODIFY: this stays the same as the modified diffuers file
        if hasattr(attn, 'feature_gatherer'):
            attn.feature_gatherer.gather(query, 'q')
            attn.feature_gatherer.gather(key, 'k')
            attn.feature_gatherer.gather(value, 'v')

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        if hasattr(attn, 'feature_gatherer'):
            attn.feature_gatherer.gather(value, 'special-value')

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # MODIFY #2
        # https://github.com/huggingface/diffusers/blob/v0.21.2/src/diffusers/models/attention_processor.py#L428C24-L428C24
        # this shape is batch_size x seq_len x dim
        # batch_size is actual_batch_size * heads
        # now we change it to actual_batch_size x heads x seq_len x dim
        head_size = attn.heads
        batch_size, seq_len, dim = attention_probs.shape
        to_store = attention_probs.reshape(batch_size // head_size, head_size, seq_len, dim)
        if self.attnstore:
            self.attnstore(to_store.mean(1), is_cross, self.place_in_unet)
        if hasattr(attn, 'feature_gatherer'):
            attn.feature_gatherer.gather(to_store, 'map')
        # MODIFY END

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        if hasattr(attn, 'feature_gatherer'):
            attn.feature_gatherer.gather(hidden_states, 'special-post-value')

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        if hasattr(attn, 'feature_gatherer'):
            attn.feature_gatherer.gather(attn.to_out[0].weight, 'special-weight')
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if hasattr(attn, 'feature_gatherer'):
            attn.feature_gatherer.gather(hidden_states, 'out')

        return hidden_states


def my_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0,
    is_causal=False, scale=None, enable_gqa=False
):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight

class HunyuanAttnStoreProcessor:
    '''a custom attention processor, replacing the default one in the pipeline,
    so that we can store the inner attention maps during forward call'''
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    # but the call func is copied from attention_processor.py's AttnProcessor
    # and it's suggested you update it to your diffusers version accordingly
    # there are two modifications and they are specially marked with "MODIFY"
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if hasattr(attn, 'feature_gatherer'):
            attn.feature_gatherer.gather(query, 'q')
            attn.feature_gatherer.gather(key, 'k')
            attn.feature_gatherer.gather(value, 'v')

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if hasattr(attn, 'feature_gatherer'):
            attn.feature_gatherer.gather(value[0], 'special-value')

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states, attention_probs = my_scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        to_store = attention_probs
        if self.attnstore:
            self.attnstore(to_store.mean(1), is_cross, self.place_in_unet)
        if hasattr(attn, 'feature_gatherer'):
            attn.feature_gatherer.gather(to_store, 'map')

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        if hasattr(attn, 'feature_gatherer'):
            attn.feature_gatherer.gather(hidden_states, 'special-post-value')

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        if hasattr(attn, 'feature_gatherer'):
            attn.feature_gatherer.gather(attn.to_out[0].weight, 'special-weight')
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class FluxAttnStoreProcessor:
    '''a custom attention processor, replacing the default one in the pipeline,
    so that we can store the inner attention maps during forward call'''
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    # but the call func is copied from attention_processor.py's AttnProcessor
    # and it's suggested you update it to your diffusers version accordingly
    # there are two modifications and they are specially marked with "MODIFY"
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        if hasattr(attn, 'feature_gatherer'):
            if encoder_hidden_states is not None:
                attn.feature_gatherer.gather(query, 'q')
                attn.feature_gatherer.gather(key, 'k')
                attn.feature_gatherer.gather(value, 'v')
            else:
                # batch x text_len+img_len x dim
                attn.feature_gatherer.gather(query[:,attn.text_len:,:], 'q')
                attn.feature_gatherer.gather(key[:,attn.text_len:,:], 'k')
                attn.feature_gatherer.gather(value[:,attn.text_len:,:], 'v')

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if encoder_hidden_states is not None and hasattr(attn, 'feature_gatherer'):
            attn.feature_gatherer.gather(value[0], 'cross-special-value')

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            if hasattr(attn, 'feature_gatherer'):
                text_len = encoder_hidden_states_query_proj.shape[2]
                img_len = query.shape[2]
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        if hasattr(attn, 'feature_gatherer') and encoder_hidden_states is None:
            text_len = attn.text_len

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states, attention_probs = my_scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        to_store = attention_probs  # batch x head x query x key
        cross_attn = to_store[:,:,text_len:,:text_len]
        self_attn = to_store[:,:,text_len:,text_len:]
        if self.attnstore:
            self.attnstore(cross_attn.mean(1), True, self.place_in_unet)
            self.attnstore(self_attn.mean(1), False, self.place_in_unet)
        if hasattr(attn, 'feature_gatherer'):
            attn.feature_gatherer.gather(cross_attn, 'cross-map')
            attn.feature_gatherer.gather(self_attn, 'self-map')
            # o_x = hidden_states[:,:,text_len:,:]  # batch x head x img_len x dim
            # o_p = hidden_states[:,:,:text_len,:]  # batch x head x text_len x dim
            # pseudo_probs = F.softmax(o_x @ o_p.permute(0, 1, 3, 2), dim=3)
            # attn.feature_gatherer.gather(pseudo_probs, 'cross-map')
            # pseudo_probs = F.softmax(o_x @ o_x.permute(0, 1, 3, 2), dim=3)
            # attn.feature_gatherer.gather(pseudo_probs, 'self-map')

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None and hasattr(attn, 'feature_gatherer'):
            attn.feature_gatherer.gather(hidden_states[:,text_len:,:], 'cross-special-post-value')

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            if hasattr(attn, 'feature_gatherer'):
                attn.feature_gatherer.gather(attn.to_out[0].weight, 'cross-special-weight')
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            if hasattr(attn, 'feature_gatherer'):
                attn.feature_gatherer.gather(hidden_states, 'attn-out')

            return hidden_states, encoder_hidden_states
        else:
            if hasattr(attn, 'feature_gatherer'):
                attn.feature_gatherer.gather(hidden_states[:,text_len:,:], 'attn-out')
            return hidden_states


# copied from https://github.com/huggingface/diffusers/blob/v0.14.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_attend_and_excite.py#L646
def register_attention_store(version, pipe, img_size, train_unet, processor_only=False):
    '''call this when loading pipeline'''
    if version == 'hunyuan':
        InjectedProcessor = HunyuanAttnStoreProcessor
    elif version == 'flux':
        InjectedProcessor = FluxAttnStoreProcessor
    else:
        InjectedProcessor = AttnStoreProcessor

    if not hasattr(pipe, 'transformer'):
        if not processor_only:
            attention_store = AttentionStore(img_size // 32, img_size // 16, train_unet)
        else:
            attention_store = None

        attn_procs = {}
        cross_att_count = 0
        for name in pipe.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = InjectedProcessor(
                attnstore=attention_store, place_in_unet=place_in_unet
            )

        pipe.unet.set_attn_processor(attn_procs)
        if not processor_only:
            attention_store.num_att_layers = cross_att_count

    else:
        if not processor_only:
            attention_store = AttentionStore(img_size // 32, img_size // 8, train_unet)
        else:
            attention_store = None

        if not hasattr(pipe.transformer, 'transformer_blocks'):
            for i, basic_block in enumerate(pipe.transformer.blocks):
                # self-attention
                basic_block.attn1.processor = InjectedProcessor(
                    attnstore=attention_store, place_in_unet='up'
                )
                # cross-attention
                basic_block.attn2.processor = InjectedProcessor(
                    attnstore=attention_store, place_in_unet='up'
                )
        else:
            for i, basic_block in enumerate(pipe.transformer.transformer_blocks):
                if hasattr(basic_block, 'attn1'):
                    # self-attention
                    basic_block.attn1.processor = InjectedProcessor(
                        attnstore=attention_store, place_in_unet='up'
                    )
                    # cross-attention
                    basic_block.attn2.processor = InjectedProcessor(
                        attnstore=attention_store, place_in_unet='up'
                    )
                else:
                    basic_block.attn.processor = InjectedProcessor(
                        attnstore=attention_store, place_in_unet='up'
                    )
            if hasattr(pipe.transformer, 'single_transformer_blocks'):
                for i, basic_block in enumerate(pipe.transformer.single_transformer_blocks):
                    basic_block.attn.processor = InjectedProcessor(
                        attnstore=attention_store, place_in_unet='up'
                    )
    return attention_store


def demo_usage(attention_store):
    '''this is how you can retrieve attention maps'''
    # first register attention store at the beginning

    # then, remember to reset it per loop, before calling unet

    # after unet call, gather attention maps
    attention_maps = attention_store.aggregate_attention(
        ['down_cross', 'up_cross']
    )


# https://github.com/adobe-research/custom-diffusion/issues/54
def visualize(attn, filename):
    image = attn
    image = 255 * image / image.max()
    image = image.unsqueeze(-1).expand(*image.shape, 3)
    image = image.numpy().astype(np.uint8)
    image = Image.fromarray(image).resize((256, 256))
    image.save(filename)


def demo_visualize(attention_store):
    root = '../sample_image/attn/'
    attention_maps = attention_store.aggregate_attention(args.attention)
    for category, maps in attention_maps.items():
        print(category)
        for size, attn in maps.items():
            print(size, attn.shape)
            for token in range(7):
                visualize(attn[0,token,:,:].cpu(), root+category+str(size)+'_'+str(token)+'.png')
