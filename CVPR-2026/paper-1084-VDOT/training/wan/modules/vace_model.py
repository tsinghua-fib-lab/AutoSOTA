# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from wan.modules.model import WanModel, WanAttentionBlock, sinusoidal_embedding_1d


class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c
    
    
class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x
    
    
class VaceWanModel(WanModel):
    @register_to_config
    def __init__(self,
                 vace_layers=None,
                 vace_in_dim=None,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        model_type = "t2v"   # TODO: Hard code for both preview and official versions.
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim,
                         num_heads, num_layers, window_size, qk_norm, cross_attn_norm, eps)

        self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim
        self.param_dtype = torch.bfloat16

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # blocks
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                  self.cross_attn_norm, self.eps,
                                  block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None)
            for i in range(self.num_layers)
        ])

        # vace blocks
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                     self.cross_attn_norm, self.eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward_vace(
        self,
        x,
        vace_context,
        seq_len,
        kwargs
    ):
        with torch.amp.autocast('cuda', dtype=self.param_dtype):
            # embeddings
            c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
            c = [u.flatten(2).transpose(1, 2) for u in c]
            # c = torch.cat([
            #     torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
            #             dim=1) for u in c
            # ])
            c = torch.cat([
                torch.cat([
                    u, 
                    u.new_zeros(u.size(0), seq_len - u.size(1), u.size(2))
                ], dim=1) if seq_len - u.size(1) > 0 else u[:, :seq_len, :]
                for u in c
            ])
            #print(f'c shape: {c.shape}')

            # arguments
            new_kwargs = dict(x=x)
            new_kwargs.update(kwargs)

            def create_custom_forward2(module):
                def custom_forward2(*inputs, **kwargs):
                    return module(*inputs, **kwargs)
                return custom_forward2

            for ii, block in enumerate(self.vace_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward2(block),
                        c, **new_kwargs,
                        use_reentrant=False,
                    )
                else:
                    c = block(c, **new_kwargs)

            # for block in self.vace_blocks:
            #     c = block(c, **new_kwargs)
            hints = torch.unbind(c)[:-1]
        return hints

    def forward(
        self,
        x,
        t,
        vace_context,
        context,
        seq_len,
        vace_context_scale=1.0,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        with torch.amp.autocast('cuda', dtype=self.param_dtype):
            
            # embeddings
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
            
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            # print(f'seq lens shape: {seq_lens.shape}')
            assert seq_lens.max() <= seq_len
            x = torch.cat([
                torch.cat([
                    u,
                    u.new_zeros(u.size(0), seq_len - u.size(1), u.size(2))
                ], dim=1) if seq_len - u.size(1) > 0 else u[:, :seq_len, :]
                for u in x
            ])
            # print(f'x shape: {x.shape}')

            # time embeddings
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            #assert e.dtype == torch.float32 and e0.dtype == torch.float32

            # context
            context_lens = None
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))
            
            # arguments
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens)
            
            def create_custom_forward(module):
                def custom_forward(*inputs, **kwargs):
                    return module(*inputs, **kwargs)
                return custom_forward
            
            hints = self.forward_vace(x, vace_context, seq_len, kwargs)
            kwargs['hints'] = hints
            kwargs['context_scale'] = vace_context_scale
            
            for ii, block in enumerate(self.blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, **kwargs,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, **kwargs)

            # head
            x = self.head(x, e)
            # unpatchify
            x = self.unpatchify(x, grid_sizes)

            torch.cuda.empty_cache()

        return torch.stack(x)