# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from models.sit import SiT



class SiTELIT(SiT):
    """
    SiT model with ELIT (Efficient Latent Image Transformer) extensions.
    Extends the base SiT with token masking and read/write operations for
    efficient inference with variable compute budgets.
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        enable_repa=False,
        z_dims=None,
        projector_dim=2048,
        # ELIT parameters
        dit_encoder_depth=0,
        dit_decoder_depth=0,
        enable_elit=True,
        group_size=8,
        elit_max_mask_prob=0.5,
        elit_min_mask_prob=None,
        elit_read_depth=1,
        elit_write_depth=1,
        **block_kwargs # fused_attn
    ):
        # Call parent init - sets up the base SiT architecture
        super().__init__(
            path_type=path_type,
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            encoder_depth=encoder_depth,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            use_cfg=use_cfg,
            enable_repa=enable_repa,
            z_dims=z_dims,
            projector_dim=projector_dim,
            **block_kwargs,
        )
        
        # ELIT parameters
        self.dit_encoder_depth = dit_encoder_depth
        self.dit_decoder_depth = dit_decoder_depth
        self.enable_elit = enable_elit
        self.group_size = group_size
        self.elit_max_mask_prob = elit_max_mask_prob
        self.elit_min_mask_prob = elit_min_mask_prob if elit_min_mask_prob is not None else elit_max_mask_prob
        self.elit_read_depth = elit_read_depth
        self.elit_write_depth = elit_write_depth
        
        # Initialize ELIT components if enabled
        self.input_token_editor = None
        self.output_token_editor = None
        self.input_token_gather = None
        self.output_token_scatter = None
        self.input_token_masking_strategy = None
        
        if self.enable_elit:
            from elit_utils.token_editors import (
                SequentialInputTokenEditor,
                SequentialOutputTokenEditor,
                LatentTokensGroupAdapter,
                InputTokensGatherMaskedTokens,
                OutputTokensRestoreMaskedTokens,
                FiTLayerWModulationLayer,
                FiTWriteLayer
            )
            from elit_utils.masking_strategies import MultiOrderedTokenMaskingStrategy
            
            # Read operation config
            latent_adapter_config = {
                'target': LatentTokensGroupAdapter,
                "window_size": group_size,
                "patch_channels": hidden_size,
                "use_learnable_positional_encoding": True,
                "use_latent_learnable_positional_encoding": True, # since we are not using RoPE, we need to use learnable positional encoding for latent tokens
                "zero_init_latent_tokens": False,
                "fit_layer_config":  {
                    "target": FiTLayerWModulationLayer,
                    "window_size": group_size,
                    "patch_channels": hidden_size,
                    "num_heads": num_heads,
                    "depth": elit_read_depth}}
            
            # Read operation 
            self.input_token_editor = SequentialInputTokenEditor({
                "adapters": [latent_adapter_config]
            })
            
            # Input token gather
            self.input_token_gather = InputTokensGatherMaskedTokens()
            
            # Output token scatter  
            self.output_token_scatter = OutputTokensRestoreMaskedTokens()
            
            # Output token editor with FiTWriteLayer
            write_layer_config = {
                "target": FiTWriteLayer,
                "window_size": group_size,
                "depth": elit_write_depth,
                "qk_norm": True,
                "patch_channels": hidden_size,
                "num_heads": num_heads,
                "use_learnable_positional_encoding": True,
                "mask_aware_attn": False
            }
            
            self.output_token_editor = SequentialOutputTokenEditor({
                "adapters": [write_layer_config]
            })
            
            # Define masking strategy
            self.input_token_masking_strategy = MultiOrderedTokenMaskingStrategy({
                "max_mask_prob": elit_max_mask_prob,
                "min_mask_prob": elit_min_mask_prob,
                "window_size": group_size
            })
            
            # Initialize ELIT-specific weights
            self.input_token_editor.reset_parameters()
            self.output_token_editor.reset_parameters()
            
            # Zero-init ELIT output projections so read/write start as near-identity.
            # This matches the DiT block pattern where adaLN gates are zero-initialized.
            self._zero_init_elit_layers()

    def _zero_init_elit_layers(self):
        """Zero-init output projections of ELIT read/write layers for stable training.
        
        This mirrors the DiT convention of zero-initializing adaLN modulation gates
        so that new components start as near-identity and gradually learn.
        """
        import torch.nn as nn
        
        # Zero-init read layer's adaLN modulation (gates start at zero = identity)
        for adapter in self.input_token_editor.adapters:
            if hasattr(adapter, 'fit_layer'):
                fit_layer = adapter.fit_layer
                if hasattr(fit_layer, 'adaLN_modulation'):
                    nn.init.constant_(fit_layer.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(fit_layer.adaLN_modulation[-1].bias, 0)
        
        # Zero-init write layer's cross-attention output projection
        for adapter in self.output_token_editor.adapters:
            if hasattr(adapter, 'patches_attend_to_latents'):
                for attn_layer in adapter.patches_attend_to_latents:
                    nn.init.constant_(attn_layer.proj.weight, 0)
                    nn.init.constant_(attn_layer.proj.bias, 0)
            if hasattr(adapter, 'write_ff'):
                for ff_layer in adapter.write_ff:
                    # Zero-init last linear in MLP (fc2)
                    nn.init.constant_(ff_layer.fc2.weight, 0)
                    if ff_layer.fc2.bias is not None:
                        nn.init.constant_(ff_layer.fc2.bias, 0)

    # ELIT helper methods

    def maybe_apply_input_token_editor(self, x, thw_shape, block_kwargs):
        if self.enable_elit:
            if self.input_token_editor is not None:
                x, thw_shape, block_kwargs = self.input_token_editor(x, thw_shape, block_kwargs=block_kwargs)
            
            if self.input_token_gather is not None:
                x, thw_shape, block_kwargs = self.input_token_gather(x, thw_shape, block_kwargs=block_kwargs)
        
        return x, thw_shape, block_kwargs
    
    def maybe_apply_output_token_editor(self, x, thw_shape, block_kwargs):
        if self.enable_elit:
            if self.output_token_scatter is not None:
                x, thw_shape, block_kwargs = self.output_token_scatter(x, thw_shape, block_kwargs=block_kwargs)
                
            if self.output_token_editor is not None:
                x, thw_shape, block_kwargs = self.output_token_editor(x, thw_shape, block_kwargs=block_kwargs)
                
        return x, thw_shape, block_kwargs 
        
    def forward(self, x, t, y, return_logvar=False, inference_budget=None, **kwargs):
        """
        Forward pass of SiTELIT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels

        Returns:
            x: (N, out_channels, H, W) denoised output
            zs: list of projected representations (only when REPA is enabled, else None)
        """
        thw_shape = (1, x.shape[2] // (self.patch_size), x.shape[3] // (self.patch_size))
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        N, T, D = x.shape
        

        # Compute input masking
        if self.enable_elit and self.input_token_masking_strategy is not None:
            token_keep_mask, token_masking_info = self.input_token_masking_strategy(x, thw_shape, inference_budget=inference_budget)
        
            
        # Timestep and class embedding
        t_embed = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t_embed + y                                # (N, D)

        # DIT encoder blocks
        zs = None
        start_block = 0
        end_block = self.encoder_depth
        for i, block in enumerate(self.blocks[start_block:end_block], start=start_block):
            x = block(x, c)                      # (N, T, D)
            if self.enable_repa and self.projectors is not None and (i + 1) == self.encoder_depth:
                zs = [projector(x.reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
        
        
        # ELIT read operation 
        if self.enable_elit:
            block_kwargs_elit = {}
            block_kwargs_elit['input_keep_mask'] = token_keep_mask
            block_kwargs_elit['modulation'] = c
            x, thw_shape, block_kwargs_elit = self.maybe_apply_input_token_editor(x, thw_shape, block_kwargs=block_kwargs_elit)
            
        
        # Core blocks 
        start_block = self.encoder_depth
        end_block = len(self.blocks) - self.dit_decoder_depth
        for i, block in enumerate(self.blocks[start_block:end_block], start=start_block):
            x = block(x, c)                      # (N, T, D)
        
        
        # ELIT write operation
        if self.enable_elit:
            x, thw_shape, block_kwargs_elit = self.maybe_apply_output_token_editor(x, thw_shape, block_kwargs=block_kwargs_elit)
                
            
        # DIT decoder blocks
        start_block = len(self.blocks) - self.dit_decoder_depth
        end_block = len(self.blocks)
        for i, block in enumerate(self.blocks[start_block:end_block], start=start_block):
            x = block(x, c)                      # (N, T, D)
        
        
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x, zs


def ELIT_SiT_XL_2(**kwargs):
    return SiTELIT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, 
                dit_encoder_depth=4, dit_decoder_depth=4, **kwargs)

def ELIT_SiT_XL_4(**kwargs):
    return SiTELIT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16,
                   dit_encoder_depth=4, dit_decoder_depth=4, **kwargs)

def ELIT_SiT_XL_8(**kwargs):
    return SiTELIT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, 
                   dit_encoder_depth=4, dit_decoder_depth=4, **kwargs)

def ELIT_SiT_L_2(**kwargs):
    return SiTELIT(depth=24, hidden_size=1024, decoder_hidden_size=1024, 
                   dit_decoder_depth=4, dit_encoder_depth=4, patch_size=2, num_heads=16, **kwargs)

def ELIT_SiT_L_4(**kwargs):
    return SiTELIT(depth=24, hidden_size=1024, decoder_hidden_size=1024, 
                   dit_decoder_depth=4, dit_encoder_depth=4, patch_size=4, num_heads=16, **kwargs)

def ELIT_SiT_L_8(**kwargs):
    return SiTELIT(depth=24, hidden_size=1024, decoder_hidden_size=1024, 
                   dit_decoder_depth=4, dit_encoder_depth=4, patch_size=8, num_heads=16, **kwargs)

def ELIT_SiT_B_2(**kwargs):
    return SiTELIT(depth=12, hidden_size=768, decoder_hidden_size=768, 
                   dit_decoder_depth=2, dit_encoder_depth=2, patch_size=2, num_heads=12, **kwargs)

def ELIT_SiT_B_4(**kwargs):
    return SiTELIT(depth=12, hidden_size=768, decoder_hidden_size=768, 
                   dit_decoder_depth=2, dit_encoder_depth=2, patch_size=4, num_heads=12, **kwargs)

def ELIT_SiT_B_8(**kwargs):
    return SiTELIT(depth=12, hidden_size=768, decoder_hidden_size=768, 
                   dit_decoder_depth=2, dit_encoder_depth=2, patch_size=8, num_heads=12, **kwargs)

def ELIT_SiT_S_2(**kwargs):
    return SiTELIT(depth=12, hidden_size=384, patch_size=2, decoder_hidden_size=384,
                   dit_decoder_depth=2, dit_encoder_depth=2, num_heads=6, **kwargs)

def ELIT_SiT_S_4(**kwargs):
    return SiTELIT(depth=12, hidden_size=384, patch_size=4, decoder_hidden_size=384, 
                   dit_decoder_depth=2, dit_encoder_depth=2, num_heads=6, **kwargs)

def ELIT_SiT_S_8(**kwargs):
    return SiTELIT(depth=12, hidden_size=384, patch_size=8, decoder_hidden_size=384, 
                   dit_decoder_depth=2, dit_encoder_depth=2, num_heads=6, **kwargs)


SiT_models = {
    'ELIT-SiT-XL/2': ELIT_SiT_XL_2,  'ELIT-SiT-XL/4': ELIT_SiT_XL_4,  'ELIT-SiT-XL/8': ELIT_SiT_XL_8,
    'ELIT-SiT-L/2':  ELIT_SiT_L_2,   'ELIT-SiT-L/4':  ELIT_SiT_L_4,   'ELIT-SiT-L/8':  ELIT_SiT_L_8,
    'ELIT-SiT-B/2':  ELIT_SiT_B_2,   'ELIT-SiT-B/4':  ELIT_SiT_B_4,   'ELIT-SiT-B/8':  ELIT_SiT_B_8,
    'ELIT-SiT-S/2':  ELIT_SiT_S_2,   'ELIT-SiT-S/4':  ELIT_SiT_S_4,   'ELIT-SiT-S/8':  ELIT_SiT_S_8,
}
