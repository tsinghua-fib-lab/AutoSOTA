# current implementation: only support numerical values

from functools import partial
from tkinter import E

import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from timm.models.vision_transformer import Block
from .utils import MaskEmbed, get_1d_sincos_pos_embed, ActiveEmbed
eps = 1e-6

class MaskedAutoencoder(nn.Module):
    
    """ Masked Autoencoder with Transformer backbone
    """
    
    def __init__(self, rec_len=25, embed_dim=64, depth=4, num_heads=4,
        decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=nn.LayerNorm, norm_field_loss=False, encode_func='linear'):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        
        if encode_func == 'active':
            self.mask_embed = ActiveEmbed(rec_len, embed_dim)
        else:
            self.mask_embed = MaskEmbed(rec_len, embed_dim)
        
        self.rec_len = rec_len
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, rec_len + 1, embed_dim), requires_grad=False)  
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)


        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, rec_len + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 1, bias=True)  # decoder to patch
        
        # --------------------------------------------------------------------------

        self.norm_field_loss = norm_field_loss
        self.initialize_weights()


    def initialize_weights(self):
        
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.mask_embed.rec_len, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.mask_embed.rec_len, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.mask_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def random_masking(self, x, m, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if self.training:
            len_keep = int(L * (1 - mask_ratio))
        else:
            len_keep = int(torch.min(torch.sum(m, dim=1)))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        noise[m < eps] = 1

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        nask = torch.ones([N, L], device=x.device) - mask

        if self.training:
            mask[m < eps] = 0

        return x_masked, mask, nask, ids_restore


    def forward_encoder(self, x, m, mask_ratio=0.5):
        
        # embed patches
        x = self.mask_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]    

        # masking: length -> length * mask_ratio
        x, mask, nask, ids_restore = self.random_masking(x, m, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, mask, nask, ids_restore


    def forward_decoder(self, x, ids_restore):
        
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        # x = self.decoder_pred(x)
        x = torch.tanh(self.decoder_pred(x))/2 + 0.5

        # remove cls token
        x = x[:, 1:, :]
    
        return x


    def forward_loss(self, data, pred, mask, nask):
        """
        data: [N, 1, L]
        pred: [N, L]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # target = self.patchify(data)
        target = data.squeeze(dim=1)
        if self.norm_field_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + eps)**.5
        
        
        loss = (pred.squeeze(dim=2) - target) ** 2
        loss = (loss * mask).sum() / mask.sum()  + (loss * nask).sum() / nask.sum()
        # mean loss on removed patches
        
        return loss


    def forward(self, data, miss_idx, mask_ratio=0.5):
        
        latent, mask, nask, ids_restore = self.forward_encoder(data, miss_idx, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore) 
        loss = self.forward_loss(data, pred, mask, nask)
        return loss, pred, mask, nask


def mae_base(**kwargs):
    model = MaskedAutoencoder(
        embed_dim=64, depth=8, num_heads=4,
        decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=2., norm_layer=partial(nn.LayerNorm, eps=eps), **kwargs)
    return model


def mae_medium(**kwargs):
    model = MaskedAutoencoder(
        embed_dim=32, depth=4, num_heads=4,
        decoder_embed_dim=32, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=eps), **kwargs)
    return model


def mae_large(**kwargs):
    model = MaskedAutoencoder(
        embed_dim=64, depth=8, num_heads=4,
        decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=eps), **kwargs)
    return model


if __name__ == '__main__':

    model = MaskedAutoencoder(
        rec_len=4, embed_dim=8, depth=1, num_heads=1,
        decoder_embed_dim=8, decoder_depth=1, decoder_num_heads=1,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=eps)
    )
    
    X = pd.DataFrame([[np.nan, 0.5, np.nan, 0.8]])

    X = torch.tensor(X.values, dtype=torch.float32)
    M = 1 - (1 * (np.isnan(X)))
    X = torch.nan_to_num(X)
    
    X = X.unsqueeze(dim=1)
    print(model.forward(X, M, 0.75))


    
