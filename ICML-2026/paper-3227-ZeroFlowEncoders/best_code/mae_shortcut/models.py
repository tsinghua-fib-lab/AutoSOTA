import torch
import torch.nn as nn
import timm
import numpy as np
from utils.shuffle import patchify

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block

class PatchShuffle(nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor): # patches: (B, n_patches, emb_dim)
        B, N, D = patches.shape
        n_keep = int(N * (1 - self.ratio))

        noise = torch.rand(B, N, device=patches.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :n_keep]
        patches_keep = torch.gather(patches, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=patches.device) # 1 = masked, 0 = unmasked
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore) # (B, n_patches)

        return patches_keep, mask, ids_keep, ids_restore

class MaskedViTEncoder(nn.Module):
    def __init__(self,
                img_size,
                patch_size,
                patch_ch=3,
                emb_dim=192,
                depth=12,
                num_heads=3,
                mask_ratio=0.75,
                global_pool=True
                ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.global_pool = global_pool
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim)) # (1, n_patches, emb_dim)
        self.shuffle = PatchShuffle(mask_ratio)        
        self.patchify = nn.Conv2d(patch_ch, emb_dim, patch_size, patch_size) # (B, 3, H, W) -> (B, emb_dim, H/ps, W/ps); patchify and tokenize
        self.transformer = nn.Sequential(*[Block(emb_dim, num_heads) for _ in range(depth)])
        self.layer_norm = nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.pos_embedding, std=.02)

    def forward_features(self,img):
        patches = self.patchify(img) # (B, emb_dim, h = H/ps, w = W/ps); h * w = n_patches; patchify and tokenize
        patches = rearrange(patches, 'b c h w -> b (h w) c') # (B, n_patches, emb_dim)

        cls_tokens = self.cls_token.expand(patches.shape[0], -1, -1)  # (B, 1, emb_dim)
        patches = torch.cat((cls_tokens, patches), dim=1)  # (B, 1 + n_patches, emb_dim)
        patches = patches + self.pos_embedding  # add pos embedding
        patches = self.transformer(patches)  # (B, n_patches, emb_dim)
        if self.global_pool:
            patches = patches[:, 1:, :].mean(dim=1)  # global pool, (B, emb_dim) without cls token; img level
            outcome = self.layer_norm(patches)
        else:
            patches = self.layer_norm(patches)
            outcome = patches[:, 0] # (B, emb_dim); cls token; img level

        return outcome 

    def forward(self, img):
        patches = self.patchify(img) # (B, emb_dim, h = H/ps, w = W/ps); h * w = n_patches
        patches = rearrange(patches, 'b c h w -> b (h w) c') # (B, n_patches, emb_dim)
        patches = patches + self.pos_embedding[:, 1:, :]  # add pos embedding

        patches_keep , mask, ids_keep, ids_restore = self.shuffle(patches)  # random masking
        cls_token = self.cls_token + self.pos_embedding[:, :1, :]  # (1, 1, emb_dim)
        cls_tokens = cls_token.expand(patches.shape[0], -1, -1)
        patches_keep = torch.cat((cls_tokens, patches_keep), dim=1)  # (B, 1 + n_keep, emb_dim)
        fea_from_keep = self.layer_norm(self.transformer(patches_keep))  # (B, 1 + n_keep, emb_dim)
        return fea_from_keep, mask, ids_keep, ids_restore  # return (B, 1 + N_keep, D), CLS included
    
class PretrainMaskedViTEncoder(nn.Module):
    def __init__(
        self,
        model_name='vit_small_patch16_224',
        mask_ratio=0.75,
        global_pool=True
    ):
        super().__init__()

        self.global_pool = global_pool
        self.shuffle = PatchShuffle(mask_ratio)

        vit = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool=''
        )

        self.patchify = vit.patch_embed.proj          # Conv2d
        self.cls_token = vit.cls_token                # nn.Parameter
        self.pos_embedding = vit.pos_embed            # nn.Parameter
        self.transformer = vit.blocks                 # nn.Sequential
        self.layer_norm = vit.norm                    # nn.LayerNorm

        self.patch_size = vit.patch_embed.patch_size[0]
        self.num_patches = vit.patch_embed.num_patches
        self.emb_dim = vit.embed_dim

    def forward_features(self, img):
        patches = self.patchify(img)                              # (B, D, H/ps, W/ps)
        patches = rearrange(patches, 'b c h w -> b (h w) c')

        cls_tokens = self.cls_token.expand(patches.size(0), -1, -1)
        patches = torch.cat((cls_tokens, patches), dim=1)
        patches = patches + self.pos_embedding

        patches = self.transformer(patches)

        if self.global_pool:
            patches = patches[:, 1:, :].mean(dim=1)
            outcome = self.layer_norm(patches)
        else:
            patches = self.layer_norm(patches)
            outcome = patches[:, 0]

        return outcome

    def forward(self, img):
        # patchify
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> b (h w) c')

        # add pos (no cls yet)
        patches = patches + self.pos_embedding[:, 1:, :]

        # mask & shuffle
        patches_keep, mask, ids_keep, ids_restore = self.shuffle(patches)

        # cls token
        cls_token = self.cls_token + self.pos_embedding[:, :1, :]
        cls_tokens = cls_token.expand(patches_keep.size(0), -1, -1)
        patches_keep = torch.cat((cls_tokens, patches_keep), dim=1)

        # transformer
        fea = self.transformer(patches_keep)
        fea = self.layer_norm(fea)

        return fea, mask, ids_keep, ids_restore

    
class DecoderMLP(nn.Module):
    def __init__(self,
                img_size = 32,
                patch_size = 4,
                emb_dim = 192,
                h_dim=256,
                n_layers=4):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        layers = []
        layers.append(nn.Linear(emb_dim, h_dim))
        layers.append(nn.GELU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(h_dim, h_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(h_dim, patch_size * patch_size * 3))
        self.mlp = nn.Sequential(*layers)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)

    # def forward(self, latent, ids_restore):
    #     """
    #     latent: (B, N_keep + 1, enc_dim)
    #     ids_restore: (B, N)
    #     """
    #     B, N_keep, D = latent.shape
    #     N_keep = N_keep - 1 # remove cls token
    #     N = ids_restore.shape[1]

    #     # 1. add mask tokens
    #     mask_tokens = self.mask_token.repeat(B, N - N_keep, 1)
    #     x_ = torch.cat([latent[:, 1:, :], mask_tokens], dim=1)

    #     # 2. restore order
    #     x_ = torch.gather(
    #         x_, 1,
    #         ids_restore.unsqueeze(-1).repeat(-1, -1, D)
    #     )
    #     x = torch.cat([latent[:, :1, :], x_], dim=1)  # add back cls token

    #     # 3. predict patch pixels independently
    #     pred_patches = self.mlp(x)  # (B, N, patch_dim); patches
    #     pred_patches = pred_patches[:, 1:, :]  # remove cls token
    #     return pred_patches
    def forward(self, latent, ids_restore):
        """
        latent: (B, 1 + N_keep, D)  # includes CLS
        ids_restore: (B, N)
        """
        latent = latent[:, 1:, :]  # drop CLS, (B, N_keep, D)

        B, N_keep, D = latent.shape
        N = ids_restore.shape[1]

        # add mask tokens
        mask_tokens = self.mask_token.repeat(B, N - N_keep, 1)
        x = torch.cat([latent, mask_tokens], dim=1)  # (B, N, D)

        # restore original patch order
        x = torch.gather(
            x, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )

        # predict pixels
        pred_patches = self.mlp(x)  # (B, N, patch_dim)
        return pred_patches
    
class DecoderTransformer(nn.Module):
    def __init__(self, img_size, patch_size, patch_ch=3, enc_dim =192,
                 dec_dim=128, depth=2, num_heads=4):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))

        self.decoder_embed = nn.Linear(enc_dim, dec_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, dec_dim)
        )

        self.blocks = nn.ModuleList([
            Block(dec_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dec_dim)
        self.head = nn.Linear(dec_dim, patch_size * patch_size * patch_ch)

        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, latent, ids_restore):
        latent = latent[:, 1:, :]  # drop CLS
        latent = self.decoder_embed(latent)

        B, N_keep, D = latent.shape
        N = ids_restore.shape[1]

        mask_tokens = self.mask_token.repeat(B, N - N_keep, 1)
        x = torch.cat([latent, mask_tokens], dim=1)
        x = torch.gather(
            x, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )

        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.head(self.norm(x))

        return x # (B, N, patch_dim)

                 
    
class MAEModel(nn.Module):
    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 patch_ch=3,
                 enc_dim=192,
                 enc_depth=12,
                 enc_heads=3,
                 enc_pretrained=False,
                 mask_ratio=0.75,
                 dec_hidden_dim=128,
                 dec_layers=2,
                 ) -> None:
        super().__init__()

        if enc_pretrained:
            self.encoder = PretrainMaskedViTEncoder(
                model_name='vit_small_patch16_224',
                mask_ratio=mask_ratio,
                global_pool=True
            )
        else:

            self.encoder = MaskedViTEncoder(
                img_size=img_size,
                patch_size=patch_size,
                patch_ch=patch_ch,
                emb_dim=enc_dim,
                depth=enc_depth,
                num_heads=enc_heads,
                mask_ratio=mask_ratio,
            )

        self.decoder = DecoderTransformer(
            img_size=img_size,
            patch_size=patch_size,
            patch_ch=patch_ch,
            enc_dim=enc_dim,
            dec_dim=dec_hidden_dim,
            depth=dec_layers,
        )

    def forward(self, img):
        latent, mask, ids_keep, ids_restore = self.encoder(img)
        pred_patches = self.decoder(latent, ids_restore)
        return pred_patches, mask
    

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of shape [batch_size, 1]
        Returns:
            Tensor of shape [batch_size, embed_dim]
        """
        t = t.view(-1, 1)
        projection = t * self.W[None,:] * 2 * torch.pi  # [B, embed_dim // 2]
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)  # [B, embed_dim]
    

class VectorField(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        enc_dim,
        h_dim,
        n_layers=2,
        t_emb_dim=32
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(t_emb_dim),
            nn.Linear(t_emb_dim, t_emb_dim)
        )
        self.patch_dim = 3 * patch_size * patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_emb = nn.Embedding(self.num_patches, h_dim)
        self.ctx_y = nn.Linear(self.patch_dim, h_dim)
        self.patch_size = patch_size
        self.ctx_fy = nn.Linear(enc_dim, h_dim) # cuurently we discard cls token in vf training
        self.in_x = nn.Linear(self.patch_dim, h_dim)
        self.ctx_t = nn.Linear(t_emb_dim, h_dim)

        layers = []
        layers.append(nn.SiLU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(h_dim, h_dim)) # linear only acts on the last dim
            layers.append(nn.SiLU())
        layers.append(nn.Linear(h_dim, self.patch_dim))
        self.net = nn.Sequential(*layers)

    # def forward(self, xt, y, enc_y, t):
    #     # xt: (B, N_mask, patch_dim)
    #     # y: (B, N_keep, patch_dim)
    #     # enc_y: (B, N_keep + 1, enc_dim)
    #     # t: (B,1)
    #     y_ctx   = self.ctx_y(y).mean(dim=1)        # (B, H) avg for alignment as n_keep =/= n_mask
    #     enc_ctx = self.ctx_fy(enc_y[:, 1:, :]).mean(dim=1)   # (B, H) same
    #     t_ctx   = self.ctx_t(self.time_embed(t))   # (B, H)
    #     h = self.in_x(xt)                          # (B, N_mask, H)
    #     h = h + y_ctx[:,None,:] + enc_ctx[:,None,:] + t_ctx[:,None,:]
    #     dxdt_pred = self.net(h)                    # (B, N_mask, patch_dim)
    #     return dxdt_pred
    def forward(self, xt, y, enc_y, t, ids_masked):
        """
        xt         : (B, N_mask, patch_dim)
        y          : (B, N_keep, patch_dim)
        enc_y      : (B, N_keep + 1, enc_dim)
        t          : (B,)
        ids_masked : (B, N_mask)   <-- ABSOLUTELY REQUIRED
        """

        # ---- context terms (global) ----
        y_ctx   = self.ctx_y(y).mean(dim=1)                 # (B, H)
        enc_ctx = self.ctx_fy(enc_y[:, 1:, :]).mean(dim=1)  # (B, H)
        t_ctx   = self.ctx_t(self.time_embed(t))            # (B, H)

        # ---- token flow ----
        h = self.in_x(xt)                                   # (B, N_mask, H)

        # ---- ADD POSITIONAL INFORMATION ----
        pos = self.pos_emb(ids_masked)                      # (B, N_mask, H)
        h = h + pos

        # ---- inject conditioning (additive) ----
        h = h + y_ctx[:, None, :] + enc_ctx[:, None, :] + t_ctx[:, None, :]

        dxdt = self.net(h)                                  # (B, N_mask, patch_dim)
        return dxdt
    
class AdaLN(nn.Module):
    def __init__(self, h_dim = 128, cond_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(h_dim,elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, h_dim * 2)  # output gamma and beta
        )

    def forward(self, x, cond):
        # x: (B, N, D)
        # cond: (B, cond_dim)
        gamma, beta = self.mlp(cond).chunk(2, dim=-1)  # each (B, D)
        gamma = gamma[:, None, :]  # (B, 1, D)
        beta = beta[:, None, :]    # (B, 1, D)
        return self.norm(x) * (1 + gamma) + beta
    
class SiTBlock(nn.Module):
    def __init__(
            self,
            h_dim,
            cond_dim,
            num_heads=4
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=h_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim * 4),
            nn.SiLU(),
            nn.Linear(h_dim * 4, h_dim)
        )
        self.adaln1 = AdaLN(h_dim, cond_dim)
        self.adaln2 = AdaLN(h_dim, cond_dim)

    def forward(self, x, cond):
        h = self.adaln1(x, cond)
        h_attn, _ = self.attn(h, h, h)
        x = x + h_attn
        x = x + self.mlp(self.adaln2(x, cond))
        return x
    
class SiTVectorField(nn.Module):
    def __init__(
        self,
        patch_size,
        enc_dim,
        t_emb_dim=128,
        h_dim = 128,
        cond_dim=64,
        depth=2,
        num_heads=4
    ):
        super().__init__()
        self.patch_dim = 3 * patch_size * patch_size
        self.patch_size = patch_size
        # ---- input projection (x_t only) ----
        self.in_proj = nn.Linear(self.patch_dim, h_dim)
        # ---- condition encoders ----
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(t_emb_dim),
            nn.Linear(t_emb_dim, cond_dim),
        )
        self.y_embed = nn.Linear(self.patch_dim, cond_dim)
        self.fy_embed = nn.Linear(enc_dim, cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )
        
        # ---- SiT blocks ----
        self.blocks = nn.ModuleList([
            SiTBlock(h_dim, cond_dim, num_heads)
            for _ in range(depth)
        ])
        # ---- output projection ----
        self.out_proj = nn.Linear(h_dim, self.patch_dim)

    def forward(self, xt, y, enc_y, t):
        # xt: (B, N_mask, patch_dim)
        # y: (B, N_keep, patch_dim)
        # enc_y: (B, N_keep + 1, enc_dim)
        # t: (B,1)
        
        t_cond = self.time_embed(t)               # (B, cond_dim)
        y_cond = self.y_embed(y).mean(dim=1)      # (B, cond_dim)
        fy_cond = self.fy_embed(enc_y[:, 1:, :]).mean(dim=1)  # (B, cond_dim) drop cls token
        cond = t_cond + y_cond + fy_cond          # (B, cond_dim)
        cond = self.cond_proj(cond)

        # -------- token flow --------
        h = self.in_proj(xt)                      # (B, N_mask, h_dim)
        for blk in self.blocks:
            h = blk(h, cond)
        dxdt_pred = self.out_proj(h)              # (B, N_mask, patch_dim)
        return dxdt_pred
        
    
class EncoderZFE(MaskedViTEncoder):
    """
    Encoder with explicit mask control for Zero-Flow / Masked Recovery
    Inherits all structures from Encoder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def sample_mask(self, B, device):
        """
        Sample a random mask ONCE and return indices.
        """
        N = self.num_patches
        D = self.pos_embedding.shape[2]

        # dummy tensor just to reuse PatchShuffle
        dummy = torch.empty(B, N, D, device=device)
        _, mask, ids_keep, ids_restore = self.shuffle(dummy)

        return mask, ids_keep, ids_restore

    def forward_with_ids(self, img, ids_keep):
        """
        Encode image under a GIVEN mask (ids_keep).
        This does NOT resample mask.
        """
        patches = self.patchify(img)                          # (B, C, h, w)
        patches = rearrange(patches, 'b c h w -> b (h w) c')  # (B, N, D)
        patches = patches + self.pos_embedding[:, 1:, :]       # add patch pos embed

        B, _, D = patches.shape
        patches_keep = torch.gather(
            patches,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )                                                      # (B, N_keep, D)
        cls_token = self.cls_token + self.pos_embedding[:, :1, :]
        cls_tokens = cls_token.expand(patches.shape[0], -1, -1)
        patches_keep = torch.cat((cls_tokens, patches_keep), dim=1)      # (B, 1 + N_keep, D)
        features = self.transformer(patches_keep)
        features = self.layer_norm(features)
        return features                                       # (B, N_keep + 1, D)

    
class PretrainEncoderZFE(PretrainMaskedViTEncoder):
    """
    Pretrained Encoder with explicit mask control for Zero-Flow / Masked Recovery
    Inherits all structures from PretrainEncoder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def sample_mask(self, B, device):
        """
        Sample a random mask ONCE and return indices.
        """
        N = self.num_patches
        D = self.pos_embedding.shape[2]

        # dummy tensor just to reuse PatchShuffle
        dummy = torch.empty(B, N, D, device=device)
        _, mask, ids_keep, ids_restore = self.shuffle(dummy)

        return mask, ids_keep, ids_restore

    def forward_with_ids(self, img, ids_keep):
        """
        Encode image under a GIVEN mask (ids_keep).
        This does NOT resample mask.
        """
        patches = self.patchify(img)                          # (B, C, h, w)
        patches = rearrange(patches, 'b c h w -> b (h w) c')  # (B, N, D)
        patches = patches + self.pos_embedding[:, 1:, :]       # add patch pos embed

        B, _, D = patches.shape
        patches_keep = torch.gather(
            patches,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )                                                      # (B, N_keep, D)
        cls_token = self.cls_token + self.pos_embedding[:, :1, :]
        cls_tokens = cls_token.expand(patches.shape[0], -1, -1)
        patches_keep = torch.cat((cls_tokens, patches_keep), dim=1)      # (B, 1 + N_keep, D)
        features = self.transformer(patches_keep)
        features = self.layer_norm(features)
        return features                                       # (B, N_keep + 1, D)    
    
class ZFEModel(nn.Module):
    def __init__(self,
                 img_size=32,
                 patch_size=16,
                 enc_dim=192,
                 enc_depth=12,
                 enc_heads=3,
                 enc_pretrained=False,
                 mask_ratio=0.75,
                 v_h_dim=128,
                 v_cond_dim=64,
                 v_n_layers=2,
                 ) -> None:
        super().__init__()
        if enc_pretrained:
            self.encoder = PretrainEncoderZFE(
                model_name='vit_small_patch16_224',
                mask_ratio=mask_ratio,
                global_pool=False
            )
        else:
            self.encoder = EncoderZFE(
                img_size=img_size,
                patch_size=patch_size,
                emb_dim=enc_dim,
                depth=enc_depth,
                num_heads=enc_heads,
                mask_ratio=mask_ratio,
            )

        # self.vector_field = VectorField(
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     enc_dim=enc_dim,
        #     h_dim=v_h_dim,
        #     n_layers=v_n_layers,
        # )
        self.vector_field = SiTVectorField(
            patch_size=patch_size,
            enc_dim=enc_dim,
            h_dim=v_h_dim,
            cond_dim=v_cond_dim,
            depth=v_n_layers,
            num_heads=4,
        )

# class Classifier(nn.Module):
#     def __init__(self,
#                  encoder: nn.Module,
#                  num_classes: int
#                  ):
#         super().__init__()
#         self.encoder = encoder
#         self.classifier = nn.Linear(encoder.emb_dim, num_classes)

#     def freeze_encoder(self):
#         for param in self.encoder.parameters():
#             param.requires_grad = False

#     def unfreeze_encoder(self):
#         for param in self.encoder.parameters():
#             param.requires_grad = True

#     def forward(self, x):
#         features = self.encoder.forward_features(x)  # (B, emb_dim)
#         logits = self.classifier(features)         # (B, num_classes)
#         return logits

if __name__ == "__main__":
    model = MaskedViTEncoder(img_size=32, patch_size=4, enc_dim=192)
    print(model)