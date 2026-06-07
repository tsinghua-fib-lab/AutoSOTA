from typing import Union
import clip
import clip.model
import open_clip.transformer
import timm
import torch
import torch.nn as nn
import transformers
from open_clip.transformer import _expand_token, text_global_pool
from SLIP.models import SLIP
from copy import deepcopy 
import torch.nn.functional as F
## Image Encoder
import numpy as np 
import math 
import torch.nn.functional as F
import torch 
from typing import Final, Optional, Type
import sys 

def get_encode_image_with_noproj(clip_model: Union[clip.model.CLIP, SLIP, open_clip.model.CLIP]) -> callable:
    if isinstance(clip_model, clip.model.CLIP):
        return encode_image_noproj_openai # OpenAI CLIP B/32, B/16, L/14  
    elif isinstance(clip_model, SLIP):
        return  encode_image_noproj_slip # SLIP model 
    elif isinstance(clip_model.visual, open_clip.transformer.VisionTransformer):
        return encode_image_noproj_openclip # OpenCLIP ViT-B/32, ViT-B/16, ViT-L/14 
    elif isinstance(clip_model.visual, open_clip.timm_model.TimmModel):
        return encode_image_noproj_timmvit # Timm ViT models (including EVA and SigLIP and Perception Encoder) 
    else:
        sys.exit("Not implemented")
 

@torch.no_grad()
def encode_image_noproj_openai(clip_model, images):
    x = clip_model.visual.conv1(images)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + clip_model.visual.positional_embedding.to(x.dtype)
    x = clip_model.visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.visual.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.visual.ln_post(x[:, 0, :]) # removed projection
    return x 

@torch.no_grad()
def encode_image_noproj_slip(clip_model, images):
    x = clip_model.visual(images)
    return x 

@torch.no_grad()
def encode_image_noproj_openclip(clip_model, images):
    x = clip_model.visual._embeds(images)
    x = clip_model.visual.transformer(x)
    pooled, tokens = clip_model.visual._pool(x)
    return pooled 


def encode_image_noproj_timmvit(clip_model, images):  
    
    # Check if model has BOTH attn_pool AND Linear head (like Perception Encoder)
    if (hasattr(clip_model.visual.trunk, 'attn_pool') and clip_model.visual.trunk.attn_pool is not None and
        hasattr(clip_model.visual.trunk, 'head') and isinstance(clip_model.visual.trunk.head, torch.nn.Linear)):
        # Manually reconstruct forward: forward_features -> attn_pool -> STOP (no head)
        x = clip_model.visual.trunk.forward_features(images)  # [batch, num_tokens, embed_dim]
        x = clip_model.visual.trunk.attn_pool(x)  # Apply overridden attn_pool (MLP skipped in utils.py)
        return x
    
       # Check if model has attn_pool only (no Linear head, e.g., SigLIP with Identity head)
    elif hasattr(clip_model.visual.trunk, 'attn_pool') and clip_model.visual.trunk.attn_pool is not None:
        # For models with attention pooling (attn_pool.forward has been overridden in utils.py)
        # This will use the overridden forward that skips the MLP
        x = clip_model.visual.trunk(images)
        return x
    
        # Check if model has simple Linear head only (no attn_pool)
    elif hasattr(clip_model.visual.trunk, 'head') and isinstance(clip_model.visual.trunk.head, torch.nn.Linear):
        # Use forward_features to get features before the head projection
        x = clip_model.visual.trunk.forward_features(images)  # [batch, num_tokens, embed_dim]

        # Need to pool the tokens to get a single feature vector
        # Check if there's a global pooling method
        if hasattr(clip_model.visual.trunk, 'global_pool') and clip_model.visual.trunk.global_pool != '':
            # Use the model's pooling strategy
            if clip_model.visual.trunk.global_pool == 'avg':
                x = x.mean(dim=1)  # Average pool across tokens
            elif clip_model.visual.trunk.global_pool == 'token':
                x = x[:, 0]  # Use first token (class token)
            else:
                # Default to using first token
                x = x[:, 0]
        else:
            # Default to using first token
            x = x[:, 0]

        return x
    else:
        # Fallback: use full trunk forward
        x = clip_model.visual.trunk(images)
        return x 
    
 

## Text Encoder


def get_encode_text_with_noproj(clip_model: Union[clip.model.CLIP, SLIP, open_clip.model.CLIP]) -> callable:
    if isinstance(clip_model, clip.model.CLIP) or isinstance(clip_model, SLIP):
        return encode_text_noproj_openaislip
    elif isinstance(clip_model.visual, open_clip.transformer.VisionTransformer):
        return encode_text_noproj_openclip
    elif isinstance(clip_model.visual, open_clip.timm_model.TimmModel):
        sys.exit("Not implemented text forward for SigLIP/PE models")
   




@torch.no_grad()
def encode_text_noproj_openaislip(clip_model, text):
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] # removed projection

    return x

@torch.no_grad()
def encode_text_noproj_openclip(clip_model, text):
    cast_dtype = clip_model.transformer.get_cast_dtype()

    x = clip_model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.to(cast_dtype)
    x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)
    x = clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
    x = text_global_pool(x, text, clip_model.text_pool_type)
   

    return x

@torch.no_grad()
def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    return scores if attn_mask is None else scores + attn_mask

@torch.no_grad() # For Perception Encoder and SigLIP
def encode_attention_module(attention_module, x, attn_mask=None):

    B, N, C = x.shape

    if attention_module.pos_embed is not None:
        # FIXME interpolate
        x = x + attention_module.pos_embed.unsqueeze(0).to(x.dtype)

    q_latent = attention_module.latent.expand(B, -1, -1)
    q = attention_module.q(q_latent).reshape(B, attention_module.latent_len, attention_module.num_heads, attention_module.head_dim).transpose(1, 2)

    kv = attention_module.kv(x).reshape(B, N, 2, attention_module.num_heads, attention_module.head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)

    q, k = attention_module.q_norm(q), attention_module.k_norm(k)

    if attention_module.fused_attn:
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    else:
        q = q * attention_module.scale
        attn = q @ k.transpose(-2, -1)
        attn = maybe_add_mask(attn, attn_mask)
        attn = attn.softmax(dim=-1)
        x = attn @ v
    x = x.transpose(1, 2).reshape(B, attention_module.latent_len, C)
    x = attention_module.proj(x)
    x = attention_module.proj_drop(x)

    # x = x + self.mlp(self.norm(x))

    # optional pool if latent seq_len > 1 and pooled output is desired
    if attention_module.pool == 'token':
        x = x[:, 0]
    elif attention_module.pool == 'avg':
        x = x.mean(1)
    return x
 
 
 

def gelu_prime(z):
    return 0.5*(1+torch.erf(z/torch.sqrt(torch.tensor(2.0)))) \
           + (z*torch.exp(-z**2/2))/(np.sqrt(2*torch.pi))
           
def get_projection_layers(clip_model: Union[clip.model.CLIP, SLIP, open_clip.model.CLIP], clip_model_name):
    
    if "EVA" in clip_model_name :
        W_image = clip_model.visual.trunk.head.weight
        W_text = clip_model.text.text_projection.data
        W_image = W_image.T 
        W_text = W_text.T 
        return W_image, W_text 
    
    if isinstance(clip_model, clip.model.CLIP):
        return clip_model.visual.proj, clip_model.text_projection.data # No bias 
    elif isinstance(clip_model, SLIP):
        return clip_model.image_projection, clip_model.text_projection.data # No bias 
    elif isinstance(clip_model.visual, open_clip.transformer.VisionTransformer):
        return clip_model.visual.proj, clip_model.text_projection.data # No bias in B/32 - L14
    
    elif isinstance(clip_model.visual, open_clip.timm_model.TimmModel):
        # Perception Encoder and SigLIP-V2
        
        if hasattr(clip_model.visual.trunk, 'head') and isinstance(clip_model.visual.trunk.head, torch.nn.Linear):
            # PE Case with Linear head
            W_img = clip_model.visual.trunk.head.weight  
            W_img = W_img.T
 
            # Get text projection
            if hasattr(clip_model, 'text') and hasattr(clip_model.text, 'text_projection'):
                # CustomTextCLIP case
                if isinstance(clip_model.text.text_projection, torch.nn.Parameter):
                    # Parameter matrix (no bias)
                    W_text = clip_model.text.text_projection.data  # [text_dim, out_dim]
                    # The bias is zero in the perception encoder 
                elif isinstance(clip_model.text.text_projection, torch.nn.Linear):
                    # Linear layer with bias
                    # if some models have bias in text projection, we can concatenate it as an extra column to the weight matrix, 
                    # and add a dummy 1 to the input features during encoding.
                    W_text = clip_model.text.text_projection.weight
                    b_text = clip_model.text.text_projection.bias
                    W_text = torch.cat([W_text, b_text.unsqueeze(1)], dim=1)  
            else:
                # CLIP case
                W_text = clip_model.text_projection.data
                W_text = torch.cat([W_text, torch.zeros(W_text.size(0), 1, device=W_text.device)], dim=1)

            return W_img, W_text

 
        else:
            # SigLIP Case with attn_pool MLP
            W1_img = clip_model.visual.trunk.attn_pool.mlp.fc1.weight  # [3072×768]
            b1_img = clip_model.visual.trunk.attn_pool.mlp.fc1.bias     # [3072]
            W2_img = clip_model.visual.trunk.attn_pool.mlp.fc2.weight   # [768×3072]
            b2_img = clip_model.visual.trunk.attn_pool.mlp.fc2.bias     # [768]
        
            ln = clip_model.visual.trunk.attn_pool.norm
            gamma = ln.weight        # [768]
            beta  = ln.bias          # [768]
            W_1 = W1_img 
            b_1 = b1_img 
            
            W1_tilde = W_1 * gamma.unsqueeze(0)   
            b1_tilde = W_1 @ beta + b_1              # [3072]           # [3072]
            
            I = torch.eye(W1_img.size(1), device=W1_img.device)           # [768×768]
            W_eff_img = I + 0.5*(W2_img @ W1_tilde)          #gelu_slope *          # [768×768]
            b_eff_img = 0.5 *(W2_img @ b1_tilde) + b2_img                # [768]
            
    
            W_eff_text = clip_model.text.text_projection.weight  # [768×d_t]
            b_eff_text = clip_model.text.text_projection.bias     # [768]
            W_img = torch.cat([W_eff_img, b_eff_img.unsqueeze(0)], dim=0)  # [768×(768+1)]
            W_text = torch.cat([W_eff_text, b_eff_text.unsqueeze(0)], dim=0)  # [768×(768+1)]
            
    
            
            return W_img,  W_text

 
        
        