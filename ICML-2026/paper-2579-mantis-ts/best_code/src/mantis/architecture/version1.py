import torch

from torch import nn
from einops import repeat, pack, unpack
from huggingface_hub import PyTorchModelHubMixin

from .tokgen_utils.convolution import Convolution
from .tokgen_utils.encoders import MultiScaledScalarEncoder, LinearEncoder
from .transformer_v1_utils.positional_encoding import PositionalEncoding
from .transformer_v1_utils.transformer import Transformer


# ==================================
# ====       Organization:      ====
# ==================================
# ==== class TokenGeneratorUnit ====
# ==== class TransformerUnit    ====
# ==== class ViTUnit (legacy)   ====
# ==== class MantisV1           ====
# ==== class Mantis8M (legacy)  ====
# ==================================


class TokenGeneratorUnit(nn.Module):
    def __init__(self, hidden_dim, num_patches, patch_window_size, scalar_scales, hidden_dim_scalar_enc,
                 epsilon_scalar_enc):
        super().__init__()
        self.num_patches = num_patches
        # token generator for time series objects
        num_ts_feats = 2  # original ts + its diff
        kernel_size = patch_window_size + \
            1 if patch_window_size % 2 == 0 else patch_window_size
        self.convs = nn.ModuleList([
            Convolution(kernel_size=kernel_size,
                        out_channels=hidden_dim, dilation=1)
            for i in range(num_ts_feats)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-5)
            for i in range(num_ts_feats)
        ])

        # token generator for scalar statistics
        if scalar_scales is None:
            scalar_scales = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        num_scalar_stats = 2  # mean + std
        self.scalar_encoders = nn.ModuleList([
            MultiScaledScalarEncoder(
                scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc)
            for i in range(num_scalar_stats)
        ])

        # final token projector
        self.linear_encoder = LinearEncoder(
            hidden_dim_scalar_enc * num_scalar_stats + hidden_dim * (num_ts_feats), hidden_dim)

        # scales each time-series w.r.t. its mean and std
        self.ts_scaler = lambda x: (
            x - torch.mean(x, axis=2, keepdim=True)) / (torch.std(x, axis=2, keepdim=True) + 1e-5)

    def forward(self, x):
        with torch.no_grad():
            # compute statistics for each patch
            x_patched = x.reshape(x.shape[0], self.num_patches, -1)
            mean_patched = torch.mean(x_patched, axis=-1, keepdim=True)
            std_patched = torch.std(x_patched, axis=-1, keepdim=True)
            statistics = [mean_patched, std_patched]

        # for each encoder output is (batch_size, num_sub_ts, hidden_dim_scalar_enc)
        scalar_embeddings = [self.scalar_encoders[i](
            statistics[i]) for i in range(len(statistics))]

        # apply convolution for original ts and its diff
        ts_var_embeddings = []
        # diff
        with torch.no_grad():
            diff_x = torch.diff(x, n=1, axis=2)
            # pad by zeros to have same dimensionality as x
            diff_x = torch.nn.functional.pad(diff_x, (0, 1))
        # dim(bs, hidden_dim, len_ts-patch_window_size-1)
        embedding = self.convs[0](self.ts_scaler(diff_x))
        ts_var_embeddings.append(embedding)
        
        # original ts
        # dim(bs, hidden_dim, len_ts-patch_window_size-1)
        embedding = self.convs[1](self.ts_scaler(x))
        ts_var_embeddings.append(embedding)

        # split ts_var_embeddings into patches
        patched_ts_var_embeddings = []
        for i, embedding in enumerate(ts_var_embeddings):
            embedding = self.layer_norms[i](embedding)
            embedding = embedding.reshape(
                embedding.shape[0], self.num_patches, -1, embedding.shape[2])
            embedding = torch.mean(embedding, dim=2)
            patched_ts_var_embeddings.append(embedding)

        # concatenate diff_x, x, mu and std embeddinga and send them to the linear projector
        x_embeddings = torch.cat([
            torch.cat(patched_ts_var_embeddings, dim=-1),
            torch.cat(scalar_embeddings, dim=-1)
        ], dim=-1)
        x_embeddings = self.linear_encoder(x_embeddings)

        return x_embeddings


class TransformerUnit(nn.Module):
    def __init__(self, hidden_dim, num_patches, depth, heads, mlp_dim, dim_head, dropout, device):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim, dropout=dropout, max_len=num_patches+1)
        self.cls_token = nn.Parameter(torch.randn(hidden_dim).to(device))
        self.transformer = Transformer(
            hidden_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x, output_token='cls_token', return_layer=-1):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x_embeddings, ps = pack([cls_tokens, x], 'b * d')
        x_embeddings = self.pos_encoder(
            x_embeddings.transpose(0, 1)).transpose(0, 1)
        x_embeddings = self.transformer(x_embeddings, return_layer=return_layer)
        cls_tokens, other = unpack(x_embeddings, ps, 'b * d')
        cls_token = cls_tokens.reshape(cls_tokens.shape[0], -1)
        if output_token == 'cls_token':
            return cls_token
        mean_token = other.mean(dim=1)
        if output_token == 'mean_token':
            return mean_token
        if output_token == 'combined':
            return torch.cat([cls_token, mean_token], dim=1)
        raise KeyError("Unknown output type")


class ViTUnit(nn.Module):
    """Legacy class name for TransformerUnit. Maintains backward compatibility."""
    def __init__(self, hidden_dim, num_patches, depth, heads, mlp_dim, dim_head, dropout, device):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim, dropout=dropout, max_len=num_patches+1)
        self.cls_token = nn.Parameter(torch.randn(hidden_dim).to(device))
        self.transformer = Transformer(
            hidden_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x_embeddings, ps = pack([cls_tokens, x], 'b * d')
        x_embeddings = self.pos_encoder(
            x_embeddings.transpose(0, 1)).transpose(0, 1)
        x_embeddings = self.transformer(x_embeddings)
        cls_tokens, _ = unpack(x_embeddings, ps, 'b * d')
        return cls_tokens.reshape(cls_tokens.shape[0], -1)


def rename_vit_unit_weights_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    # Iterate over a copy of the keys to avoid modification errors
    for key in list(state_dict.keys()):
        # Check if the key corresponds to the old 'vit_unit'
        # 'prefix' handles cases where this model is a submodule of a larger model
        if key.startswith(prefix + 'vit_unit.'):
            # Create the new key name
            new_key = key.replace(prefix + 'vit_unit.', prefix + 'transf_unit.')
            
            # Move the data to the new key and delete the old one
            state_dict[new_key] = state_dict.pop(key)


class MantisV1(
    nn.Module,
    PyTorchModelHubMixin,
    # optionally, you can add metadata which gets pushed to the model card
    library_name="mantis",
    repo_url="https://huggingface.co/paris-noah/Mantis-8M/tree/main",
    pipeline_tag="time-series-foundation-model",
    license="mit",
    tags=["time-series-foundation-model"],
):
    """
    The architecture of Mantis time series foundation model.

    Parameters
    ----------
    seq_len: int, default 512
        The sequence length, i.e., the length of each time series. This model does not support data with non-fixed
        sequence length, please make all the time series to be of a fixed length by resizing or padding.
    hidden_dim: int, default=256
        Size of a patch (token), i.e., what the hidden dimension each patch is projected to. At the same time,
        ``hidden_dim`` corresponds to the dimension of the embedding space.
    num_patches: int, default=32
        Number of patches (tokens).
    scalar_scales: list, default=None
        List of scales used for MultiScaledScalarEncoder in TokenGeneratorUnit. By default, initialized as [1e-4, 1e-3,
        1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4].
    hidden_dim_scalar_enc: int, default=32
        Hidden dimension of a scalar encoder used for MultiScaledScalarEncoder in TokenGeneratorUnit.
    epsilon_scalar_enc: float, default=1.1
        A constant term used to tolerate the computational error in computation of scale weights for
        MultiScaledScalarEncoder in TokenGeneratorUnit.
    transf_depth: int, default=6
        Number of transformer layers used for Transformer in TransformerUnit.
    transf_num_heads: int, default=8
        Number of self-attention heads used for Transformer in TransformerUnit.
    transf_mlp_dim: int, default=512
        Hidden dimension of the MLP (feed-forward) transformer's part used for Transformer in TransformerUnit.
    transf_dim_head: int, default=128
        Hidden dimension of the keys, queries and values used for Transformer in TransformerUnit.
    transf_dropout: froat, default=0.1
        Dropout value used for Transformer in TransformerUnit.
    device: {'cpu', 'cuda'}, default='cuda'
        On which device the model is located.
    pre_training: bool, default=False
        If True, applies an MLP projector after the TransformerUnit, which originally was used to pre-train the model using
        InfoNCE contrastive loss.
    """

    def __init__(self, seq_len=512, hidden_dim=256, num_patches=32, scalar_scales=None, hidden_dim_scalar_enc=32,
                 epsilon_scalar_enc=1.1, transf_depth=6, transf_num_heads=8, transf_mlp_dim=512, transf_dim_head=128,
                 transf_dropout=0.1, device='cuda', pre_training=False, return_transf_layer=-1, output_token='cls_token'):

        super().__init__()
        assert (seq_len % num_patches) == 0, print(
            'Seq_len must be the multiple of num_patches')
        patch_window_size = int(seq_len / num_patches)

        self.hidden_dim = 2 * hidden_dim if output_token == 'combined' else hidden_dim
        self.num_patches = num_patches
        self.scalar_scales = scalar_scales
        self.hidden_dim_scalar_enc = hidden_dim_scalar_enc
        self.epsilon_scalar_enc = epsilon_scalar_enc
        self.seq_len = seq_len
        self.pre_training = pre_training
        self.return_transf_layer = return_transf_layer
        self.output_token = output_token

        # Hook
        self._register_load_state_dict_pre_hook(rename_vit_unit_weights_hook)

        self.tokgen_unit = TokenGeneratorUnit(hidden_dim=hidden_dim,
                                              num_patches=num_patches,
                                              patch_window_size=patch_window_size,
                                              scalar_scales=scalar_scales,
                                              hidden_dim_scalar_enc=hidden_dim_scalar_enc,
                                              epsilon_scalar_enc=epsilon_scalar_enc)
        self.transf_unit = TransformerUnit(hidden_dim=hidden_dim, num_patches=num_patches, depth=transf_depth,
                           heads=transf_num_heads, mlp_dim=transf_mlp_dim, dim_head=transf_dim_head,
                           dropout=transf_dropout, device=device)

        self.prj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.to(device)

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, x):
        x_embeddings = self.tokgen_unit(x)
        transf_out = self.transf_unit(x_embeddings, output_token=self.output_token,
                                      return_layer=self.return_transf_layer)
        if self.pre_training:
            return self.prj(transf_out)
        else:
            return transf_out
        
    def from_pretrained(self, *args, **kwargs):
        network = super().from_pretrained(*args, **kwargs)
        network.return_transf_layer = self.return_transf_layer
        network.output_token = self.output_token
        network.pre_training = self.pre_training
        network.hidden_dim = self.hidden_dim
        network.device = self.device
        return network.to(self.device)


class Mantis8M(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="mantis",
    repo_url="https://huggingface.co/paris-noah/Mantis-8M/tree/main",
    pipeline_tag="time-series-foundation-model",
    license="mit",
    tags=["time-series-foundation-model"],
):
    """
    Legacy class name for MantisV1. Maintains exact backward compatibility
    with old code expecting Mantis8M and ViTUnit module names.
    
    This class is functionally identical to MantisV1 but uses the old naming
    convention for internal modules (vit_unit instead of transf_unit).
    
    For new code, use MantisV1 instead.
    """

    def __init__(self, seq_len=512, hidden_dim=256, num_patches=32, scalar_scales=None, hidden_dim_scalar_enc=32,
                 epsilon_scalar_enc=1.1, transf_depth=6, transf_num_heads=8, transf_mlp_dim=512, transf_dim_head=128,
                 transf_dropout=0.1, device='cuda', pre_training=False):

        super().__init__()
        assert (seq_len % num_patches) == 0, print(
            'Seq_len must be the multiple of num_patches')
        patch_window_size = int(seq_len / num_patches)

        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.scalar_scales = scalar_scales
        self.hidden_dim_scalar_enc = hidden_dim_scalar_enc
        self.epsilon_scalar_enc = epsilon_scalar_enc
        self.seq_len = seq_len
        self.pre_training = pre_training

        self.tokgen_unit = TokenGeneratorUnit(hidden_dim=hidden_dim,
                                              num_patches=num_patches,
                                              patch_window_size=patch_window_size,
                                              scalar_scales=scalar_scales,
                                              hidden_dim_scalar_enc=hidden_dim_scalar_enc,
                                              epsilon_scalar_enc=epsilon_scalar_enc)
        # Use old naming: vit_unit instead of transf_unit
        self.vit_unit = ViTUnit(hidden_dim=hidden_dim, num_patches=num_patches, depth=transf_depth,
                                heads=transf_num_heads, mlp_dim=transf_mlp_dim, dim_head=transf_dim_head,
                                dropout=transf_dropout, device=device)

        self.prj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.to(device)

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, x):
        x_embeddings = self.tokgen_unit(x)
        vit_out = self.vit_unit(x_embeddings)
        if self.pre_training:
            return self.prj(vit_out)
        else:
            return vit_out

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load a model from HuggingFace Hub with backward compatibility.
        
        Loads old checkpoints that use the 'vit_unit' naming.
        """
        # Call parent's from_pretrained
        model = super().from_pretrained(*args, **kwargs)
        
        return model
