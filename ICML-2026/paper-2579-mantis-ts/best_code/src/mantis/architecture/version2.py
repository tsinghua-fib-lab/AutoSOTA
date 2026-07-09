import torch

from torch import nn

from einops import repeat
from huggingface_hub import PyTorchModelHubMixin

from .tokgen_utils.convolution import Convolution
from .tokgen_utils.encoders import MultiScaledScalarEncoder, LinearEncoder
from .transformer_v2_utils.transformer import Transformer


# ==================================
# ====       Organization:      ====
# ==================================
# ==== class TokenGeneratorUnit ====
# ==== class TransformerUnit    ====
# ==== class MantisV2           ====
# ==================================


class TokenGeneratorUnit(nn.Module):
    def __init__(self, hidden_dim, num_patches, scalar_scales, hidden_dim_scalar_enc,
                 epsilon_scalar_enc, kernel_size):
        super().__init__()
        self.num_patches = num_patches
        # token generator for time series objects
        num_ts_feats = 2  # original ts + its diff
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
        # dim(bs, hidden_dim, len_ts)
        embedding = self.convs[0](self.ts_scaler(diff_x))
        ts_var_embeddings.append(embedding)
        
        # original ts
        # dim(bs, hidden_dim, len_ts)
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
        self.cls_token = nn.Parameter(torch.randn(hidden_dim).to(device))
        self.transformer = Transformer(num_layers=depth, embed_dim=hidden_dim, num_heads=heads, 
                                       mlp_hidden_dim=mlp_dim, dropout=dropout, dim_head=dim_head)
        
    def forward(self, x, output_token='cls_token', return_layer=-1):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=b)

        x_embeddings = torch.cat([x, cls_tokens], dim=1)
        x_embeddings = x_embeddings[:, None, :, :]
        x_embeddings = self.transformer(x_embeddings, return_layer=return_layer)[:, 0, :, :]
        cls_token = x_embeddings[:, -1, :]
        if output_token == 'cls_token':
            return cls_token
        mean_token = x_embeddings[:, :-1, :].mean(axis=1)
        if output_token == 'mean_token':
            return mean_token
        if output_token == 'combined':
            return torch.cat([cls_token, mean_token], dim=1)
        raise KeyError("Unknown output type")


class MantisV2(
    nn.Module,
    PyTorchModelHubMixin,
    # optionally, you can add metadata which gets pushed to the model card
    library_name="mantis",
    repo_url="https://huggingface.co/paris-noah/MantisV2/tree/main",
    pipeline_tag="time-series-foundation-model",
    license="mit",
    tags=["time-series-foundation-model"],
):
    """
    The architecture of MantisV2 time series foundation model.
    The whole model has 4,188,672 parameters.

    According to the paper, for zero-shot feature extraction it's better to return the 3rd (index 2) transformer layer.
    In this case, the number of parameters is 2,214,144.
    Set `return_transf_layer=2` and `output_token='combined'` to reproduce the results.

    Parameters
    ----------
    hidden_dim: int, default=256
        Size of a patch (token), i.e., what the hidden dimension each patch is projected to. At the same time,
        ``hidden_dim`` corresponds to the dimension of the embedding space.
    num_patches: int, default=32
        Number of patches (tokens).
    kernel_size: int, default=41
        The kernel size of convolution layers.
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
    transf_dropout: float, default=0.1
        Dropout value used for Transformer in TransformerUnit.
    return_transf_layer: int, default=-1
        Truncate the model by taking the output of one of the intermediate transformer layers. 
        The default value (-1) means that the output of the last layer is returned.
        IMPORTANT: enumeration starts from 0.
    output_token: {'cls_token', 'mean_token', 'combined'}, default='cls_token'
        How to aggregate output tokens. By default, only the classification token is returned,
        which is usually the best choice for the last layer. 'mean_token' returns the average of all tokens
        except the classification one. 'combined' is concatenation of the mean and classification tokens.
        For intermediate transformer layers, it is better to return 'combined'.
    device: {'cpu', 'cuda'}, default='cuda'
        On which device the model is located.
    pre_training: bool, default=False
        If True, applies an MLP projector after the TransformerUnit, which originally was used to pre-train the model using
        InfoNCE contrastive loss.
    """

    def __init__(self, hidden_dim=256, num_patches=32, kernel_size=41, scalar_scales=None, hidden_dim_scalar_enc=32,
                 epsilon_scalar_enc=1.1, transf_depth=6, transf_num_heads=8, transf_mlp_dim=512, transf_dim_head=32,
                 transf_dropout=0.1, return_transf_layer=-1, output_token='cls_token', device='cuda', pre_training=False):

        super().__init__()

        # final hidden_dim, used only fine-tuning
        self.hidden_dim = 2 * hidden_dim if output_token == 'combined' else hidden_dim
        self.num_patches = num_patches
        self.scalar_scales = scalar_scales
        self.hidden_dim_scalar_enc = hidden_dim_scalar_enc
        self.epsilon_scalar_enc = epsilon_scalar_enc
        self.pre_training = pre_training
        self.return_transf_layer = return_transf_layer
        self.output_token = output_token

        self.tokgen_unit = TokenGeneratorUnit(hidden_dim=hidden_dim,
                                              num_patches=num_patches,
                                              scalar_scales=scalar_scales,
                                              hidden_dim_scalar_enc=hidden_dim_scalar_enc,
                                              epsilon_scalar_enc=epsilon_scalar_enc,
                                              kernel_size=kernel_size)
        self.transf_unit = TransformerUnit(hidden_dim=hidden_dim, num_patches=num_patches, depth=transf_depth,
                                heads=transf_num_heads, mlp_dim=transf_mlp_dim, dim_head=transf_dim_head,
                                dropout=transf_dropout, device=device)

        self.prj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.to(device)

    def to(self, device):
        self.device = device
        return super().to(device)

    def remove_transf_layers(self,):
        """If you want to remove transformer layers after the layer specified by `return_transf_layer`, use this method."""
        self.transf_unit.transformer.layers = self.transf_unit.transformer.layers[:self.return_transf_layer+1]

    def from_pretrained(self, *args, **kwargs):
        network = super().from_pretrained(*args, **kwargs)
        network.return_transf_layer = self.return_transf_layer
        network.output_token = self.output_token
        network.pre_training = self.pre_training
        network.hidden_dim = self.hidden_dim
        network.device = self.device
        return network.to(self.device)

    def forward(self, x):
        """
        x should be of size (n_samples, n_channels=1, seq_len), where seq_len must be a multiple of num_patches.
        If your time series data are with non-fixed sequence length, please pad or resize them to the same size.
        """
        seq_len = x.shape[2]
        assert (seq_len % self.num_patches) == 0, print('Seq_len must be the multiple of num_patches')

        x_embeddings = self.tokgen_unit(x)
        x_embeddings = self.transf_unit(x_embeddings,
                                        output_token=self.output_token,
                                        return_layer=self.return_transf_layer,
                                        )
        if self.pre_training:
            return self.prj(x_embeddings)
        else:
            return x_embeddings
