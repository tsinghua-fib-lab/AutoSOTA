#Taken from https://github.com/acerbilab/amortized-conditioning-engine/blob/main/src/model/encoder.py
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer
from typing import Optional, List


class ACETransformerEncoderLayer(TransformerEncoderLayer):
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        # Self-attention between the full sequence and context sequence which is the same as cross-attention
        # This function rewrites the self-attention block to save computation

        slice_ = attn_mask[0, :]
        zero_mask = slice_ == 0
        num_ctx = torch.sum(
            zero_mask
        ).item()  # We can calculate the number of context elements from the mask a hack

        # self-attention (multihead) is query, key, value,
        x = self.self_attn(
            x,
            x[:, :num_ctx, :],
            x[:, :num_ctx, :],
            attn_mask=None,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]

        return self.dropout1(x)


class TNPDEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward,
        n_head,
        dropout,
        num_layers,
        activation,
        name=None,
    ):
        super().__init__()
        encoder_layer = ACETransformerEncoderLayer(
            d_model, n_head, dim_feedforward, dropout, batch_first=True, activation=activation
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.name = name

    def create_mask(self, batch):
        num_ctx = batch.xc.shape[1]
        num_tar = batch.xt.shape[1]
        num_all = num_ctx + num_tar

        mask = torch.zeros(num_all, num_all, device=batch.xc.device).fill_(float("-inf"))
        mask[:, :num_ctx] = 0.0

        return mask, num_tar

    def forward(self, batch, embeddings):
        mask, num_tar = self.create_mask(batch)
        out = self.encoder(embeddings, mask=mask)
        z_target = out[:, -num_tar:]  # [Bx T x hidden_dim]
        return z_target
    


# Define a dummy batch class for testing
class DummyBatch:
    def __init__(self, xc, xt):
        self.xc = xc
        self.xt = xt


# Test the implementation
if __name__ == "__main__":

    def create_mask(batch):
        num_ctx = batch.xc.shape[1]
        num_tar = batch.xt.shape[1]
        num_all = num_ctx + num_tar

        mask = torch.zeros(num_all, num_all).fill_(float("-inf"))
        mask[:, :num_ctx] = 0.0

        return mask, num_tar

    d_model = 512
    dim_feedforward = 2048
    nhead = 4
    dropout = 0.0
    num_layers = 1

    # Create a dummy batch
    xc = torch.rand(32, 20, d_model)  # (batch_size, num_ctx, d_model)
    xt = torch.rand(32, 12, d_model)  # (batch_size, num_tar, d_model)
    batch = DummyBatch(xc, xt)

    # Concatenate context and target for embeddings
    embeddings = torch.cat(
        (batch.xc, batch.xt), dim=1
    )  # (batch_size, num_all, d_model)

    # Instantiate standard TransformerEncoderLayer
    standard_encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True,
    )

    mask, num_tar = create_mask(batch)

    # Run both models
    standard_output = standard_encoder_layer(embeddings)
    src_key_padding_mask = None

    num_ctx = 20
    out = standard_encoder_layer._sa_block(embeddings, mask, src_key_padding_mask)
    z_target = out[:, -num_tar:]

    xc_in = embeddings[:, :num_ctx, :]
    xt_in = embeddings[:, num_ctx:, :]

    x_out = standard_encoder_layer.self_attn(
        embeddings,
        xc_in,
        xc_in,
        attn_mask=None,
        key_padding_mask=src_key_padding_mask,
        need_weights=False,
        is_causal=False,
    )[0]

    cross_attn_output = standard_encoder_layer.self_attn(
        xt_in,
        xc_in,
        xc_in,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=False,
        is_causal=False,
    )[0]

    print(x_out[0, 0, :10])
    print(out[0, 0, :10])
    print(torch.allclose(out, x_out, atol=0.01))

    print(out[-1, -1, :10])
    print(cross_attn_output[-1, -1, :10])
    print(torch.allclose(out[:, -num_tar:], cross_attn_output, atol=0.01))