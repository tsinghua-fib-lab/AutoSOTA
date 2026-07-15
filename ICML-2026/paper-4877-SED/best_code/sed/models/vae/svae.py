import copy
import math

import lightning as L
import torch
from torch import nn
from torch.nn.functional import log_softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


class SVAE(L.LightningModule):
    def __init__(self, data_dimensions: int, num_layers: int = 6, d_model: int = 512, d_ff: int = 2048, h: int = 8, dropout: float = 0.1, beta: float = 0.000001, input_mode: str = None, lr: float = None):
        super().__init__()
        self.save_hyperparameters()
        self.input_mode: str = input_mode
        self.d_model: int = d_model
        self.data_dimensions: int = data_dimensions
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(
            d_model, c(attn), c(ff), dropout), num_layers, d_model)
        self.decoder = Decoder(DecoderLayer(
            d_model, c(attn), c(attn), c(ff), dropout), num_layers)
        self.embed = PosValueEmbedding(d_model, dropout, data_dimensions+3)
        # +1 for value, +3 for special tokens
        self.generator = Generator(d_model, data_dimensions+4)
        self.beta: float = beta
        self.lr: float = lr
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, src_positions, src_values):
        pad_token = self.data_dimensions + 2
        src_mask = (src_positions != pad_token).unsqueeze(-2)
        tgt_positions = src_positions[:, :-1]
        tgt_values = src_values[:, :-1]
        tgt_y_positions = src_positions[:, 1:]
        tgt_y_values = src_values[:, 1:]
        tgt_mask = (tgt_positions != pad_token).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt_positions.size(-1)).type_as(
            tgt_mask
        )
        mu_logvar = self.encode(src_positions, src_values,
                                src_mask).view(-1, 2, self.d_model)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterize(mu, logvar)
        return self.decode(z, None, tgt_positions, tgt_values, tgt_mask), mu, logvar, tgt_y_positions, tgt_y_values

    def encode(self, src_positions, src_values, src_mask):
        return self.encoder(self.embed(src_positions, src_values), src_mask)

    def decode(self, memory, src_mask, tgt_positions, tgt_values, tgt_mask):
        return self.decoder(self.embed(tgt_positions, tgt_values), memory, src_mask, tgt_mask)

    def training_step(self, input, batch_idx):
        loss_dict, _, _, _ = self.step(
            input)
        log_dict = {"train/loss": loss_dict["loss"], "train/CE": loss_dict["CE"], "train/MSE": loss_dict["MSE"],
                    "train/KLD": loss_dict["KLD"]}
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=True)
        return loss_dict["loss"]

    def validation_step(self, input, batch_idx):
        loss_dict, in_positions, in_values, mu = self.step(
            input)
        log_dict = {"val/loss": loss_dict["loss"], "val/CE": loss_dict["CE"], "val/MSE": loss_dict["MSE"],
                    "val/KLD": loss_dict["KLD"]}
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)
        return in_positions, in_values, mu

    def step(self, input):
        in_positions, in_values = input
        out, mu, logvar, tgt_y_positions, tgt_y_values = self(
            in_positions, in_values)
        out_positions, out_values = self.generator(out)
        out_values.squeeze_()
        CE_loss = nn.functional.cross_entropy(
            out_positions.view(-1, out_positions.size(-1)), tgt_y_positions.contiguous().view(-1), ignore_index=self.data_dimensions + 2)
        non_pad_mask = (tgt_y_positions != self.data_dimensions +
                        2).logical_and(tgt_y_positions != self.data_dimensions + 1)
        filtered_out_values = out_values[non_pad_mask]
        filtered_tgt_y_values = tgt_y_values[non_pad_mask]
        MSE_loss = nn.functional.mse_loss(
            filtered_out_values, filtered_tgt_y_values
        )
        KLD_loss = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
        loss = MSE_loss + CE_loss + self.beta * KLD_loss
        loss_dict = {"loss": loss, "CE": CE_loss, "MSE": MSE_loss,
                     "KLD": KLD_loss}
        return loss_dict, in_positions, in_values, mu

    def sample(self, batch_size, z=None):
        if z == None:
            z = torch.randn((batch_size, self.d_model)).to(self.device)
        start_position = self.data_dimensions
        start_value = -1
        end_position = self.data_dimensions +1
        end_value = -2
        out_positions = torch.zeros(batch_size, 1).fill_(
            start_position).to(device=self.device, dtype=torch.int64)
        out_values = torch.zeros(batch_size, 1).fill_(
            start_value).to(device=self.device, dtype=torch.float32)
        max_len = self.data_dimensions+1  # +1 for end token
        end_reached = torch.full((batch_size,), False).to(self.device)
        for i in range(max_len):
            out = self.decode(z, None, out_positions, out_values, subsequent_mask(
                out_positions.size(1)).type_as(z))
            prob_pos, next_value = self.generator(out[:, -1].unsqueeze(1))
            _, next_position = torch.max(prob_pos, dim=2)
            end_reached = torch.logical_or(
                end_reached, next_position.squeeze() == end_position)
            out_positions = torch.cat(
                [out_positions, next_position], dim=1
            )
            out_values = torch.cat(
                [out_values, next_value.squeeze(2)], dim=1
            )
            if torch.all(end_reached):
                break
        return out_positions, out_values

    def configure_optimizers(self):
        if self.lr != None:
            optimizer = Adam(self.parameters(),
                             lr=self.lr, betas=(0.9, 0.99), eps=1e-9)
            return optimizer
        else:
            optimizer = Adam(self.parameters(),
                             lr=1.0, betas=(0.9, 0.99), eps=1e-9)
            lr_scheduler = LambdaLR(
                optimizer=optimizer,
                lr_lambda=lambda step: rate(
                    step, self.d_model, factor=1, warmup=4000
                ),
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}}


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, out_dim):
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x):
        x = self.proj(x)
        return x[:, :, :-1], x[:, :, -1:]


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N, d_model):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.proj_mu_logvar = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * d_model)
        )

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x).mean(dim=1)
        x = self.proj_mu_logvar(x)
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class PosValueEmbedding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.value_embedding = ValueEmbedding(d_model)
        self.position_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, positions, values):
        emb_positions = self.position_encoding(positions)
        emb_values = self.value_embedding(values)
        return self.dropout(emb_positions+emb_values)


class ValueEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.stv = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model)
        )

    def forward(self, value):
        return self.stv(value.unsqueeze(-1)) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, positions):
        return self.pe[positions].requires_grad_(False)
