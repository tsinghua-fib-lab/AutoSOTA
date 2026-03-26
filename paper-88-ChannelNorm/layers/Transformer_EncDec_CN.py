import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelwiseLayerNorm(nn.Module):
    def __init__(self, num_channels, num_features, eps=1e-5):
        super(ChannelwiseLayerNorm, self).__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.eps = eps
        
        # Learnable weights and biases for each channel
        self.weight = nn.Parameter(torch.ones(num_channels, num_features))
        self.bias = nn.Parameter(torch.zeros(num_channels, num_features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # (batch_size, num_channels, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (batch_size, num_channels, 1)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (batch_size, num_channels, num_features)
        x_out = x_normalized * self.weight + self.bias  # (batch_size, num_channels, num_features)
        return x_out
    
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, C, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        self.norm1 = ChannelwiseLayerNorm(C, d_model)
        self.norm2 = ChannelwiseLayerNorm(C, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

    def viz_per_layer2(self, x, attn_mask=None, tau=None, delta=None):
        x_preprenorm = x.clone().detach()
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        x_prenorm = x.clone().detach()
        y = x = self.norm1(x)
        x_post_norm = x.clone().detach()
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, x_preprenorm,x_prenorm, x_post_norm


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def viz_per_layer(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        return_xs = []
        return_xs_prenorm = []
        return_xs_preprenorm = []
        return_xs_postnorm = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn,x_preprenorm,x_prenorm, x_post_norm = attn_layer.viz_per_layer2(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                return_xs.append(x)
                return_xs_preprenorm.append(x_preprenorm)
                return_xs_prenorm.append(x_prenorm)
                return_xs_postnorm.append(x_post_norm)
                attns.append(attn)
            x, attn,x_preprenorm,x_prenorm, x_post_norm = (self.attn_layers[-1]).viz_per_layer2(x, tau=tau, delta=None)
            return_xs.append(x)
            return_xs_preprenorm.append(x_preprenorm)
            return_xs_prenorm.append(x_prenorm)
            return_xs_postnorm.append(x_post_norm)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn,x_preprenorm,x_prenorm, x_post_norm = attn_layer.viz_per_layer2(x, attn_mask=attn_mask, tau=tau, delta=delta)
                return_xs.append(x)
                return_xs_prenorm.append(x_prenorm)
                return_xs_preprenorm.append(x_preprenorm)
                return_xs_postnorm.append(x_post_norm)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
            return_xs.append(x)

        return return_xs, return_xs_preprenorm, return_xs_prenorm, return_xs_postnorm
    
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
