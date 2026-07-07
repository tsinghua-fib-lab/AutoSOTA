import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PCRNet(nn.Module):
    def __init__(self, args):
        super(PCRNet, self).__init__()
        in_channel = args.eeg_channel
        seq_len = args.window_length

        self.out = nn.Linear(5, 2)
        self.flatten = nn.Flatten()

        self.channelAttention = Temporal_Context_Calibration(
            args, dim=in_channel, num_heads=16, bias=False
        )

        self.Context_Fusion = Dual_Domain_Integration(
            in_channel=1,
            seq_len=seq_len,
            base_dim=16,
        )

        self.Spatiotemporal_Convolution = Spatiotemporal_Convolution(
            in_channel, seq_len
        )

    def forward(self, x):
        # Input: [B, 1, C, T]
        x = x.permute(0, 2, 1, 3)
        x = self.channelAttention(x)
        x = x.permute(0, 2, 1, 3)

        x = self.Context_Fusion(x)

        x = self.Spatiotemporal_Convolution(x)
        x = self.flatten(x)
        x = self.out(x)
        return x


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class Complex_Feature_Mixer(nn.Module):
    def __init__(self, channels, expansion_factor=2):
        super(Complex_Feature_Mixer, self).__init__()
        hidden_dim = channels * expansion_factor
        self.linear_res = nn.Conv2d(channels, channels, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
        )
        self.linear_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        res = self.linear_res(x)
        hidden = self.mlp(x)
        out = res + hidden
        out = self.linear_out(out)
        out = self.norm(out)
        return out


class Gated_Spectral_Refinement(nn.Module):
    def __init__(self, channels, expand=2):
        super(Gated_Spectral_Refinement, self).__init__()

        self.amplitude_importance_gate = nn.Sequential(
            nn.Conv2d(channels, channels * expand, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels * expand, channels, 1),
            nn.Sigmoid(),  # Outputs a 0-1 mask
        )

        self.real_mixer = Complex_Feature_Mixer(channels, expansion_factor=expand)
        self.imag_mixer = Complex_Feature_Mixer(channels, expansion_factor=expand)

        self.post_process = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.GELU())

    def forward(self, x):
        _, _, H, W = x.shape

        x_freq = torch.fft.rfft2(x, norm="backward")

        mag = torch.abs(x_freq)
        real = x_freq.real
        imag = x_freq.imag

        mag_mask = self.amplitude_importance_gate(mag)

        real_feat = self.real_mixer(real)
        imag_feat = self.imag_mixer(imag)

        real_refined = real_feat * mag_mask
        imag_refined = imag_feat * mag_mask

        x_freq_refined = torch.complex(real_refined, imag_refined)
        x_out = torch.fft.irfft2(x_freq_refined, s=(H, W), norm="backward")

        return self.post_process(x_out)


class Residual_Spectral_Interface(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.gsrm = Gated_Spectral_Refinement(channels, expand=2)  # Uses GSRM

        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
        shortcut = inp

        x = self.norm(inp)
        x_freq = self.gsrm(x)

        x = shortcut + x_freq * self.gamma
        return x


class Dilated_Temporal_Context_Unit(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation=1):
        super(Dilated_Temporal_Context_Unit, self).__init__()
        k_h, k_w = kernel_size
        pad_h = (k_h - 1) * dilation // 2
        pad_w = (k_w - 1) * dilation // 2

        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=(pad_h, pad_w),
            dilation=dilation,
        )
        self.norm = nn.BatchNorm2d(out_channel)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x))) + x


class Dual_Domain_Integration(nn.Module):
    def __init__(self, in_channel, seq_len, base_dim=16):
        super(Dual_Domain_Integration, self).__init__()

        self.preprocess = nn.Sequential(
            nn.BatchNorm2d(in_channel), nn.Conv2d(in_channel, base_dim, 1), nn.GELU()
        )

        self.time_stream = nn.Sequential(
            Dilated_Temporal_Context_Unit(base_dim, base_dim, kernel_size=(1, 3), dilation=1),
            Dilated_Temporal_Context_Unit(base_dim, base_dim, kernel_size=(1, 3), dilation=2),
            Dilated_Temporal_Context_Unit(base_dim, base_dim, kernel_size=(1, 3), dilation=4),
        )

        self.freq_stream = Residual_Spectral_Interface(channels=base_dim)

        # Cross-Domain Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim, 1),
            nn.BatchNorm2d(base_dim),
            nn.GELU(),
            nn.Conv2d(base_dim, in_channel, 1),  # Project back to original dim
        )

    def forward(self, x):
        shortcut = x
        x_emb = self.preprocess(x)

        x_time = self.time_stream(x_emb)
        x_freq = self.freq_stream(x_emb)

        combined = torch.cat([x_time, x_freq], dim=1)
        out = self.fusion(combined)

        return out + shortcut


class Spatiotemporal_Convolution(nn.Module):
    def __init__(self, in_channel, seq_len):
        super(Spatiotemporal_Convolution, self).__init__()
        self.Temporal_Convolution = nn.Sequential(
            nn.Conv2d(1, 5, (1, 2), stride=1), nn.BatchNorm2d(5), nn.GELU()
        )
        self.Spatio_Convolution = nn.Sequential(
            nn.Conv2d(5, 5, (in_channel, 1), stride=1), nn.BatchNorm2d(5), nn.GELU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.Temporal_Convolution(x)
        x = self.Spatio_Convolution(x)
        x = self.pool(x)
        return x


class Temporal_Context_Calibration(nn.Module):
    def __init__(self, args, dim, num_heads, bias):
        super(Temporal_Context_Calibration, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5)
        self.w_q = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(
                dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
            ),
        )
        self.w_k = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(
                dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
            ),
        )
        self.w_v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(
                dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
            ),
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.mta_q = Multiscale_Temporal_Attention(args)
        self.mta_k = Multiscale_Temporal_Attention(args)
        self.mta_v = Multiscale_Temporal_Attention(args)
        self.bn_q = nn.BatchNorm2d(dim)
        self.bn_k = nn.BatchNorm2d(dim)
        self.bn_v = nn.BatchNorm2d(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = q * (1 + torch.sigmoid(self.mta_q(q)))
        k = k * (1 + torch.sigmoid(self.mta_k(k)))
        v = v * (1 + torch.sigmoid(self.mta_v(v)))
        q = self.bn_q(q)
        k = self.bn_k(k)
        v = self.bn_v(v)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        scale = 1.0 / math.sqrt(q.size(-1))
        attn = (q @ k.transpose(-2, -1)) * scale * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out = self.project_out(out)
        return out


class Multiscale_Temporal_Layer(nn.Module):
    def __init__(self, seq_len, kernel_size):
        super(Multiscale_Temporal_Layer, self).__init__()
        self.multiscaleConv = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=kernel_size, padding="same"
        )
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(seq_len)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.multiscaleConv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class Multiscale_Temporal_Attention(nn.Module):
    def __init__(self, args):
        super(Multiscale_Temporal_Attention, self).__init__()
        in_channel = args.eeg_channel
        seq_len = args.window_length
        self.spatioConv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(in_channel, 1)
        )
        self.upChannelConv = nn.Conv1d(
            in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0
        )
        self.project_out = nn.Conv2d(
            in_channels=1, out_channels=in_channel, kernel_size=1, stride=1
        )
        self.multiTemporal_K_2 = Multiscale_Temporal_Layer(seq_len, kernel_size=2)
        self.multiTemporal_K_4 = Multiscale_Temporal_Layer(seq_len, kernel_size=4)
        self.multiTemporal_K_6 = Multiscale_Temporal_Layer(seq_len, kernel_size=6)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.spatioConv(x)
        x = self.upChannelConv(x.squeeze(2))
        x, y, z = x.chunk(3, dim=1)
        x_attn = self.multiTemporal_K_2(x)
        y_attn = self.multiTemporal_K_4(y)
        z_attn = self.multiTemporal_K_6(z)
        out = x_attn * x + y_attn * y + z_attn * z
        out = out.view(x.shape[0], 1, 1, -1)
        out = self.project_out(out)
        return out

