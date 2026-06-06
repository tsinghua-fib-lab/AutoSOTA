from utils import *


class RevIN(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        if self.CFG.affine:
            self.affine_w = nn.Parameter(data=torch.ones(self.CFG.enc_in))
            self.affine_b = nn.Parameter(data=torch.zeros(self.CFG.enc_in))

    def forward(self, x, mode=None):
        if self.CFG.revin:
            if mode == 'norm':
                self._get_statistics(x)
                return self._normalize(x)
            elif mode == 'denorm':
                return self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim = tuple(range(1, x.ndim - 1))
        self.means = torch.mean(x, dim=dim, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim, correction=0, keepdim=True) + self.CFG.eps).detach()

    def _normalize(self, x):
        x = x - self.means
        x = x / self.stdev
        if self.CFG.affine:
            x = self.affine_w * x + self.affine_b
        return x

    def _denormalize(self, x):
        if self.CFG.affine:
            x = (x - self.affine_b) / (self.affine_w + self.CFG.eps * self.CFG.eps)
        x = x * self.stdev
        x = x + self.means
        return x


class Embedding(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        self.embedding = nn.Linear(self.CFG.seq_len, self.CFG.d_model, bias=True)
        self.dropout = nn.Dropout(self.CFG.dropout)

    def forward(self, x, x_mark):
        embedding = self.dropout(self.embedding(x.permute(0, 2, 1)))
        return embedding


class FullAttn(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        self.dropout = nn.Dropout(self.CFG.dropout)

    def forward(self, Q, K, V):
        B, L, H, E = Q.shape
        _, S, _, D = V.shape
        A = torch.einsum("blhe,bshe->bhls", Q, K)
        A = self.dropout(torch.softmax(A / math.sqrt(E), dim=-1))
        x = torch.einsum("bhls,bshd->blhd", A, V).contiguous()
        return x


class AttnLayer(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        self.attn = FullAttn(self.CFG)
        d_heads = self.CFG.d_model // self.CFG.n_heads
        self.WQ = nn.Linear(self.CFG.d_model, self.CFG.n_heads * d_heads)
        self.WK = nn.Linear(self.CFG.d_model, self.CFG.n_heads * d_heads)
        self.WV = nn.Linear(self.CFG.d_model, self.CFG.n_heads * d_heads)
        self.WO = nn.Linear(self.CFG.n_heads * d_heads, self.CFG.d_model)

    def forward(self, Q, K, V):
        B, L, _ = Q.shape
        _, S, _ = K.shape
        H = self.CFG.n_heads
        Q = self.WQ(Q).view(B, L, H, -1)
        K = self.WK(K).view(B, S, H, -1)
        V = self.WV(V).view(B, S, H, -1)
        x = self.attn(Q, K, V)
        x = self.WO(x.view(B, L, -1))
        return x


class FeedLayer(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        self.feed = nn.Sequential(
            nn.Conv1d(self.CFG.d_model, self.CFG.d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(self.CFG.dropout),
            nn.Conv1d(self.CFG.d_ff, self.CFG.d_model, kernel_size=1)
        )

    def forward(self, x):
        x = self.feed(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        self.attn_layer = AttnLayer(self.CFG)
        self.feed_layer = FeedLayer(self.CFG)
        if self.CFG.network_norm == 'instance':
            self.norma = nn.InstanceNorm1d(self.CFG.enc_in, affine=True)
            self.normf = nn.InstanceNorm1d(self.CFG.enc_in, affine=True)
        elif self.CFG.network_norm == 'layer':
            self.norma = nn.LayerNorm(self.CFG.d_model)
            self.normf = nn.LayerNorm(self.CFG.d_model)
        else:
            self.norma = nn.Identity()
            self.normf = nn.Identity()
        self.dropout = nn.Dropout(self.CFG.dropout)

    def forward(self, Q, K, V, C):
        A = self.norma(C + self.dropout(self.attn_layer(Q, K, V)))
        x = self.normf(A + self.dropout(self.feed_layer(A)))
        return x, A


class EncoderLayer(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        self.transformer_layer = TransformerLayer(self.CFG)

    def forward(self, x):
        x, _ = self.transformer_layer(x, x, x, x)
        return x


class Encoder(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.CFG) for _ in range(self.CFG.e_layers)])

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x


class AHead(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        in_features = self.CFG.d_model + self.CFG.seq_len // 2 + 1
        out_features = self.CFG.spectrum_size // 2 + 1
        self.amplitude_head = nn.Sequential(
            nn.Linear(in_features, self.CFG.d_ff),
            nn.GELU(),
            nn.Dropout(self.CFG.dropout),
            nn.Linear(self.CFG.d_ff, out_features)
        )
        self.activation = ALU(w=0.5)
        # self._get_spectrum_prior()

    def _get_spectrum_prior(self):
        train_data = TSFactory(self.CFG)('train')[0].x
        spectrum_prior = torch.zeros(1, self.CFG.enc_in, self.CFG.spectrum_size // 2 + 1)
        for i in range(len(train_data) - self.CFG.spectrum_size):
            x = train_data[i:i + self.CFG.spectrum_size].unsqueeze(0)
            if self.CFG.revin:
                dim = tuple(range(1, x.ndim - 1))
                means = torch.mean(x, dim=dim, keepdim=True).detach()
                stdev = torch.sqrt(torch.var(x, dim=dim, correction=0, keepdim=True) + self.CFG.eps).detach()
                x = (x - means) / stdev
            spectrum_prior += torch.abs(torch.fft.rfft(x.permute(0, 2, 1), norm=self.CFG.fourier_norm))
        self.spectrum_prior = spectrum_prior / (len(train_data) - self.CFG.spectrum_size)

    def forward(self, x):
        # spectrum_prior = self.spectrum_prior.to(x.device).repeat(x.shape[0], 1, 1)
        amplitude = self.activation(self.amplitude_head(x))
        return amplitude


class PHead(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        in_features = self.CFG.d_model + self.CFG.seq_len // 2 + 1
        out_features = self.CFG.spectrum_size // 2 + 1
        self.sin_head = nn.Sequential(
            nn.Linear(in_features, self.CFG.d_ff),
            nn.GELU(),
            nn.Dropout(self.CFG.dropout),
            nn.Linear(self.CFG.d_ff, out_features),
            nn.Tanh()
        )
        self.cos_head = nn.Sequential(
            nn.Linear(in_features, self.CFG.d_ff),
            nn.GELU(),
            nn.Dropout(self.CFG.dropout),
            nn.Linear(self.CFG.d_ff, out_features),
            nn.Tanh()
        )

    def forward(self, x):
        sin = self.sin_head(x)
        cos = self.cos_head(x)
        phase = torch.atan2(sin, cos)
        return phase


class ImplicitForecaster(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        self.a_head = AHead(self.CFG)
        self.p_head = PHead(self.CFG)

    def forward(self, x_enc, x):
        fft_x = torch.fft.rfft(x.permute(0, 2, 1), norm=self.CFG.fourier_norm)
        amp_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)
        amp_out = self.a_head(torch.cat((x_enc, amp_x), dim=-1))
        pha_out = self.p_head(torch.cat((x_enc, pha_x), dim=-1))
        x = torch.fft.irfft(amp_out * torch.exp(1j * pha_out), norm=self.CFG.fourier_norm)
        return x.permute(0, 2, 1)
