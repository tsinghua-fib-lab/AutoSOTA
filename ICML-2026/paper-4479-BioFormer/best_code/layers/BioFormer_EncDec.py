import torch
import torch.nn as nn
import torch.nn.functional as F

class Sample_con_ln(nn.Module):
    """
    LayerNorm + data-dependent scale-shift modulation.
    """

    def __init__(
        self,
        normalized_shape,
        proj_hidden=64,
        hidden_dim=128,
        alpha=0.1,
    ):
        super().__init__()

        self.alpha = alpha
        self.C = int(normalized_shape)
        self.ln = nn.LayerNorm(self.C)

        # pool & project to low-dim
        self.pool_proj = nn.Linear(self.C, proj_hidden)

        # MLP generates γ and β
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(proj_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.C * 2),  # 2C: γ||β
        )

        # init as plain LayerNorm: γ=1, β=0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        with torch.no_grad():
            self.mlp[-1].bias[: self.C] += 1.0  # γ part starts at 1

    def forward(self, x):
        x_norm = self.ln(x)  # [B, T, C]
        pooled = x.mean(dim=1).detach()  # [B, C]
        h = self.pool_proj(pooled)       # [B, proj_hidden]
        gamma_beta = self.mlp(h)         # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C] each

        gamma = gamma.unsqueeze(1)  # [B, 1, C]
        beta = beta.unsqueeze(1)

        mod = gamma * x_norm + beta  # scale & shift
        out = x_norm + self.alpha * (mod - x_norm)  # residual blending

        return out

class FreqBandAlign(nn.Module):

    def __init__(self, num_class, mag_learning, phase_learning, num_bands, d_token=64, kernel_size=3,
                 dropout=0.1, gamma=0.5, eps=1e-6):
        super().__init__()
        assert kernel_size in (3, 5)
        self.num_class = num_class
        self.mag_learning = mag_learning
        self.phase_learning = phase_learning
        self.num_bands = num_bands
        self.kernel_size = kernel_size
        self.eps = eps

        # Learnable parameters
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.tokens = nn.Parameter(torch.zeros(self.num_bands, d_token))
        # nn.init.xavier_uniform_(self.tokens)

        # Band statistics projection
        self.band_proj = nn.Sequential(
            nn.Linear(5, d_token),
            nn.LayerNorm(d_token)
        )

        # Cross-attention projections
        self.q_proj = nn.Linear(d_token, d_token, bias=False)
        self.k_proj = nn.Linear(d_token, d_token, bias=False)
        self.v_proj = nn.Linear(d_token, d_token, bias=False)
        self.attn_norm = nn.LayerNorm(d_token)

        # Dynamic kernel & gain heads
        self.to_kernel = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.LayerNorm(d_token),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_token, kernel_size)
        )

        self.to_gain = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.LayerNorm(d_token),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1)
        )

        # Phase drift head
        self.to_phase = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.LayerNorm(d_token),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1)
        )

    # Frequency transform
    def _fft(self, x):
        """FFT and split DC / non-DC components"""
        z = x.permute(0, 2, 1).contiguous()
        Z = torch.fft.rfft(z, dim=-1, norm='ortho')
        Z_dc = Z[..., :1]
        Z_rest = Z[..., 1:]
        mag = torch.abs(Z_rest) + self.eps
        phase = Z_rest / mag  # unit-complex phase
        return Z_dc, mag, phase

    # Band splitting and statistics
    def _split_bands(self, mag):
        """Split spectrum into bands and compute statistics"""
        B, D, F = mag.shape
        edges = torch.linspace(0, F, self.num_bands + 1,
                               device=mag.device, dtype=torch.long)

        band_mags = []
        band_feats = []

        for k in range(self.num_bands):
            s, e = edges[k].item(), edges[k + 1].item()
            if e <= s:
                e = s + 1

            band_mag = mag[..., s:e]
            band_mags.append((s, e, band_mag))

            # Band-level statistics (log-scaled)
            bm = torch.log(band_mag.mean(dim=-1).mean(dim=-1))
            bv = band_mag.var(dim=-1, unbiased=False)
            bs = torch.log(torch.sqrt(bv + self.eps).mean(dim=-1) + self.eps)
            bx = torch.log(band_mag.amax(dim=-1).mean(dim=-1))
            bf = band_mag.argmax(dim=-1).float().mean(dim=-1)
            be = torch.log(band_mag.pow(2).sum(dim=-1).mean(dim=-1) + self.eps)

            band_feats.append(torch.stack([bm, bs, bx, be, bf], dim=-1))

        band_feats = torch.stack(band_feats, dim=1)
        return band_mags, band_feats

    # Cross-attention over frequency bands
    def _cross_attention(self, band_feats, B):
        """Compute band tokens via cross-attention"""
        band_ctx = self.band_proj(band_feats)
        tokens = self.tokens.unsqueeze(0).expand(B, -1, -1) 
               
        Q = self.q_proj(tokens)
        K = self.k_proj(band_ctx)
        V = self.v_proj(band_ctx)

        scale = K.shape[-1] ** 0.5
        attn_logits = torch.matmul(Q, K.transpose(1, 2)) / scale
        attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True)[0]

        attn = torch.softmax(attn_logits, dim=-1)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        O = torch.matmul(attn, V)
        return self.attn_norm(O)

    # Apply magnitude & phase drift per band
    def _apply_band_drift(self, mag, phase, band_mags, O):
        """Apply dynamic convolution and phase rotation per band"""
        ker = torch.softmax(self.to_kernel(O), dim=-1)
        gain = torch.tanh(self.to_gain(O).squeeze(-1)) * 0.5

        mag_new = mag.clone()
        phase_new = phase.clone()
        pad = self.kernel_size // 2

        for k, (s, e, band_mag) in enumerate(band_mags):
            B_k, D_k, Fk = band_mag.shape

            # Padding for convolution
            if Fk <= pad:
                band_mag_padded = F.pad(band_mag, (pad, pad), mode='replicate')
            else:
                band_mag_padded = F.pad(band_mag, (pad, pad), mode='reflect')

            # Dynamic convolution on magnitude
            if self.num_class > 2:
                w = ker[:, k, :].mean(dim=0, keepdim=True)
                w = w / (w.sum(dim=-1, keepdim=True) + self.eps)
                w = w.repeat(D_k, 1, 1)
                conv_out = F.conv1d(band_mag_padded, w, groups=D_k)
            else:
                inp = band_mag_padded.reshape(1, B_k * D_k, -1)
                w = ker[:, k, :].unsqueeze(1).repeat(1, D_k, 1)
                w = w.reshape(B_k * D_k, 1, -1)
                w = w / (w.sum(dim=-1, keepdim=True) + self.eps)
                conv_out = F.conv1d(inp, w, groups=B_k * D_k)
                conv_out = conv_out.reshape(B_k, D_k, -1)

            # Magnitude residual update
            g_mag = 1 + self.gamma * gain[:, k].view(B_k, 1, 1)
            g_mag = torch.clamp(g_mag, 0.5, 1.5)
            delta_mag = g_mag * (conv_out - band_mag)
            delta_mag = torch.clamp(delta_mag,
                                    -band_mag * 0.5,
                                    band_mag * 0.5)
            
            # Phase rotation
            phase_gain = torch.tanh(self.to_phase(O[:, k])).view(B_k, 1, 1)
            band_phase = phase[..., s:e]
            delta_phase = torch.exp(1j * phase_gain * torch.angle(band_phase))
            band_phase_new = band_phase * delta_phase
            band_phase_new = band_phase_new / (torch.abs(band_phase_new) + self.eps)

            mag_new[..., s:e] = band_mag + delta_mag
            phase_new[..., s:e] = band_phase_new

        return mag_new, phase_new

    # Forward
    def forward(self, x_list):
        out_list = []

        for x in x_list:
            B, P, D = x.shape
            # 1) FFT
            Z_dc, mag, phase = self._fft(x)

            # Band split & statistics
            band_mags, band_feats = self._split_bands(mag)

            # Cross-attention
            O = self._cross_attention(band_feats, B)

            # Band-wise drift
            mag_new, phase_new = self._apply_band_drift(
                mag, phase, band_mags, O
            )
            # Inverse FFT
            if self.mag_learning == False:
                Z_new = torch.cat([Z_dc, mag * phase_new], dim=-1)
            elif self.phase_learning == False:
                Z_new = torch.cat([Z_dc, mag_new * phase], dim=-1)
            else:
                Z_new = torch.cat([Z_dc, mag_new * phase_new], dim=-1)
            z_new = torch.fft.irfft(Z_new, n=P, dim=-1, norm='ortho').real
            x_out = z_new.permute(0, 2, 1).contiguous()

            out_list.append(x_out)

        return out_list

class TimeSegAlign(nn.Module):
    """
    Segment-based version (uniform temporal split), NO filterbank.

    - Split sequence into K equal (or near-equal) temporal segments along P.
    - Compute 5-dim stats per segment.
    - Cross-attention from learnable tokens -> segment controls O.
    - Apply per-segment dynamic Conv1d + residual gain (along time within each segment).
    - Merge segments back by concatenation to recover [B,P,D].

    Input:  x_list: list of [B,P,D]
    Output: out_list: list of [B,P,D]
    """

    def __init__(
        self,
        num_class: int,
        num_bands: int = 6,
        d_token: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
        gamma: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert kernel_size in (3, 5)
        self.num_class = num_class
        self.num_bands = num_bands
        self.kernel_size = kernel_size
        self.eps = eps

        # Learnable parameters
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.tokens = nn.Parameter(torch.zeros(self.num_bands, d_token))

        # Band(=segment) statistics projection: 5 -> d_token
        self.band_proj = nn.Sequential(
            nn.Linear(5, d_token),
            nn.LayerNorm(d_token)
        )

        # Cross-attention projections
        self.q_proj = nn.Linear(d_token, d_token, bias=False)
        self.k_proj = nn.Linear(d_token, d_token, bias=False)
        self.v_proj = nn.Linear(d_token, d_token, bias=False)
        self.attn_norm = nn.LayerNorm(d_token)

        # Dynamic kernel & gain heads
        self.to_kernel = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.LayerNorm(d_token),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_token, kernel_size)
        )

        self.to_gain = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.LayerNorm(d_token),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1)
        )

    # -------- Uniform temporal split & stats --------
    def _split_segments_time(self, x: torch.Tensor):
        """
        Split along time axis into K segments (as equal as possible).

        x: [B,P,D]
        returns:
          segs: list[K] of [B,D,Pk]   (note: Pk can vary by at most 1)
          seg_feats: [B,K,5]
          seg_spans: list[K] of (s,e) indices on time axis (for merging)
        """
        B, P, D = x.shape

        # time-major -> channel-major for conv convenience: [B,D,P]
        x_dp = x.permute(0, 2, 1).contiguous()

        # edges: 0..P split into K parts (near-equal)
        # Use integer linspace-like split that covers all indices exactly.
        edges = torch.linspace(0, P, self.num_bands + 1, device=x.device)
        edges = torch.round(edges).long()
        edges[0] = 0
        edges[-1] = P

        segs = []
        feats = []
        spans = []

        for k in range(self.num_bands):
            s = int(edges[k].item())
            e = int(edges[k + 1].item())
            if e <= s:
                e = min(s + 1, P)

            seg = x_dp[:, :, s:e]  # [B,D,Pk]
            segs.append(seg)
            spans.append((s, e))

            absb = seg.abs() + self.eps

            # bm: log(mean |x|)
            bm = absb.mean(dim=-1).mean(dim=-1) + self.eps  # [B]

            # bs: log(mean std over time)
            bv = seg.var(dim=-1, unbiased=False)  # [B,D]
            bs = torch.sqrt(bv + self.eps).mean(dim=-1) + self.eps  # [B]

            # bx: log(mean max |x|)
            bx = absb.amax(dim=-1).mean(dim=-1) + self.eps  # [B]

            # be: log(mean energy)
            be = seg.pow(2).sum(dim=-1).mean(dim=-1) + self.eps  # [B]

            # bf: mean argmax position of |x| within the segment
            bf = absb.argmax(dim=-1).float().mean(dim=-1)  # [B]

            feats.append(torch.stack([bm, bs, bx, be, bf], dim=-1))  # [B,5]

        seg_feats = torch.stack(feats, dim=1)  # [B,K,5]
        return segs, seg_feats, spans

    # -------- Cross-attention --------
    def _cross_attention(self, band_feats: torch.Tensor, B: int):
        """
        band_feats: [B,K,5]
        returns O: [B,K,d_token]
        """
        band_ctx = self.band_proj(band_feats)                 # [B,K,d]
        tokens = self.tokens.unsqueeze(0).expand(B, -1, -1)   # [B,K,d]

        Q = self.q_proj(tokens)
        K = self.k_proj(band_ctx)
        V = self.v_proj(band_ctx)

        scale = K.shape[-1] ** 0.5
        attn_logits = torch.matmul(Q, K.transpose(1, 2)) / scale
        attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True)[0]

        attn = torch.softmax(attn_logits, dim=-1)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)

        O = torch.matmul(attn, V)
        return self.attn_norm(O)

    # -------- Segment-wise drift (dynamic conv + gain) --------
    def _apply_segment_drift_time(self, segs, O):
        """
        segs: list[K] of [B,D,Pk]
        O:    [B,K,d_token]
        returns segs_new: list[K] of [B,D,Pk]
        """
        ker = torch.softmax(self.to_kernel(O), dim=-1)              # [B,K,ks]
        gain = torch.tanh(self.to_gain(O).squeeze(-1)) * 0.5         # [B,K]

        pad = self.kernel_size // 2
        segs_new = []

        for k, seg in enumerate(segs):
            Bk, Dk, Pk = seg.shape

            # padding along time inside this segment
            if Pk <= pad:
                seg_pad = F.pad(seg, (pad, pad), mode="replicate")
            else:
                seg_pad = F.pad(seg, (pad, pad), mode="reflect")

            # dynamic conv along time axis (within segment)
            if self.num_class > 2:
                # shared kernel across batch
                w = ker[:, k, :].mean(dim=0, keepdim=True)  # [1,ks]
                w = w / (w.sum(dim=-1, keepdim=True) + self.eps)
                w = w.view(1, 1, -1).repeat(Dk, 1, 1)       # [D,1,ks]
                conv_out = F.conv1d(seg_pad, w, groups=Dk)  # [B,D,Pk]
            else:
                # per-sample, per-channel kernels
                inp = seg_pad.reshape(1, Bk * Dk, -1)
                w = ker[:, k, :].unsqueeze(1).repeat(1, Dk, 1)  # [B,D,ks]
                w = w.reshape(Bk * Dk, 1, -1)
                w = w / (w.sum(dim=-1, keepdim=True) + self.eps)
                conv_out = F.conv1d(inp, w, groups=Bk * Dk).reshape(Bk, Dk, -1)

            # residual update
            g = 1 + self.gamma * gain[:, k].view(Bk, 1, 1)
            g = torch.clamp(g, 0.5, 1.5)

            delta = g * (conv_out - seg)

            # clamp residual relative to segment magnitude
            ref = seg.abs() + self.eps
            delta = torch.clamp(delta, -0.5 * ref, 0.5 * ref)

            segs_new.append(seg + delta)

        return segs_new

    def _merge_segments_time(self, segs_new, spans, P: int):
        """
        Concatenate segments in original order to recover [B,D,P], then permute back to [B,P,D].
        segs_new: list[K] of [B,D,Pk]
        spans: list[K] of (s,e) used only for sanity; concatenation preserves order.
        """
        # Just concatenate in k order (since we split sequentially).
        y = torch.cat(segs_new, dim=-1)  # [B,D,P'] should equal P
        # In rare rounding cases, adjust to exact length P:
        if y.shape[-1] > P:
            y = y[..., :P]
        elif y.shape[-1] < P:
            # pad at end if needed
            y = F.pad(y, (0, P - y.shape[-1]), mode="replicate")

        return y.permute(0, 2, 1).contiguous()  # [B,P,D]

    # -------- Forward --------
    def forward(self, x_list):
        out_list = []

        for x in x_list:
            B, P, D = x.shape

            # 1) uniform temporal split + stats
            segs, seg_feats, spans = self._split_segments_time(x)

            # 2) cross-attention over segments
            O = self._cross_attention(seg_feats, B)

            # 3) segment-wise drift
            segs_new = self._apply_segment_drift_time(segs, O)

            # 4) merge segments back
            x_out = self._merge_segments_time(segs_new, spans, P)

            out_list.append(x_out)

        return out_list

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_list, attn_mask=None,tau=None, delta=None):
        # x_list: list of (B, Li, D)
        out_list = []
        attn_list = []

        for x in x_list:
            new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
            x = x + self.dropout(new_x)

            y = self.norm1(x)
            y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
            y = self.dropout(self.conv2(y).transpose(-1,1))

            out = self.norm2(x + y)

            out_list.append(out)
            attn_list.append(attn)

        return out_list, attn_list

class Encoder(nn.Module):
    def __init__(self, attn_layers, freq_layers, norm_layer):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.freq_layers = nn.ModuleList(freq_layers)
        self.norm = norm_layer
        
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []

        for l, (attn_layer, freq_layer) in enumerate(
            zip(self.attn_layers, self.freq_layers)
        ):
            delta = delta if l == 0 else None
            x = freq_layer(x)
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        x = torch.cat(x, dim=1) 
        x = self.norm(x)
        return x, attns


