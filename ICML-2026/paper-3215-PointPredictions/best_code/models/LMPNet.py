import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # 1. Decomposition (Manifold Pre-processing)
        kernel_size = getattr(configs, 'moving_avg', 25)
        self.decomposition = SeriesDecomp(kernel_size)

        # 2. Latent Dimension (Manifold Capacity)
        # d_model >> seq_len (e.g., 1024) to capture rich semantics via Expansion
        self.d_model = getattr(configs, 'd_model', 512)
        self.dropout = nn.Dropout(configs.dropout)

        # 3. Low-Frequency Branch (Trend Dynamics)
        # Keeps simple linear mapping for robustness
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # 4. High-Frequency Branch (Manifold Dynamics)
        # --- The Core of LMP-Net ---


        # Maps low-dim time series to high-dim latent space
        # [Length] -> [Latent]
        self.manifold_expander = nn.Linear(self.seq_len, self.d_model)

        # Non-linearity is key for "Manifold" modeling (Cover's Theorem)
        self.act = nn.GELU()


        # Evolving features in the high-dimensional manifold
        self.e_layers = getattr(configs, 'e_layers', 1)
        if self.e_layers > 1:
            self.latent_dynamics = nn.ModuleList([
                nn.Linear(self.d_model, self.d_model)
                for _ in range(self.e_layers - 1)
            ])
        else:
            self.latent_dynamics = None


        # Maps latent features back to prediction horizon
        # [Latent] -> [Pred_Len]
        self.manifold_compressor = nn.Linear(self.d_model, self.pred_len)

        # 5. RevIN (Normalization)
        self.rev_in = True if getattr(configs, 'rev_in', 1) else False
        if self.rev_in:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, return_feature=False):
        # x: [Batch, Input_Len, Channel]

        # --- 1. RevIN (Pre-norm) ---
        if self.rev_in:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)
            x = x * self.affine_weight + self.affine_bias

        # --- 2. Decomposition ---
        seasonal_init, trend_init = self.decomposition(x)

        # Permute: [B, L, C] -> [B, C, L]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # --- 3. Low-Freq Dynamics (Trend) ---
        trend_output = self.linear_trend(trend_init)

        # --- 4. Manifold Dynamics (Seasonal/High-Freq) ---

        # Step A: Expansion (W_exp)
        # Project to Latent Manifold: [B, C, d_model]
        latent_feat = self.manifold_expander(seasonal_init)
        latent_feat = self.act(latent_feat)
        latent_feat = self.dropout(latent_feat)

        # Step B: Evolution (Dynamics)
        # Non-linear dynamics in latent space
        if self.latent_dynamics is not None:
            for layer in self.latent_dynamics:
                latent_feat = layer(latent_feat)
                latent_feat = self.act(latent_feat)
                latent_feat = self.dropout(latent_feat)

        # Step C: Compression (W_comp)
        # Map back to Horizon: [B, C, Pred_Len]
        seasonal_output = self.manifold_compressor(latent_feat)

        # --- 5. Summation ---
        x_out = seasonal_output + trend_output
        x_out = x_out.permute(0, 2, 1)  # Back to [B, P, C]

        # --- 6. RevIN (De-norm) ---
        if self.rev_in:
            x_out = (x_out - self.affine_bias) / self.affine_weight
            x_out = x_out * torch.sqrt(seq_var) + seq_mean

        # --- 7. Feature Return for Manifold Alignment ---
        if return_feature:
            # Return the "Evolved Latent Features" (before compression)
            # Shape: [B, C, d_model] -> Perfect for Gram Matrix / OT
            return x_out, latent_feat

        return x_out