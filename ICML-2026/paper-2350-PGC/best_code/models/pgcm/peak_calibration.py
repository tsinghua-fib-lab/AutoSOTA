import torch
import torch.nn as nn


class PeakGuidedCalibrationModule(nn.Module):

    def __init__(
        self,
        rgb_dim: int,
        residual_dim: int,
        tau_rgb: float = 0.5,
        tau_res: float = 0.5,
    ):
        super().__init__()

        # Phi_Linear: maps each [D]-dim patch token to a scalar score.
        self.rgb_score_head = nn.Linear(rgb_dim, 1)
        nn.init.zeros_(self.rgb_score_head.weight)
        nn.init.zeros_(self.rgb_score_head.bias)

        # Phi_Conv: 1x1 conv reducing residual feature map to a 1-channel score map.
        self.residual_score_head = nn.Conv2d(
            residual_dim, 1, kernel_size=1, stride=1, padding=0
        )
        nn.init.zeros_(self.residual_score_head.weight)
        if self.residual_score_head.bias is not None:
            nn.init.zeros_(self.residual_score_head.bias)

        # Learnable scalar modulating the RGB aggregate (paper Eq. 4).
        self.lambda_rgb = nn.Parameter(torch.tensor(1.0))

        # Fixed temperatures (paper hyper-parameters, not learned).
        if tau_rgb <= 0.0 or tau_res <= 0.0:
            raise ValueError(
                f"tau_rgb and tau_res must be > 0, got {tau_rgb} / {tau_res}"
            )
        self.tau_rgb = float(tau_rgb)
        self.tau_res = float(tau_res)

    @staticmethod
    def peak_aggregation(scores: torch.Tensor, tau: float) -> torch.Tensor:

        if scores.dim() != 2:
            raise ValueError(f"Expected [B, N] scores, got {list(scores.shape)}")

        n = scores.size(1)
        if n == 0:
            # Empty patch grid: aggregate is zero (no evidence).
            return scores.new_zeros((scores.size(0), 1))

        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}")

        log_n = scores.new_tensor(float(n)).log()
        z = tau * (torch.logsumexp(scores / tau, dim=1) - log_n)
        return z.unsqueeze(1)

    def forward(
        self,
        f_rgb_tokens: torch.Tensor,
        f_residual_map: torch.Tensor,
    ) -> torch.Tensor:

        # ---- RGB stream: per-patch scores ----
        s_rgb = self.rgb_score_head(f_rgb_tokens).squeeze(-1)   # [B, N]
        z_rgb = self.peak_aggregation(s_rgb, self.tau_rgb)      # [B, 1]

        # ---- Residual stream: per-spatial-location scores ----
        s_res = self.residual_score_head(f_residual_map).flatten(1)   # [B, Hr*Wr]
        z_res = self.peak_aggregation(s_res, self.tau_res)            # [B, 1]

        # ---- Fusion (paper Eq. 4) ----
        z_local = z_res + self.lambda_rgb * z_rgb
        return z_local
