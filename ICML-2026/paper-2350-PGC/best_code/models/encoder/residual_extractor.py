import torch
import torch.nn as nn


# RGB -> YCbCr (BT.601) transform; matches paper Eq. 1 exactly.
M_YUV = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=torch.float32,
)

M_YUV_INV = torch.tensor(
    [
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ],
    dtype=torch.float32,
)


class QuantizationResidualExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer("M_t", M_YUV)
        self.register_buffer("M_t_inv", M_YUV_INV)

    def forward(self, x):
        """Compute the residual map.

        Args:
            x: ``[B, 3, H, W]`` tensor in ``[0, 1]`` range (post-ToTensor).

        Returns:
            ``[B, 3, H, W]`` residual tensor, clamped to ``[-1, 1]``.
        """
        x_scaled = x * 255.0
        x_permuted = x_scaled.permute(0, 2, 3, 1)  # [B, H, W, 3]

        x_yuv = torch.matmul(x_permuted, self.M_t.T)
        x_quantized = torch.round(x_yuv)
        x_reconstructed = torch.matmul(x_quantized, self.M_t_inv.T)

        residual = x_permuted - x_reconstructed
        residual = residual.permute(0, 3, 1, 2)

        residual = torch.clamp(residual, -1.0, 1.0)
        return residual


def get_residual_extractor():
    return QuantizationResidualExtractor()
