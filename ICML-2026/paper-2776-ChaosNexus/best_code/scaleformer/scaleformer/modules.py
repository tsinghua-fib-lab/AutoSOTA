import torch
import torch.nn as nn
from transformers import PatchTSTConfig


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class PatchTSTKernelEmbedding(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        poly_degrees_lst = range(2, 2 + config.poly_degrees)
        assert (
            config.patch_length
            + len(poly_degrees_lst) * config.num_poly_feats
            + config.num_rff
            == config.d_model
        ), (
            f"Sum of features must equal d_model: d_poly + d_rff + patch_length = "
            f"{len(poly_degrees_lst) * config.num_poly_feats} + {config.num_rff}"
            f" + {config.patch_length} != {config.d_model}"
        )
        self.num_poly_feats = config.num_poly_feats
        self.patch_indices = [
            torch.randint(
                high=config.patch_length,
                size=(self.num_poly_feats, d),
                requires_grad=False,
            )
            for d in poly_degrees_lst
        ]
        self.freq_weights = nn.Parameter(
            config.rff_scale * torch.randn(config.patch_length, config.num_rff // 2),
            requires_grad=config.rff_trainable,
        )
        self.freq_biases = nn.Parameter(
            torch.randn(1, 1, 1, config.num_rff // 2),
            requires_grad=config.rff_trainable,
        )
        self.projection = nn.Linear(config.d_model, config.d_model, bias=False)
        # self.projection = nn.Linear(
        #     config.patch_length
        #     + config.num_rff
        #     + len(self.patch_indices) * config.num_poly_feats,
        #     config.d_model,
        #     bias=False,
        # )
        # self.projection = nn.Linear(
        #     3
        #     * (
        #         config.patch_length - 1
        #     )  # for x, and its 1st and 2nd order finite differences
        #     + len(self.patch_indices) * config.num_poly_feats
        #     + config.num_rff,
        #     config.d_model,
        #     bias=False,
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input for embedding
        return:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, d_model)`
        """
        # centered difference & polynomial features
        # o1_cdiff = x[..., 1:] - x[..., :-1]
        # o2_cdiff = o1_cdiff[..., 1:] - o1_cdiff[..., :-1]
        # cdiff_feats = torch.cat([x, o1_cdiff, o2_cdiff], dim=-1)
        # poly_feats = [cdiff_feats[..., pis].prod(dim=-1) for pis in self.patch_indices]

        poly_feats = [x[..., pis].prod(dim=-1) for pis in self.patch_indices]

        weighted_x = x @ self.freq_weights + self.freq_biases
        rff_feats = torch.cat([torch.sin(weighted_x), torch.cos(weighted_x)], dim=-1)

        # features = torch.cat([cdiff_feats, *poly_feats, rff_feats], dim=-1)
        features = torch.cat([x, *poly_feats, rff_feats], dim=-1)
        features = self.projection(features)
        return features


class PatchTSTPolynomialEmbedding(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.poly_degrees = [2, 3]
        self.inner_dim = config.d_model
        self.num_poly_feats = 256

        self.poly_weights_q = nn.ModuleList(
            [
                nn.Linear(config.patch_length, self.inner_dim, bias=False)
                for _ in self.poly_degrees
            ]
        )
        self.poly_weights_k = nn.ModuleList(
            [
                nn.Linear(config.patch_length, self.inner_dim, bias=False)
                for _ in self.poly_degrees
            ]
        )
        self.poly_weights_v = nn.ModuleList(
            [
                nn.Linear(config.patch_length, self.num_poly_feats, bias=False)
                for _ in self.poly_degrees
            ]
        )

        self.freq_weights = nn.Parameter(
            config.rff_scale * torch.randn(config.patch_length, config.num_rff // 2),
            requires_grad=config.rff_trainable,
        )
        self.freq_biases = nn.Parameter(
            torch.randn(1, 1, 1, config.num_rff // 2),
            requires_grad=config.rff_trainable,
        )
        self.projection = nn.Linear(
            config.patch_length
            + config.num_rff
            + self.num_poly_feats * len(self.poly_degrees),
            config.d_model,
        )

    def polyattn(self, x: torch.Tensor, degree_idx: int) -> torch.Tensor:
        """
        Parameters:
            x (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input for embedding
            degree_idx (int, *required*):
                Index corresponding to the degree in `self.poly_degrees` (0, 1, 2, ...)
        return:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, num_poly_feats)`
        """
        degree = self.poly_degrees[degree_idx] - 2
        # shape: (batch_size, num_patches, num_channels, patch_length)
        x = x.transpose(1, 2)
        qproj, kproj, vproj = (
            self.poly_weights_q[degree_idx],
            self.poly_weights_k[degree_idx],
            self.poly_weights_v[degree_idx],
        )
        Q = qproj(x**degree)
        K = kproj(x)
        # shape: (batch_size, num_patches, num_channels, num_poly_feats)
        polyfeats = (Q @ K.transpose(-2, -1)) @ vproj(x)
        return polyfeats.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input for embedding
        return:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, d_model)`
        """
        polyfeats = torch.cat(
            [self.polyattn(x, d) for d in range(len(self.poly_degrees))], dim=-1
        )
        weighted_x = x @ self.freq_weights + self.freq_biases
        rff_feats = torch.cat([torch.sin(weighted_x), torch.cos(weighted_x)], dim=-1)
        features = torch.cat([x, polyfeats, rff_feats], dim=-1)
        return self.projection(features)


class PatchTSTRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Stolen from Llama
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class PatchTSTPatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    NOTE: Exposed from original source code. Allow for variable sequence length

    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.sequence_length = config.context_length
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride

        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        sequence_length = past_values.shape[-2]
        num_patches = (sequence_length - self.patch_length) // self.patch_stride + 1
        new_sequence_length = self.patch_length + self.patch_stride * (num_patches - 1)
        sequence_start = sequence_length - new_sequence_length

        # output: [bs x new_sequence_length x num_channels]
        output = past_values[:, sequence_start:, :]
        # output: [bs x num_patches x num_input_channels x patch_length]
        output = output.unfold(
            dimension=-2, size=self.patch_length, step=self.patch_stride
        )
        # output: [bs x num_input_channels x num_patches x patch_length]
        output = output.transpose(-2, -3).contiguous()
        return output


def apply_p_rope_to_qk(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    max_wavelength: int,
    rope_percent: float,
):
    """
    Apply p-rotary positional embeddings to the query and key tensors

    from: https://arxiv.org/pdf/2410.06205
    """
    rope_angles = int(rope_percent * head_dim // 2)
    nope_angles = head_dim // 2 - rope_angles
    fraction = (
        2.0
        * torch.arange(
            0, rope_angles, device=query_states.device, dtype=query_states.dtype
        )
        / head_dim
    )
    timescale = torch.nn.functional.pad(
        max_wavelength**fraction,
        (0, nope_angles),
        mode="constant",
        value=float('inf'), # 之前的修复，保持不变
    )

    # sin, cos: shape (..., 1, seq_len, head_dim//2)
    sinusoid_inp = position_ids[..., None, :, None] / timescale[None, None, :]
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)

    # ==================== 新增的关键修复 ====================
    # 将 sin 和 cos 强制转换回 query/key 的数据类型
    # 因为三角函数计算可能为了精度而内部使用了 float32
    cos = cos.to(query_states.dtype)
    sin = sin.to(query_states.dtype)
    # ======================================================

    query_first_half, query_second_half = torch.split(
        query_states, query_states.shape[-1] // 2, dim=-1
    )
    key_first_half, key_second_half = torch.split(
        key_states, key_states.shape[-1] // 2, dim=-1
    )

    query_first_part = query_first_half * cos - query_second_half * sin
    query_second_part = query_second_half * cos + query_first_half * sin

    key_first_part = key_first_half * cos - key_second_half * sin
    key_second_part = key_second_half * cos + key_first_half * sin

    query_states = torch.cat([query_first_part, query_second_part], dim=-1)
    key_states = torch.cat([key_first_part, key_second_part], dim=-1)

    return query_states, key_states


class PatchTSTFourierApproximator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, timeseries: torch.Tensor, k: int) -> torch.Tensor:
        """
        Use top k modes of the Fourier transform to approximate the timeseries

        Parameters:
            timeseries (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Patch input for embedding
            k (int, *required*):
                Number of modes to use

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`
        """

        batch_size, seq_length, n_channels = timeseries.shape
        # Vectorized FFT applied on sequence length dimension
        ffts = torch.fft.rfft(timeseries, axis=1)  # Shape: (batch_size, n_freqs, 3)
        # Get indices of top k modes by magnitude
        magnitudes = torch.abs(ffts)
        # Shape: (batch_size, k, 3)
        top_k_indices = torch.argsort(magnitudes, dim=1)[:, -k:, :]
        # Zero out all but top k modes
        filtered_ffts = torch.zeros_like(ffts)

        for b in range(batch_size):
            for i in range(n_channels):
                filtered_ffts[b, top_k_indices[b, :, i], i] = ffts[
                    b, top_k_indices[b, :, i], i
                ]

        # Vectorized inverse transform
        reconstructed = torch.fft.irfft(filtered_ffts, seq_length, dim=1)
        return reconstructed
