import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousWaveletTransform(nn.Module):
    """GPU accelerated continuous wavelet transform
    Implementation is based on: 
    https://github.com/Kevin-McIsaac/cmorlet-tensorflow/blob/master/cwt.py
    Args:
        n_scales: (int) Number of scales for the scalogram.
        border_crop: (int) Non-negative integer that specifies the number
            of samples to be removed at each border after computing the cwt.
            This parameter allows to input a longer signal than the final
            desired size to remove border effects of the CWT. Default 0.
        stride: (int) The stride of the sliding window across the input.
            Default is 1.
    """
    def __init__(self, n_scales, border_crop=0, stride=1):
        super().__init__()
        self.n_scales = n_scales
        self.border_crop = border_crop
        self.stride = stride
        self._build_wavelet_bank()

    def _build_wavelet_bank(self):
        real_part = None
        imaginary_part = None
        return real_part, imaginary_part

    def forward(self, inputs):
        """Computes the CWT with the specified wavelet bank.
        If the signal has more than one channel, the CWT is computed for
        each channel independently and stacked at the end along the
        channel axis.
        Args:
            inputs: (tensor) A batch of 1D tensors of shape
                [batch_size, time_len].
        Returns:
            Scalogram magnitude tensor for each input channels.
            The shape of this tensor is
            [batch_size, n_channels, n_scales, time_len]
        """

        # Generate the scalogram
        border_crop = int(self.border_crop / self.stride)
        start = border_crop
        end = (-border_crop) if (border_crop > 0) else None
        
        # Input has expected shape of [batch_size, time_len]
        # Reshape input [batch, time_len] -> [batch, 1, time_len, 1]
        inputs_expand = inputs.unsqueeze(1).unsqueeze(3)
        out_real = F.conv2d(
            input=inputs_expand, weight=self.real_part,
            stride=(self.stride, 1), 
            padding=(self.real_part.shape[2]//2, self.real_part.shape[3]//2))
        out_real = out_real[:, :, start:end, :].squeeze(3)
        if self.imaginary_part is not None:
            out_imag = F.conv2d(
                input=inputs_expand, weight=self.imaginary_part,
                stride=(self.stride, 1), 
                padding=(self.imaginary_part.shape[2]//2, self.imaginary_part.shape[3]//2))
            out_imag = out_imag[:, :, start:end, :].squeeze(3)
            # [batch, n_scales, time_len]
            scalogram = torch.sqrt(out_real ** 2 + out_imag ** 2)
        else:
            scalogram = out_real
        return scalogram


class ComplexMorletCWT(ContinuousWaveletTransform):
    """Computes the complex morlet wavelets
    Args:
        wavelet_width: (float o tensor) wavelet width.
        fs: (float) Sampling frequency of the application.
        lower_freq: (float) Lower frequency of the scalogram.
        upper_freq: (float) Upper frequency of the scalogram.
        n_scales: (int) Number of scales for the scalogram.
        size_factor: (float) Factor by which the size of the kernels will
            be increased with respect to the original size. Default 1.0.
        trainable: (boolean) If True, the wavelet width is trainable.
            Default to False.
        border_crop: (int) Non-negative integer that specifies the number
            of samples to be removed at each border after computing the cwt.
            This parameter allows to input a longer signal than the final
            desired size to remove border effects of the CWT. Default 0.
        stride: (int) The stride of the sliding window across the input.
            Default is 1.
    """
    def __init__(
            self,
            wavelet_width,
            fs,
            lower_freq,
            upper_freq,
            n_scales,
            size_factor=1.0,
            trainable_width=False,
            trainable_filter=False,
            border_crop=0,
            stride=1):
        # Checking
        if lower_freq > upper_freq:
            raise ValueError("lower_freq should be lower than upper_freq")
        if lower_freq < 0:
            raise ValueError("Expected positive lower_freq.")

        self.initial_wavelet_width = wavelet_width
        self.fs = fs
        self.lower_freq = lower_freq
        self.upper_freq = upper_freq
        self.size_factor = size_factor
        self.trainable_width = trainable_width
        self.trainable_filter = trainable_filter
        # Generate initial and last scale
        s_0 = 1 / self.upper_freq
        s_n = 1 / self.lower_freq
        # Generate the array of scales
        base = np.power(s_n / s_0, 1 / (n_scales - 1))
        self.scales = torch.from_numpy(s_0 * np.power(base, np.arange(n_scales)))
        # Generate the frequency range
        self.frequencies = 1 / self.scales
        super().__init__(n_scales, border_crop, stride)
        

    def _build_wavelet_bank(self):
        # Generate the wavelets
        self.wavelet_width = nn.Parameter(
            data=torch.tensor(
                self.initial_wavelet_width, dtype=torch.float32),
            requires_grad=self.trainable_width)
        # We will make a bigger wavelet in case the width grows
        # For the size of the wavelet we use the initial width value.
        # |t| < truncation_size => |k| < truncation_size * fs
        truncation_size = self.scales.max() * np.sqrt(4.5 * self.initial_wavelet_width) * self.fs
        one_side = int(self.size_factor * truncation_size)
        kernel_size = 2 * one_side + 1
        k_array = np.arange(kernel_size, dtype=np.float32) - one_side
        t_array = k_array / self.fs  # Time units
        # Wavelet bank shape: 1, kernel_size, 1, n_scales
        wavelet_bank_real = []
        wavelet_bank_imag = []
        for scale in self.scales:
            norm_constant = torch.sqrt(np.pi * self.wavelet_width) * scale * self.fs / 2.0
            scaled_t = t_array / scale
            exp_term = torch.exp(-(scaled_t ** 2) / self.wavelet_width)
            kernel_base = exp_term / norm_constant
            kernel_real = kernel_base * np.cos(2 * np.pi * scaled_t)
            kernel_imag = kernel_base * np.sin(2 * np.pi * scaled_t)
            wavelet_bank_real.append(kernel_real)
            wavelet_bank_imag.append(kernel_imag)
        # Stack wavelets (shape = n_scales, kernel_size)
        wavelet_bank_real = torch.stack(wavelet_bank_real, axis=0)
        wavelet_bank_imag = torch.stack(wavelet_bank_imag, axis=0)
        # Give it proper shape for convolutions
        # -> shape: n_scales, 1, kernel_size, 1
        wavelet_bank_real = wavelet_bank_real.unsqueeze(1).unsqueeze(3)
        wavelet_bank_imag = wavelet_bank_imag.unsqueeze(1).unsqueeze(3)
        self.real_part = nn.Parameter(
            data=wavelet_bank_real,
            requires_grad=self.trainable_filter)
        self.imaginary_part = nn.Parameter(
            data=wavelet_bank_imag,
            requires_grad=self.trainable_filter)


class RickerCWT(ContinuousWaveletTransform):
    """Computes the ricker wavelets
    Args:
    """
    def __init__(
            self,
            fs, 
            lower_freq, 
            upper_freq, 
            n_scales,
            size_factor=1.0,
            trainable_filter=False,
            border_crop=0,
            stride=1):
        self.fs = fs
        self.lower_freq = lower_freq
        self.upper_freq = upper_freq
        self.size_factor = size_factor
        self.trainable_filter = trainable_filter
        # Generate initial and last scale
        s_0 = 1 / self.upper_freq
        s_n = 1 / self.lower_freq
        # Generate the array of scales
        base = np.power(s_n / s_0, 1 / (n_scales - 1))
        self.scales = torch.from_numpy(s_0 * np.power(base, np.arange(n_scales)))
        # Generate the frequency range
        self.frequencies = 1 / self.scales
        super().__init__(n_scales, border_crop, stride)
        

    def _build_wavelet_bank(self):
        truncation_size = self.scales.max() * np.sqrt(4.5) * self.fs
        one_side = int(self.size_factor * truncation_size)
        kernel_size = 2 * one_side + 1
        k_array = np.arange(kernel_size, dtype=np.float32) - one_side
        t_array = k_array / self.fs  # Time units
        wavelet_bank_real = []
        for scale in self.scales:
            A = 2 / (np.sqrt(3 * scale) * (np.pi ** 0.25))
            scaled_t = t_array / scale
            mod = 1 - scaled_t ** 2
            gauss = np.exp(-1 * t_array ** 2 / (2 * scale ** 2))
            kernel_real = A * mod * gauss
            wavelet_bank_real.append(kernel_real)
        wavelet_bank_real = torch.stack(wavelet_bank_real, axis=0)
        wavelet_bank_real = wavelet_bank_real.unsqueeze(1).unsqueeze(3)
        self.real_part = nn.Parameter(
            data=wavelet_bank_real,
            requires_grad=self.trainable_filter)
        self.imaginary_part = None
