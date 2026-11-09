import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal
from torchaudio.functional import bandpass_biquad, lfilter
from tqdm import tqdm

from src.config import INPUT_PATH


def load_file(id_, folder="train"):
    path = INPUT_PATH / folder / id_[0] / id_[1] / id_[2] / f"{id_}.npy"
    waves = np.load(path)
    return waves


def build_design_curves(window=("tukey", 0.2)):
    df = (
        pd.read_csv(INPUT_PATH / "training_labels.csv")
        .query("target == 0")
        .reset_index(drop=True)
    )
    hanford = 0
    livingston = 0
    virgo = 0
    n = len(df)
    window = ("tukey", 0.2)

    for i in tqdm(df["id"]):
        data = load_file(i)
        hanford += signal.periodogram(data[0], fs=2048, window=window)[1]
        livingston += signal.periodogram(data[1], fs=2048, window=window)[1]
        virgo += signal.periodogram(data[2], fs=2048, window=window)[1]

    hanford /= n
    livingston /= n
    virgo /= n

    design_curves = np.stack([hanford, livingston, virgo])[:, :-1] ** 0.5
    fname = "design_curves_"
    if type(window) == tuple:
        fname += "_".join(str(x) for x in window)
    else:
        fname += window

    np.save(INPUT_PATH / fname, design_curves)
    return design_curves


def get_design_curves(window=("tukey", 0.2)):
    fname = "design_curves_"
    if type(window) == tuple:
        fname += "_".join(str(x) for x in window)
    else:
        fname += window

    if (INPUT_PATH / f"{fname}.npy").exists():
        return np.load(INPUT_PATH / f"{fname}.npy")
    else:
        print(f"{fname}.npy not found. Building design curves")
        return build_design_curves(window)


def apply_whiten(signal, design_curves):
    """Whitens a waveform according to a design curve

    Args:
        signal (tensor): A waveform with a window already applied
        design_curves (tensor): Design curves for the window used

    Returns:
        tensor: Whitened waveform scaled between -1 and +1
    """
    spec = torch.fft.fft(signal)
    n = signal.shape[-1]
    dc_len = design_curves.shape[-1]
    whitened = torch.real(torch.fft.ifft(spec[:, :dc_len] / design_curves, n=n))
    whitened *= np.sqrt(n / 2)
    whitened /= torch.max(torch.abs(whitened), axis=1)[0].reshape(3, 1)
    return whitened.to(dtype=torch.float32)


# Cell 33 of https://www.gw-openscience.org/LVT151012data/LOSC_Event_tutorial_LVT151012.html
# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def apply_bandpass(x, lf=25, hf=500, order=4, sr=2048):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    return signal.sosfiltfilt(sos, x) / normalization


def apply_lowpass(x, lf=25, order=4, sr=2048):
    sos = signal.butter(order, lf, btype="lowpass", output="sos", fs=sr)
    return signal.sosfiltfilt(sos, x)


def apply_notch(x, w0=306, q=None):

    if q is None:
        q = w0  # 1 Hz bandwidth

    b, a = signal.iirnotch(w0, q, fs=2048)
    return signal.filtfilt(b, a, x)


# https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/265367#1476566
def biquad_bandpass_filter(data, lowcut, highcut, fs):
    return bandpass_biquad(
        data, fs, (highcut + lowcut) / 2, (highcut - lowcut) / (highcut + lowcut)
    )


class BiquadBandpass(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        central_freq: float,
        Q: float = 0.707,
        const_skirt_gain: bool = False,
        trainable: bool = False,
    ):
        super().__init__()
        central_freq = torch.as_tensor(central_freq)
        Q = torch.as_tensor(Q)

        w0 = 2 * np.pi * central_freq / sample_rate
        alpha = torch.sin(w0) / 2 / Q

        temp = torch.sin(w0) / 2 if const_skirt_gain else alpha
        self.b0 = torch.as_tensor(temp).view(1)
        self.b1 = torch.as_tensor(0.0).view(1)
        self.b2 = torch.as_tensor(-temp).view(1)
        self.a0 = torch.as_tensor(1 + alpha).view(1)
        self.a1 = torch.as_tensor(-2 * torch.cos(w0)).view(1)
        self.a2 = torch.as_tensor(1 - alpha).view(1)

        if trainable:
            self.b0 = nn.Parameter(self.b0, requires_grad=True)
            self.b1 = nn.Parameter(self.b1, requires_grad=True)
            self.b2 = nn.Parameter(self.b2, requires_grad=True)
            self.a0 = nn.Parameter(self.a0, requires_grad=True)
            self.a1 = nn.Parameter(self.a1, requires_grad=True)
            self.a2 = nn.Parameter(self.a2, requires_grad=True)
            self.register_parameter("b0", self.b0)
            self.register_parameter("b1", self.b1)
            self.register_parameter("b2", self.b2)
            self.register_parameter("a0", self.a0)
            self.register_parameter("a1", self.a1)
            self.register_parameter("a2", self.a2)

    def forward(self, waveform):
        output_waveform = lfilter(
            waveform,
            torch.cat([self.a0, self.a1, self.a2]),
            torch.cat([self.b0, self.b1, self.b2]),
        )

        return output_waveform
