import timm
import torch
import torch.nn as nn
from nnAudio import Spectrogram
from scipy import signal
import torch.nn.functional as F
from bisect import bisect
import numpy as np


class Model_2D(nn.Module):
    def __init__(self, encoder='resnet', use_raw_wave=False, avrSpecDir="/home/data/", fmin=15, cut_612=False):
        super().__init__()
        self.encoder = timm.create_model(
            encoder,
            pretrained=True,
            num_classes=1,  # 0 = feature extraction
            in_chans=4,
        )
        self.window = nn.Parameter(torch.FloatTensor(signal.windows.tukey(4096 + 2 * 2048, 0.5)), requires_grad=False)
        self.avr_spec = nn.Parameter(torch.load(avrSpecDir+"avr_w0.pth"), requires_grad=False)
        self.spec_transform = Spectrogram.CQT1992v2(sr=2048, fmin=fmin, n_bins=64, hop_length=32,
                                                    output_format='Magnitude', norm=1, bins_per_octave=12,
                                                    window='nuttall')
        self.cut_612 = cut_612
        self.cut_place = None
        if self.cut_612:
            print("Cut 612 frequency range")
            freqs = 22 * 2.0 ** (np.r_[0:64] / np.float(12))
            self.cut_place = bisect(freqs, 612)

        self.use_raw_wave = use_raw_wave
        self.n_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(self.n_features, 1)

    def frequency_encoding(self, x):
        device = x.device
        bs, fbins, t = x.shape[0], x.shape[2], x.shape[3]
        freq_encoding = 2 * torch.arange(fbins, device=device) / fbins - 1  # -1 to +1
        freq_encoding = torch.stack([freq_encoding] * t, -1).unsqueeze(0)
        freq_encoding = torch.stack([freq_encoding] * bs)
        return torch.cat([x, freq_encoding], 1)

    def forward(self, x):
        if self.use_raw_wave:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    shape = x.shape
                    c = x.view(shape[0] * shape[1], -1)
                    c = torch.cat([-c.flip(-1)[:, 4096 - 2049:-1] + 2 * c[:, 0].unsqueeze(-1), c,
                                   -c.flip(-1)[:, 1:2049] + 2 * c[:, -1].unsqueeze(-1)], 1)
                    avr_spec = self.avr_spec.repeat(shape[0], 1).view(-1, self.avr_spec.shape[-1])
                    x = torch.fft.ifft(torch.fft.fft(c * self.window) / avr_spec).real
                    x = self.spec_transform(x)
                    x = x.reshape(shape[0], shape[1], x.shape[1], x.shape[2])
                    x = x[:, :, :, 64 + 64 - 8:192 - 8]
                    if self.cut_612:
                        x = torch.cat([x[:, :, :self.cut_place, :], x[:, :, self.cut_place + 1:, :]], 2)
                    x = (8.0 * x + 1.0).log()
                    x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
                    # spec = standard_scaler(spec)
                    x = self.frequency_encoding(x)

        return self.encoder(x)
