import torch
import torch.nn as nn
from nnAudio import Spectrogram
from scipy import signal
import torch.nn.functional as F

class Combined1D2D(nn.Module):
    def __init__(self, model_1d, model_2d, emb_1d=128, emb_2d=128, first=512, ps=0.5, avrSpecDir="/home/data/"):
        super().__init__()
        self.model_1d = model_1d
        self.model_2d = model_2d

        self.window = nn.Parameter(torch.FloatTensor(signal.windows.tukey(4096 + 2 * 2048, 0.5)), requires_grad=False)
        self.avr_spec = nn.Parameter(torch.load(avrSpecDir+"avr_w0.pth"), requires_grad=False)
        self.spec_transform = Spectrogram.CQT1992v2(sr=2048, fmin=15, n_bins=64, hop_length=32,
                                                    output_format='Magnitude', norm=1, bins_per_octave=12,
                                                    window='nuttall')

        # Replace last linear layer to return a embedding of size emb_1d
        head = list(self.model_1d.head.children())
        new_linear = nn.Linear(head[-1].in_features, emb_1d)
        self.model_1d.head = nn.Sequential(*head[:-1] + [new_linear])

        # Replace last linear layer to return a embedding of size emb_2d
        old_head = self.model_2d.encoder.fc
        self.model_2d.encoder.fc = nn.Linear(old_head.in_features, emb_2d)

        self.head = nn.Sequential(
            nn.Linear(emb_1d + emb_2d, first), nn.BatchNorm1d(first), nn.Dropout(ps/2), nn.SiLU(inplace=True),
            nn.Linear(first, first//2), nn.BatchNorm1d(first//2), nn.Dropout(ps), nn.SiLU(inplace=True),
            nn.Linear(first//2, 1),
        )

    def freeze_conv(self, req_grad):
        for name, param in self.model_2d.named_parameters():
            if 'encoder.fc.' in name:
                continue
            param.requires_grad = req_grad

        for name, param in self.model_1d.named_parameters():
            if 'head.9.' in name:
                continue
            param.requires_grad = req_grad

    def frequency_encoding(self, x):
        # for 2D model
        device = x.device
        bs, fbins, t = x.shape[0], x.shape[2], x.shape[3]
        freq_encoding = 2 * torch.arange(fbins, device=device) / fbins - 1  # -1 to +1
        freq_encoding = torch.stack([freq_encoding] * t, -1).unsqueeze(0)
        freq_encoding = torch.stack([freq_encoding] * bs)
        return torch.cat([x, freq_encoding], 1)

    def forward(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                shape = x.shape
                c = x.view(shape[0] * shape[1], -1)
                c = torch.cat([-c.flip(-1)[:, 4096 - 2049:-1] + 2 * c[:, 0].unsqueeze(-1), c,
                               -c.flip(-1)[:, 1:2049] + 2 * c[:, -1].unsqueeze(-1)], 1)
                avr_spec = self.avr_spec.repeat(shape[0], 1).view(-1, self.avr_spec.shape[-1])
                x = torch.fft.ifft(torch.fft.fft(c * self.window) / avr_spec).real
                x_1d = x.view(shape[0], shape[1], x.shape[-1])[:, :, 2048:-2048]

                x_2d = self.spec_transform(x)
                x_2d = x_2d.reshape(shape[0], shape[1], x_2d.shape[1], x_2d.shape[2])
                x_2d = x_2d[:, :, :, 64 + 64 - 8:192 - 8]
                x_2d = (8.0 * x_2d + 1.0).log()
                x_2d = F.interpolate(x_2d, size=(256, 256), mode='bilinear', align_corners=True)
                # spec = standard_scaler(spec)
                x_2d = self.frequency_encoding(x_2d)

        out_1d = self.model_1d(x_1d)
        out_2d = self.model_2d(x_2d)
        embedding = torch.cat([out_1d, out_2d], -1)
        return self.head(embedding)
