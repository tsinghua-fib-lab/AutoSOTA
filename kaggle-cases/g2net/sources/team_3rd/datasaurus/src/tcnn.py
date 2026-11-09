import timm
import torch
import torch.nn as nn
from nnAudio import Spectrogram

from scalers import standard_scaler


class GeM(nn.Module):
    """
    Code modified from the 2d code in
    https://amaarora.github.io/2020/08/30/gempool.html
    """

    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        with torch.cuda.amp.autocast(enabled=False):  # to avoid NaN issue for fp16
            return nn.functional.avg_pool1d(
                x.clamp(min=eps).pow(p), self.kernel_size
            ).pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ResBlockGeM(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        downsample=1,
        act=nn.SiLU(inplace=True),
    ):
        super().__init__()
        self.act = act
        if downsample != 1 or in_channels != out_channels:
            self.residual_function = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                act,
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                GeM(kernel_size=downsample),  # downsampling
            )
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                GeM(kernel_size=downsample),  # downsampling
            )  # skip layers in residual_function, can try simple MaxPool1d
        else:
            self.residual_function = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                act,
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.act(self.residual_function(x) + self.shortcut(x))


class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`"

    def __init__(self, size=None):
        super().__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool1d(self.size)
        self.mp = nn.AdaptiveMaxPool1d(self.size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Extractor(nn.Sequential):
    def __init__(
        self, in_c=8, out_c=8, kernel_size=64, maxpool=8, act=nn.SiLU(inplace=True)
    ):
        super().__init__(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_c),
            act,
            nn.Conv1d(out_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2),
            # nn.MaxPool1d(kernel_size=maxpool),
            GeM(kernel_size=maxpool),
        )


class ModelIafossV2(nn.Module):
    def __init__(self, n=8, nh=256, act=nn.SiLU(inplace=True), ps=0.5):
        super().__init__()
        self.ex = nn.ModuleList(
            [
                nn.Sequential(
                    Extractor(1, n, 127, maxpool=2, act=act),
                    ResBlockGeM(n, n, kernel_size=31, downsample=4, act=act),
                    ResBlockGeM(n, n, kernel_size=31, act=act),
                ),
                nn.Sequential(
                    Extractor(1, n, 127, maxpool=2, act=act),
                    ResBlockGeM(n, n, kernel_size=31, downsample=4, act=act),
                    ResBlockGeM(n, n, kernel_size=31, act=act),
                ),
            ]
        )
        self.conv1 = nn.ModuleList(
            [
                nn.Sequential(
                    ResBlockGeM(
                        1 * n, 1 * n, kernel_size=31, downsample=4, act=act
                    ),  # 512
                    ResBlockGeM(1 * n, 1 * n, kernel_size=31, act=act),
                ),
                nn.Sequential(
                    ResBlockGeM(
                        1 * n, 1 * n, kernel_size=31, downsample=4, act=act
                    ),  # 512
                    ResBlockGeM(1 * n, 1 * n, kernel_size=31, act=act),
                ),
                nn.Sequential(
                    ResBlockGeM(
                        3 * n, 3 * n, kernel_size=31, downsample=4, act=act
                    ),  # 512
                    ResBlockGeM(3 * n, 3 * n, kernel_size=31, act=act),
                ),  # 128
            ]
        )
        self.conv2 = nn.Sequential(
            ResBlockGeM(6 * n, 4 * n, kernel_size=15, downsample=4, act=act),
            ResBlockGeM(4 * n, 4 * n, kernel_size=15, act=act),  # 128
            ResBlockGeM(4 * n, 8 * n, kernel_size=7, downsample=4, act=act),  # 32
            ResBlockGeM(8 * n, 8 * n, kernel_size=7, act=act),  # 8
        )
        self.head = nn.Sequential(
            AdaptiveConcatPool1d(),
            nn.Flatten(),
            nn.Linear(n * 8 * 2, nh),
            nn.BatchNorm1d(nh),
            nn.Dropout(ps),
            act,
            nn.Linear(nh, nh),
            nn.BatchNorm1d(nh),
            nn.Dropout(ps),
            act,
            nn.Linear(nh, 1),
        )

    def forward(self, x):
        x0 = [
            self.ex[0](x[:, 0].unsqueeze(1)),
            self.ex[0](x[:, 1].unsqueeze(1)),
            self.ex[1](x[:, 2].unsqueeze(1)),
        ]
        x1 = [
            self.conv1[0](x0[0]),
            self.conv1[0](x0[1]),
            self.conv1[1](x0[2]),
            self.conv1[2](torch.cat([x0[0], x0[1], x0[2]], 1)),
        ]
        x2 = torch.cat(x1, 1)
        return self.head(self.conv2(x2))


class Combined1D2D(nn.Module):
    def __init__(self, model1d, encoder="resnet18", emb_1d=128):
        super().__init__()
        self.model1d = model1d

        # Replace last linear layer to return a embedding of size emb_1d
        head = list(self.model1d.head.children())
        new_linear = nn.Linear(head[-1].in_features, emb_1d)
        self.model1d.head = nn.Sequential(*head[:-1] + [new_linear])

        self.model2d = timm.create_model(
            encoder,
            pretrained=True,
            num_classes=0,  # 0 = feature extraction
            in_chans=4,
        )

        # Find the embedding size of model2d
        o = self.model2d(torch.randn(2, 4, 224, 224))
        emb_2d = o.shape[-1]

        self.head = nn.Sequential(
            nn.Linear(emb_1d + emb_2d, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.spec_transform = Spectrogram.CQT1992v2(
            sr=2048,
            fmin=20,
            fmax=1000,
            hop_length=8,  # img width = sig_length / hop_length
            window="flattop",
            # Oversampling freq axis
            bins_per_octave=48,
            filter_scale=0.25,
        )

    def frequency_encoding(self, x):
        device = x.device
        bs, fbins, t = x.shape[0], x.shape[2], x.shape[3]
        freq_encoding = 2 * torch.arange(fbins, device=device) / fbins - 1  # -1 to +1
        freq_encoding = torch.stack([freq_encoding] * t, -1).unsqueeze(0)
        freq_encoding = torch.stack([freq_encoding] * bs)
        return torch.cat([x, freq_encoding], 1)

    def prepare_image(self, x):
        bs = x.shape[0]
        x_reshaped = x.reshape(-1, 4096)
        spec = self.spec_transform(x_reshaped)
        spec = spec.reshape(bs, 3, spec.shape[1], spec.shape[2])

        spec = standard_scaler(spec)
        spec = self.frequency_encoding(spec)
        return spec

    def forward(self, x):
        out_1d = self.model1d(x)
        out_2d = self.model2d(self.prepare_image(x))
        embedding = torch.cat([out_1d, out_2d], -1)
        return self.head(embedding)


if __name__ == "__main__":
    x = torch.randn(size=(32, 3, 4096))

    model_1d = ModelIafossV2()  # Load from checkpoint etc.
    model = Combined1D2D(model_1d, "resnet18")

    out = model(x)
    print(out.shape)
