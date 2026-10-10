from __future__ import annotations

from torch import nn

from meisenmeister.architectures.base_architecture import BaseArchitecture


def _conv3x3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv3d:
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet3D18(BaseArchitecture):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.stem_channels = 64
        self.inplanes = self.stem_channels

        self.conv1 = nn.Conv3d(
            in_channels,
            self.stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.stem_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * BasicBlock3D.expansion, num_classes)

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != out_channels * BasicBlock3D.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    out_channels * BasicBlock3D.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels * BasicBlock3D.expansion),
            )

        layers = [
            BasicBlock3D(
                self.inplanes,
                out_channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.inplanes = out_channels * BasicBlock3D.expansion

        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.inplanes, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)

    def get_grad_cam_target_layer(self) -> nn.Module:
        return self.layer4
