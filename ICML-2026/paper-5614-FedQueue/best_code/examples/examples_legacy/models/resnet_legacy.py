import torch.nn as nn
import torch
from torchvision.models import resnet18 as resnet18base
from torchvision.models import ResNet as ResNetBase
from torchvision.models.resnet import BasicBlock, ResNet18_Weights


class ResNet(ResNetBase):
    # ResNet 18 Architecture Implementation to adapt grayscale and 28 X 28 pixel size input
    def __init__(self, block, layers, num_classes, grayscale):
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super().__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

        x = torch.flatten(x, 1)
        logits = self.fc(x)

        return logits


def resnet18(num_channel, num_classes=-1, pretrained=0):
    model = None

    if num_channel == 1:
        if num_classes < 0 or pretrained > 0:
            weights = ResNet18_Weights.verify(ResNet18_Weights.IMAGENET1K_V1)
            num_classes = len(weights.meta["categories"])
            model = ResNet(
                block=BasicBlock,
                layers=[2, 2, 2, 2],
                num_classes=num_classes,
                grayscale=False,
            )
            model.load_state_dict(weights.get_state_dict(progress=True))
        else:
            model = ResNet(
                block=BasicBlock,
                layers=[2, 2, 2, 2],
                num_classes=num_classes,
                grayscale=True,
            )

    else:
        if num_classes < 0 or pretrained > 0:
            model = resnet18base(pretrained=True)
        else:
            model = resnet18base(pretrained=False, num_classes=num_classes)

    return model
