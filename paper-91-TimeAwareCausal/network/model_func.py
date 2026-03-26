"""
Adjust from DomainBed: https://github.com/facebookresearch/DomainBed/blob/main/domainbed/networks.py
"""
import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from engine.configs import Embeddings
# from torchvision.models import densenet121, DenseNet121_Weights


@Embeddings.register('toy_linear_fe')
class LinearFeatExtractor(nn.Module):
    """Just  an MLP"""

    def __init__(self, input_shape, output_dim, hidden_dim=512, depth=4, drop_rate=0.1):
        super(LinearFeatExtractor, self).__init__()
        self.input = nn.Linear(input_shape[-1], hidden_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.hiddens = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(depth - 2)])
        self.output = nn.Linear(hidden_dim, output_dim)
        self.n_outputs = output_dim

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


@Embeddings.register('mnist_cnn')
class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape, output_dim):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, output_dim, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, output_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.n_outputs = output_dim

    def forward(self, x):
        x = self.bn0(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn3(F.relu(self.conv4(x)))
        x = self.avgpool(x)
        x = x.view(len(x), -1)  # 128
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, arch, input_shape, pretrained=False, drop_rate=0.5):
        super(ResNet, self).__init__()
        if arch == 'resnet18':
            self.network = torchvision.models.resnet18(pretrained=pretrained)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=pretrained)
            self.n_outputs = 2048

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

@Embeddings.register('resnet18')
def resnet18(input_shape, output_dim, pretrained=False):
    """
    Constructs a ResNet-18 model.
    """
    model = ResNet('resnet18', input_shape, pretrained=pretrained)
    return model


@Embeddings.register('resnet50')
def resnet50(input_shape, output_dim, pretrained=False):
    """
    Constructs a ResNet-50 model.
    """
    model = ResNet('resnet50', input_shape, pretrained=pretrained)
    return model